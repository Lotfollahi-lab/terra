"""
Adapted from Assran, M. et al. Self-supervised learning from images with
a Joint-Embedding Predictive Architecture. Proc. IEEE Comput. Soc. Conf.
Comput. Vis. Pattern Recognit. 15619–15629 (2023);
https://github.com/facebookresearch/ijepa/blob/main/src/models/vision_transformer.py
(05.06.2024).
"""

from typing import Literal

import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from .adaln import AdaLN
from .rope2d import RoPE2D


class Attention(nn.Module):
    """
    Attention module used in transformer block, containing attention and
    projection layer.

    Parameters
    -----------
    dim:
        Number of dimensions.
    num_heads:
        Number of attention heads.
    qkv_bias:
        If 'True', include bias in query, key, and value layers.
    qk_scale:
        Scaling factor for query and key vectors.
    attn_drop:
        Dropout ratio in attention layer.
    proj_drop:
        Dropout ratio in projection layer.
    use_flash_attention:
        If `True`, use flash_attention.
    """
    def __init__(self,
                 dim: int,
                 num_heads: int = 8,
                 qkv_bias: bool = False,
                 qk_scale: float | None = None,
                 attn_drop: float=0.0,
                 proj_drop: float=0.0,
                 use_flash_attention: bool = True,
                 rope: RoPE2D | None = None,
                 ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Define layers
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_flash_attention = use_flash_attention
        # Optional 2D rotary position embedding. When set, q and k are
        # rotated based on the per-token (x, y) coordinates BEFORE the
        # SDPA call, so attention logits depend only on relative
        # position. The RoPE2D module is shared across all blocks of
        # the encoder/predictor that use it (one instance, many refs).
        self.rope = rope

    def forward(self,
                x: torch.Tensor,
                masks: torch.Tensor | None = None,
                coords: torch.Tensor | None = None,
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the attention module.

        Parameters
        -----------
        x:
            Input to the attention module with shape (B, N, C).
        masks:
            Mask applied to the attention vectors. Bool or additive
            float (e.g. ALiBi bias).
        coords:
            Per-token 2D coordinates of shape (B, N, 2). Required if
            RoPE is configured on this attention module; ignored
            otherwise. Special / padding tokens should carry (0, 0).

        Returns
        -----------
        x:
            Output of the attention module with shape (B, N, C).
        attn:
            Attention vector.
        """
        B, N, C = x.shape

        # Obtain query, key and value vectors
        qkv = self.qkv(x).reshape(
            B, N, 3, self.num_heads, C // self.num_heads).permute(
                2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply 2D rotary position embedding if configured. Sanitize
        # any non-finite sentinel values (e.g. -inf padding from
        # block-masking) to (0, 0) so the rotation is identity there.
        if self.rope is not None and coords is not None:
            coords_clean = torch.nan_to_num(
                coords, nan=0.0, posinf=0.0, neginf=0.0)
            q, k = self.rope(q, k, coords_clean)

        if self.use_flash_attention:
            # Memory-efficient SDPA with fp32 softmax accumulator.
            # The FLASH_ATTENTION backend (PyTorch's default on bf16
            # inputs) keeps the softmax in bf16, which loses precision
            # on the exp/sum step and noticeably hurts JEPA-style
            # training. The MATH backend matches the autocast-promoted
            # fp32 softmax of the manual path but is slow and
            # memory-hungry. EFFICIENT_ATTENTION + CUDNN_ATTENTION
            # give fp32 softmax accumulation while preserving
            # tile-wise memory savings. We prefer those over FLASH,
            # falling back to MATH so the kernel is always available.
            try:
                from torch.nn.attention import sdpa_kernel, SDPBackend
                _sdpa_backends = [
                    SDPBackend.CUDNN_ATTENTION,
                    SDPBackend.EFFICIENT_ATTENTION,
                    SDPBackend.MATH,
                ]
            except ImportError:
                print("Failed to import SDPBackend. Flash attention will be used without backend selection, which may hurt training stability. Consider upgrading to PyTorch 2.3+ for better flash attention support.")
                # PyTorch < 2.3 — no backend selector API. Fall back
                # to plain SDPA and accept whatever backend it picks.
                sdpa_kernel = None
                _sdpa_backends = None

            if sdpa_kernel is not None:
                with sdpa_kernel(_sdpa_backends):
                    if masks is not None:
                        x = nn.functional.scaled_dot_product_attention(
                            q, k, v, attn_mask=masks, scale=self.scale)
                    else:
                        x = nn.functional.scaled_dot_product_attention(
                            q, k, v, scale=self.scale)
            else:
                if masks is not None:
                    x = nn.functional.scaled_dot_product_attention(
                        q, k, v, attn_mask=masks, scale=self.scale)
                else:
                    x = nn.functional.scaled_dot_product_attention(
                        q, k, v, scale=self.scale)
            attn = None
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            if masks is not None:
                if masks.dtype in (torch.bool, torch.uint8):
                    # Boolean / 0-1 mask: -inf wherever entry is False/0.
                    attn = attn.masked_fill(masks == 0, float('-inf'))
                else:
                    # Float additive bias (ALiBi). Add directly to
                    # pre-softmax logits.
                    attn = attn + masks
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn


class Block(nn.Module):
    """
    Transformer block used in the encoder and predictor.

    Parameters
    -----------
    dim:
        Number of input and output dimenions of the transformer block.
    num_heads:
        Number of attention heads.
    mlp_ratio:
        Ratio to determine number of hidden dimensions in MLP module
        compared to input and output dimensions.
    qkv_bias:
        If 'True', include bias in query, key, and value layers of
        Attention module.
    qk_scale:
        Scaling factor for query and key vectors of Attention module.
    drop:
        Dropout ratio in projection layer of Attention module and in MLP
        module.
    attn_drop:
        Dropout ratio in attention layer of Attention module.
    act_layer:
        Activation layer used in MLP module.
    use_layer_norm:
        If `True`, use LayerNorm, else use dynamic tanh.
    norm_layer:
        Normalization layer.
    use_flash_attention:
        If `True`, use flash_attention.
    """
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = False,
                 qk_scale: float | None = None,
                 drop: float = 0.0,
                 attn_drop: float=0.0,
                 act_layer: nn.modules.activation=nn.GELU,
                 use_layer_norm: bool=True,
                 norm_layer: nn.modules.normalization=nn.LayerNorm,
                 use_flash_attention: bool=True,
                 rope: RoPE2D | None = None,
                 cond_dim: int | None = None,
                 ):
        super().__init__()
        # When `cond_dim` is provided the two norms become AdaLN (DiT-
        # style), modulated by a per-cell conditioning vector (batch
        # embedding). Zero-initialized so behavior at step 0 is
        # identical to plain LayerNorm -- see adaln.AdaLN docstring.
        # AdaLN always uses LayerNorm internally (not DyT); a future
        # extension could add a Dyt-based AdaLN if needed.
        self.uses_adaln = cond_dim is not None

        def _make_norm():
            if self.uses_adaln:
                return AdaLN(dim, cond_dim=cond_dim)
            if use_layer_norm:
                return norm_layer(dim)
            return DyT(dim)

        # Preserve the original submodule registration order
        # (norm1, attn, norm2, mlp) so existing checkpoints load and
        # named_parameters() ordering is unchanged for non-AdaLN runs.
        self.norm1 = _make_norm()
        self.attn = Attention(dim,
                              num_heads=num_heads,
                              qkv_bias=qkv_bias,
                              qk_scale=qk_scale,
                              attn_drop=attn_drop,
                              proj_drop=drop,
                              use_flash_attention=use_flash_attention,
                              rope=rope)
        self.norm2 = _make_norm()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)

    def forward(self,
                x: torch.Tensor,
                return_attention: bool = False,
                masks: torch.Tensor | None = None,
                coords: torch.Tensor | None = None,
                cond: torch.Tensor | None = None,
                ) -> torch.Tensor:
        """
        Forward pass of the transformer block.

        Parameters
        -----------
        x:
            Input to the transformer block with shape (B, N, D).
        return_attention:
            If `True`, return attention vector instead of transformer
            block output.
        masks:
            Mask used in Attention module.
        coords:
            Per-token 2D coordinates of shape (B, N, 2), forwarded to
            the Attention module. Only used if RoPE is configured.
        cond:
            Per-cell conditioning embedding of shape (B, cond_dim),
            used by AdaLN to modulate the two layer norms. Required
            when this block was constructed with ``cond_dim != None``
            and ignored otherwise.

        Returns
        -----------
        x:
            Output of the transformer block with shape (B, N, D).
        """
        # Branch the norm call signature based on whether this block
        # was built with AdaLN. Cheap and explicit -- avoids a runtime
        # type check on every forward.
        if self.uses_adaln:
            if cond is None:
                raise RuntimeError(
                    "Block was constructed with cond_dim != None (AdaLN) "
                    "but `cond` was not provided to forward().")
            h1 = self.norm1(x, cond)
        else:
            h1 = self.norm1(x)
        y, attn = self.attn(h1, masks=masks, coords=coords)
        if return_attention:
            return attn
        x = x + y
        if self.uses_adaln:
            x = x + self.mlp(self.norm2(x, cond))
        else:
            x = x + self.mlp(self.norm2(x))

        return x


class ClassificationModel(nn.Module):
    """
    Add a classification head to a base model. Adds a fully connected layer or a multi-layer perceptron (MLP)
    for classification based on the output of a base model.
    """

    def __init__(self,
                 base_model: nn.Module,
                 gt_type: Literal['rank', 'counts'],
                 num_classes: int, use_mlp: bool = False, hidden_dim: int = 512):
        """
        Initialize the classification head.

        Parameters
        -----------
        base_model:
            The base model whose output is used for classification.
        num_classes:
            The number of output classes for classification.
        use_mlp:
            Whether to use an MLP head (default is False, which uses a simple linear layer).
        hidden_dim:
            The dimension of the hidden layer in the MLP (default is 512).
        """
        super(ClassificationModel, self).__init__()

        self.base_model = base_model
        self.gt_type = gt_type

        if use_mlp:
            # Using MLP with one hidden layer
            self.classification_head = nn.Sequential(
                nn.Linear(base_model.output_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_classes)
            )
        else:
            # Using a simple linear layer
            self.classification_head = nn.Linear(
                base_model.output_dim, num_classes)

    def forward(self, **base_model_kwargs) -> torch.Tensor:
        """
        Forward pass through the classification head.

        Parameters:
        - x (Tensor): The input tensor (feature vector) from the base model.

        Returns:
        - Tensor: The class logits.
        """
        h, _= self.base_model(**base_model_kwargs)

        # Normalize over feature dim
        h = F.layer_norm(h, (h.size(-1),))
        
        logits = self.classification_head(h)
        return logits


class DyT(nn.Module):
    """
    Dynamic tanh module.

    Parameters
    -----------
    num_features:
        Number of features.
    alpha_init_value:
        Initial value for alpha.
    """
    def __init__(self,
                 num_features: int,
                 alpha_init_value: float = 0.5
                 ):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.alpha * x)
        return x * self.weight + self.bias


class MLP(nn.Module):
    """
    MLP module used in transformer block.

    Parameters
    -----------
    in_features:
        Number of input features.
    hidden_features:
        Number of hidden features. If not specified, equals number of
        input features.
    out_features:
        Number of output features. If not specified, equals number of
        input features.
    bias
        If `True`, include a bias in linear layers.
    act_layer:
        Activation layer after first fully connected layer.
    drop:
        Probability for dropout.
    """
    def __init__(self,
                 in_features: int, 
                 hidden_features: int | None = None,
                 out_features: int | None = None,
                 bias: bool = True,
                 act_layer: nn.modules.activation = nn.GELU,
                 drop: float = 0.0,
                 ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self,
                x: torch.Tensor
                ) -> torch.Tensor:
        """
        Forward pass of the MLP module.

        Parameters
        -----------
        x:
            Input to the MLP module with shape (B, N, I).

        Returns
        -----------
        x:
            Output of the MLP module with shape (B, N, O), by default
            O=I.
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class ValueEmbWeightsProjection(nn.Module):
    def __init__(self,
                 dim: int = 100
                 ):
        """
        Project counts to value embedding weights.

        Parameters
        -----------
        dim:
            Dimensionality of the value embedding.        
        """
        super().__init__()
        self.linear1 = nn.Linear(1, dim)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
        self.linear2 = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.leaky_relu(x)
        out = self.linear2(x)
        out = x + out # residual connection
        out = self.softmax(out)
        
        return out