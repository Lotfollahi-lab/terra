"""
Adapted from Assran, M. et al. Self-supervised learning from images with
a Joint-Embedding Predictive Architecture. Proc. IEEE Comput. Soc. Conf.
Comput. Vis. Pattern Recognit. 15619–15629 (2023);
https://github.com/facebookresearch/ijepa/blob/main/src/models/vision_transformer.py
(05.06.2024).
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:  # PyTorch < 2.3: no backend selection
    SDPBackend = sdpa_kernel = None

from .adaln import AdaLN
from .rope2d import RoPE2D

from terra.utils.embedding import compute_mean_unmasked_emb

logger = logging.getLogger(__name__)


# Scaled-dot-product-attention (SDPA) backend presets, selected with the
# `attention_backend` config option. Each preset is the set of backends SDPA
# may use. Within that set PyTorch takes the first backend that supports the
# inputs in its own priority order (`torch._C._get_sdp_priority_order()`,
# FLASH > EFFICIENT > MATH > CUDNN in PyTorch 2.11), NOT in the order listed
# here (`sdpa_kernel` only honours the order with `set_priority=True`, which
# TERRA does not use, to keep trained models' numerics).
#
#   * 'auto' (the default) excludes FLASH_ATTENTION: in practice the
#     memory-efficient kernel, or MATH where it cannot run. CUDNN_ATTENTION is
#     enabled but ranks after MATH, which supports every input, so it is not
#     reached with the default priority order.
#   * 'flash' adds FLASH_ATTENTION (lowest memory, largest usable batch size).
#     Flash cannot take an attention mask, so masked attention (e.g. every
#     inference pass) still uses the memory-efficient kernel. Kernels round
#     differently, so results can differ slightly between presets.
#   * 'efficient' / 'cudnn' / 'math' pin a single backend (SDPA raises if it
#     does not support the inputs, e.g. 'efficient' and 'cudnn' on CPU).
#   * MATH materializes the full attention matrix: highest memory use
#     (smallest usable batch size), but always available.
#   * 'torch' does not constrain the backend (PyTorch picks among all). This
#     is how TERRA-96M and TERRA-112M were trained, before the presets
#     existed: FlashAttention for the unmasked attention of the context
#     encoder and predictor, the memory-efficient kernel for masked attention.
_ATTENTION_BACKEND_PRESETS = {
    'auto': ('CUDNN_ATTENTION', 'EFFICIENT_ATTENTION', 'MATH'),
    'flash': ('FLASH_ATTENTION', 'CUDNN_ATTENTION', 'EFFICIENT_ATTENTION',
              'MATH'),
    'efficient': ('EFFICIENT_ATTENTION',),
    'cudnn': ('CUDNN_ATTENTION',),
    'math': ('MATH',),
    'torch': (),
}

ATTENTION_BACKENDS = tuple(_ATTENTION_BACKEND_PRESETS)


def resolve_sdpa_backends(attention_backend: str = 'auto'):
    """Resolve an ``attention_backend`` preset to a list of SDPA backends.

    Parameters
    ----------
    attention_backend:
        One of :data:`ATTENTION_BACKENDS`. See
        ``_ATTENTION_BACKEND_PRESETS`` for the memory/precision trade-off
        of each preset.

    Returns
    -------
    backends:
        List of ``torch.nn.attention.SDPBackend`` members to pass to
        ``sdpa_kernel``, or ``None`` to let PyTorch choose the backend
        (``'torch'``, or a PyTorch build without the backend selector).
    """
    if attention_backend not in _ATTENTION_BACKEND_PRESETS:
        raise ValueError(
            f"Unknown attention_backend '{attention_backend}'. "
            f"Expected one of {list(ATTENTION_BACKENDS)}.")

    names = _ATTENTION_BACKEND_PRESETS[attention_backend]
    if not names:  # 'torch' -> no backend constraint
        return None

    if SDPBackend is None:
        logger.warning(
            "torch.nn.attention.SDPBackend is unavailable (PyTorch < 2.3), so "
            "attention_backend='%s' cannot be enforced. Falling back to "
            "PyTorch's default backend selection, which differs from the "
            "backends TERRA models were trained with. Consider upgrading to "
            "PyTorch 2.3+.",
            attention_backend)
        return None

    # Skip backends the installed PyTorch does not know about (e.g.
    # CUDNN_ATTENTION was added after the selector API itself).
    backends = [getattr(SDPBackend, name) for name in names
                if hasattr(SDPBackend, name)]
    if not backends:
        logger.warning(
            "None of the backends %s are available in this PyTorch build; "
            "falling back to PyTorch's default backend selection.",
            list(names))
        return None
    return backends


def describe_attention(use_flash_attention: bool,
                       attention_backend: str = 'auto') -> str:
    """Return a human-readable description of the attention configuration."""
    if not use_flash_attention:
        return ("manual (eager) attention with fp32 softmax "
                "[use_flash_attention=False]; attention_backend is ignored")

    backends = resolve_sdpa_backends(attention_backend)
    if backends is None:
        return (f"SDPA with attention_backend='{attention_backend}' "
                "(backend: PyTorch default, unconstrained)")
    return (f"SDPA with attention_backend='{attention_backend}' (enabled "
            "backends in the order PyTorch tries them: "
            f"{' > '.join(b.name for b in _in_priority_order(backends))})")


def _in_priority_order(backends: list) -> list:
    """Sort SDPA backends by PyTorch's backend priority order (the order in
    which SDPA tries the enabled backends); unchanged if it is unknown."""
    try:
        order = [int(b) for b in torch._C._get_sdp_priority_order()]
    except (AttributeError, RuntimeError, TypeError):
        return list(backends)
    return sorted(backends, key=lambda b: (order.index(int(b))
                                           if int(b) in order else len(order)))


def log_attention_config(use_flash_attention: bool,
                         attention_backend: str = 'auto') -> None:
    """Log the attention configuration, with memory/precision caveats."""
    logger.info("Attention: %s.",
                describe_attention(use_flash_attention, attention_backend))
    if not use_flash_attention:
        logger.warning(
            "use_flash_attention=False materializes the full attention "
            "matrix, which uses considerably more memory (smaller usable "
            "batch size) than the SDPA path.")
        return
    if attention_backend == 'flash':
        logger.warning(
            "attention_backend='flash' enables FlashAttention (lowest memory, "
            "largest batch size) for unmasked attention. Attention kernels "
            "round differently, so results can differ slightly from runs with "
            "the default 'auto', which excludes it.")
    elif attention_backend == 'math':
        logger.warning(
            "attention_backend='math' materializes the full attention "
            "matrix and will substantially reduce the usable batch size.")


def guard_fully_masked_rows(masks: torch.Tensor,
                            ) -> tuple[torch.Tensor, torch.Tensor]:
    """Let attention query rows without any valid key attend to one dummy key.

    A query row whose keys are all masked gives NaN with the eager attention
    (softmax over all -inf) and, depending on the PyTorch version and kernel,
    with SDPA. Such rows attend to the first key instead, so the attention
    stays finite in the forward and backward pass; the caller zeroes their
    output with the returned `row_valid`. Rows with at least one valid key
    keep their mask entries unchanged. No host synchronization.

    Parameters
    -----------
    masks:
        Bool or 0/1 mask (`False`/0 = masked) or additive float mask (-inf =
        masked), broadcastable to (B, H, N_q, N_k).

    Returns
    -----------
    masks:
        The mask with key 0 unmasked in the fully masked rows (same shape and
        dtype).
    row_valid:
        Bool tensor of shape (..., N_q, 1), `False` for the fully masked rows.
    """
    if masks.dtype in (torch.bool, torch.uint8):
        valid = masks != 0
    else:
        valid = ~torch.isneginf(masks)
    row_valid = valid.any(dim=-1, keepdim=True)
    first_key = torch.arange(masks.shape[-1], device=masks.device) == 0
    dummy = ~row_valid & first_key
    if masks.dtype in (torch.bool, torch.uint8):
        masks = masks | dummy.to(masks.dtype)
    else:
        masks = masks.masked_fill(dummy, 0.0)
    return masks, row_valid


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
        If `True`, use PyTorch's scaled dot-product attention (SDPA)
        instead of the manual (eager) attention implementation.
    attention_backend:
        Which SDPA backend(s) may be used when `use_flash_attention` is
        `True`; one of :data:`ATTENTION_BACKENDS` (see
        ``_ATTENTION_BACKEND_PRESETS``). `'auto'` (default) excludes
        FlashAttention, `'flash'` allows it (lowest memory).
    zero_fully_masked_attention_rows:
        If `True`, a query row whose keys are all masked outputs zeros (with
        finite gradients) instead of NaN on every attention path; all other
        rows are unchanged. See :func:`guard_fully_masked_rows`.
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
                 attention_backend: str = 'auto',
                 zero_fully_masked_attention_rows: bool = False,
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
        # Resolve the SDPA backend preference once, at construction time,
        # rather than on every forward pass. `None` means "let PyTorch
        # choose" (attention_backend='torch', or PyTorch < 2.3).
        self.attention_backend = attention_backend
        # Always resolve (so an invalid name fails fast, even on the eager
        # path), but only apply the preference when SDPA is actually used.
        _backends = resolve_sdpa_backends(attention_backend)
        self._sdpa_backends = _backends if use_flash_attention else None
        self.zero_fully_masked_attention_rows = (
            zero_fully_masked_attention_rows)
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

        # Query rows without any valid key (e.g. a cell whose context is
        # only padding) attend to a dummy key, and their output is zeroed
        # below, so they stay finite whatever the attention kernel
        row_valid = None
        if masks is not None and self.zero_fully_masked_attention_rows:
            masks, row_valid = guard_fully_masked_rows(masks)

        if self.use_flash_attention:
            # SDPA, restricted to the backends selected by
            # `attention_backend` (resolved in __init__). SDPA uses the
            # first enabled backend that supports these inputs, in
            # PyTorch's priority order -- see `_ATTENTION_BACKEND_PRESETS`.
            # `None` means unconstrained: PyTorch picks the backend itself.
            _sdpa_backends = self._sdpa_backends

            if _sdpa_backends is not None:
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
            if row_valid is not None:
                x = x.masked_fill(~row_valid, 0.0)
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
            if row_valid is not None:
                # Zero the output rather than the weights the matmul keeps
                # for the backward pass, so no second (B, H, N, N) tensor
                # is saved; the returned weights are zeroed as well
                x = x.masked_fill(~row_valid, 0.0)
                attn = attn.masked_fill(~row_valid, 0.0)
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
        If `True`, use PyTorch's scaled dot-product attention (SDPA)
        instead of the manual (eager) attention implementation.
    attention_backend:
        SDPA backend preference; one of :data:`ATTENTION_BACKENDS`.
    zero_fully_masked_attention_rows:
        If `True`, query rows without any valid key output zeros instead of
        NaN in the Attention module.
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
                 attention_backend: str = 'auto',
                 zero_fully_masked_attention_rows: bool = False,
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
                              rope=rope,
                              attention_backend=attention_backend,
                              zero_fully_masked_attention_rows=(
                                  zero_fully_masked_attention_rows))
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

    def __init__(
            self,
            base_model: nn.Module,
            num_classes: int,
            use_mlp: bool = False,
            hidden_dim: int = 512
        ):
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
        self.embed_dim = self.base_model.backbone.embed_dim

        if use_mlp:
            # Using MLP with one hidden layer
            self.classification_head = nn.Sequential(
                nn.Linear(self.embed_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_classes)
            )
        else:
            # Using a simple linear layer
            self.classification_head = nn.Linear(self.embed_dim, num_classes)

    def forward(
            self,
            udata: dict,
            masks_attention: torch.Tensor,
            selection_mask: torch.Tensor = None,
        ) -> torch.Tensor:
        """
        Forward pass through the classification head.

        Parameters
        ----------
        udata : dict
            The input dictionary containing the batch data.
        masks_attention : Tensor
            The attention masks.
        selection_mask : Tensor
            The selection (cell or neighborhood) mask.

        Returns
        -------
        Tensor
            The class logits.
        """
        # h.shape = (<batch_size>, <seq_len>, <embedding_dim>)
        h, _= self.base_model(batch=udata, masks_attention=masks_attention)

        # Aggregate gene embeddings into cell/neighborhood embeddings
        # h.shape = (<batch_size>, <embedding_dim>)
        h = compute_mean_unmasked_emb(h, selection_mask)

        # Normalize over logits dimension
        h = F.layer_norm(h, (h.size(-1),))

        # Project cell/neighborhood embeddings into logits
        # logits.shape = (<batch_size>, <num_classes>)
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