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

    def forward(self,
                x: torch.Tensor,
                masks: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the attention module.

        Parameters
        -----------
        x:
            Input to the attention module with shape (B, N, C).
        masks:
            Mask applied to the attention vectors.

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

        if self.use_flash_attention:
            if masks is not None:
                x = nn.functional.scaled_dot_product_attention(
                    q, k, v, attn_mask=masks, scale=self.scale)
                attn=None
            else:
                x = nn.functional.scaled_dot_product_attention(
                    q, k, v, scale=self.scale)
                attn=None
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            if (masks is not None):
                attn = attn.masked_fill(masks == 0, float('-inf'))
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
                 ):
        super().__init__()
        if use_layer_norm:
            self.norm1 = norm_layer(dim)
        else:
            self.norm1 = DyT(dim)
        self.attn = Attention(dim,
                              num_heads=num_heads,
                              qkv_bias=qkv_bias,
                              qk_scale=qk_scale,
                              attn_drop=attn_drop,
                              proj_drop=drop,
                              use_flash_attention=use_flash_attention)
        if use_layer_norm:
            self.norm2 = norm_layer(dim)
        else:
            self.norm2 = DyT(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)

    def forward(self,
                x: torch.Tensor,
                return_attention: bool = False,
                masks: torch.Tensor | None = None,
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

        Returns
        -----------
        x:
            Output of the transformer block with shape (B, N, D).       
        """
        y, attn = self.attn(self.norm1(x),
                            masks=masks)
        if return_attention:
            return attn
        x = x + y
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