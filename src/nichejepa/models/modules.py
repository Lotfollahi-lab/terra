from typing import Optional, Tuple

import torch
import torch.nn as nn

from .utils import drop_path


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
    """
    def __init__(self,
                 dim: int,
                 num_heads: int=8,
                 qkv_bias: bool=False,
                 qk_scale: Optional[float]=None,
                 attn_drop: float=0.0,
                 proj_drop: float=0.0,
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

    def forward(self,
                x: torch.Tensor,
                masks: Optional[torch.Tensor]=None,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
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
            B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute and mask attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if masks is not None:
            attn = attn.masked_fill(masks == 0, float('-inf'))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Compute dot product of attention and value vectors and apply
        # projection
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
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
        Ratio to determine number of hidden dimensions in MLP module compared to
        input and output dimensions.
    qkv_bias:
        If 'True', include bias in query, key, and value layers of Attention
        module.
    qk_scale:
        Scaling factor for query and key vectors of Attention module.
    drop:
        Dropout ratio in projection layer of Attention module and in MLP module.
    attn_drop:
        Dropout ratio in attention layer of Attention module.
    drop_path:
        Probability for dropping paths in Drop Path module.
    act_layer:
        Activation layer used in MLP module.
    norm_layer:
        Normalization layer.
    """
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 mlp_ratio: float=4.,
                 qkv_bias: bool=False,
                 qk_scale: Optional[float]=None,
                 drop: float=0.0,
                 attn_drop: float=0.0,
                 drop_path: float=0.0,
                 act_layer: nn.modules.activation=nn.GELU,
                 norm_layer: nn.modules.normalization=nn.LayerNorm,
                 ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim,
                              num_heads=num_heads,
                              qkv_bias=qkv_bias,
                              qk_scale=qk_scale,
                              attn_drop=attn_drop,
                              proj_drop=drop)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)

    def forward(self,
                x: torch.Tensor,
                return_attention: bool=False,
                masks: Optional[torch.Tensor]=None,
                ) -> torch.Tensor:
        """
        Forward pass of the transformer block.

        Parameters
        -----------
        x:
            Input to the transformer block with shape (B, N, D).
        return_attention:
            If 'True', return attention vector instead of transformer block
            output.
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
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class DropPath(nn.Module):
    """
    DropPath module to drop paths per observation, applied in main path of
    residual blocks of transformer blocks, with stochastically increasing drop
    path rate per depth.

    Parameters
    -----------
    drop_prob:
        Probability for dropping paths.    
    """
    def __init__(self,
                 drop_prob: float=0.0,
                 ):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self,
                x: torch.Tensor
                ) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class MLP(nn.Module):
    """
    MLP module used in transformer block.

    Parameters
    -----------
    in_features:
        Number of input features.
    hidden_features:
        Number of hidden features. If not specified, equals number of input
        features.
    out_features:
        Number of output features. If not specified, equals number of input
        features.
    act_layer:
        Activation layer after first fully connected layer.
    drop:
        Probability for dropout.
    """
    def __init__(self,
                 in_features: int, 
                 hidden_features: Optional[int]=None,
                 out_features: Optional[int]=None,
                 act_layer: nn.modules.activation=nn.GELU,
                 drop: float=0.0,
                 ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
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
            Output of the MLP module with shape (B, N, O), by default O=I.
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x