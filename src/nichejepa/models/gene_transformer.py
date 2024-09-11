"""
Adapted from Assran, M. et al. Self-supervised learning from images with a
Joint-Embedding Predictive Architecture. Proc. IEEE Comput. Soc. Conf. Comput.
Vis. Pattern Recognit. 15619–15629 (2023);
https://github.com/facebookresearch/ijepa/blob/main/src/models/vision_transformer.py
(05.06.2024).
"""

import math
from functools import partial
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from ..masks.utils import apply_masks
from ..utils.tensors import (repeat_interleave_batch,
                             trunc_normal_)


def get_1d_sincos_pos_embed(embed_dim: int,
                            seq_len: int,
                            cls_token: bool=False
                            ) -> np.ndarray:
    """
    Retrieve 1D sin cos positional embedding based on number of positions and
    presence of <cls> token.

    Parameters
    -----------
    embed_dim:
        Output dimension of the positional embedding (for each position). Has to
        be divisible by 2.
    seq_len:
        Number of positions to be embedded.
    cls_token:
        If 'True', considers a <cls> token and assigns a positional embedding of
        0s.

    Returns
    -----------
    pos_embed:
        The positional embedding with shape (1+SEQ_LEN, EMBED_DIM) w/ <cls>
        token or shape (SEQ_LEN, EMBED_DIM) w/o <cls> token.
    """
    pos = np.arange(seq_len, dtype=float)
    pos_embed = get_1d_sincos_pos_embed_from_pos(embed_dim, pos)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed],
                                   axis=0)

    return pos_embed


def get_1d_sincos_pos_embed_from_pos(embed_dim: int,
                                     pos: np.ndarray
                                     ) -> np.ndarray:
    """
    Retrieve 1D sin cos positional embedding from an array of positions/sequence
    index.

    Parameters
    -----------
    embed_dim:
        Output dimension of the positional embedding (for each position). Has
        to be divisible by 2.
    pos:
        An array containing the positions to be embedded with shape (SEQ_LEN,).
        
    Returns
    -----------
    pos_emb:
        The positional embedding with shape (SEQ_LEN, EMBED_DIM).
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega # shape (EMBED_DIM/2,)

    pos = pos.reshape(-1) # shape (SEQ_LEN,)
    out = np.einsum('m,d->md', pos, omega) # shape (SEQ_LEN, EMBED_DIM/2);
                                           # outer product

    emb_sin = np.sin(out) # shape (SEQ_LEN, EMBED_DIM/2)
    emb_cos = np.cos(out) # shape (SEQ_LEN, EMBED_DIM/2)

    pos_emb = np.concatenate([emb_sin, emb_cos], axis=1) # shape (SEQ_LEN,
                                                         # EMBED_DIM)

    return pos_emb


def drop_path(x: torch.Tensor,
              drop_prob: float=0.,
              training: bool=False
              ) -> torch.Tensor:
    """
    Helper function for forward pass of DropPath module.

    Parameters
    -----------
    x:
        Input to DropPath module forward pass.
    drop_prob:
        Probability for dropping paths.
    training:
        If 'True', do not drop paths.

    Returns
    -----------
    output:
        Output of DropPath module forward pass, with some paths dropped (set to
        0) and others scaled.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    
    # Create binary random tensor
    shape = (x.shape[0],) + (1,) * (x.ndim - 1) # works with tensors of diff dim
    random_tensor = keep_prob + torch.rand(shape,
                                           dtype=x.dtype,
                                           device=x.device)
    random_tensor.floor_()

    # Drop some paths by setting them to 0 and scale rest to keep sum consistent
    output = x.div(keep_prob) * random_tensor

    return output


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
                 drop_prob: float=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
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
                 drop: float=0.
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
                 attn_drop: float=0.,
                 proj_drop: float=0.
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
                masks: Optional[torch.Tensor]=None
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
                 drop: float=0.,
                 attn_drop: float=0.,
                 drop_path: float=0.,
                 act_layer: nn.modules.activation=nn.GELU,
                 norm_layer: nn.modules.normalization=nn.LayerNorm
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
                masks: Optional[torch.Tensor]=None
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


class GeneTransformerEncoder(nn.Module):
    """
    GeneTransformerEncoder class to encode contexts or targets.
    
    Parameters
    -----------
    vocab_size:
        Size of the token vocabulary. Includes <pad> token.
    seq_len:
        Length of the token sequences (w/o <cls> token).
    has_cls:
        If 'True', sequences include a <cls> token at the start.
    pos_learnable:
        If 'True', positional embeddings are learnable, otherwise use sin cos
        positional embeddings.
    embed_dim:
        Dimension of the output embedding.
    predictor_embed_dim:
        Dimension of the embedding of the predictor.
    depth:
        Number of transformer blocks in the encoder.
    predictor_depth:
        Number of transformer blocks in the predictor.
    num_heads:
        Number of attention heads in the Attention module.
    mlp_ratio:
        Ratio to determine number of hidden dimensions in MLP module compared to
        input and output dimensions.
    qkv_bias:
        If 'True', include bias in query, key, and value layers of Attention
        module.
    qk_scale:
        Scaling factor for query and key vectors of Attention module.
    drop:
        Dropout ratio in projection layer of Attention module and in layers of
        MLP module.
    attn_drop:
        Dropout ratio in attention layer of Attention module.
    drop_path_rate:
        Probability for dropping paths in DropPath module.
    norm_layer:
        Normalization layer.
    init_std:
        Standard deviation for weight initialization.
    """
    def __init__(self,
                 vocab_size: int,
                 seq_len: int,
                 has_cls: bool=True,
                 pos_learnable: bool=False,
                 embed_dim: int=768,
                 depth: int=12,
                 predictor_embed_dim: int=384,
                 predictor_depth: int=12,
                 num_heads: int=12,
                 mlp_ratio: float=4.0,
                 qkv_bias: bool=True,
                 qk_scale: Optional[float]=None,
                 drop_rate: float=0.0,
                 attn_drop_rate: float=0.0,
                 drop_path_rate: float=0.0,
                 norm_layer: nn.modules.normalization=nn.LayerNorm,
                 init_std: float=0.02,
                 **kwargs
                 ):
        super().__init__()
        self.seq_len = seq_len
        self.has_cls = has_cls
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.init_std = init_std
        
        # Initialize gene embeddings
        self.gene_embed = nn.Embedding(
            vocab_size + (1 if self.has_cls else 0), # vocab_size incl. <pad>
            embed_dim,
            padding_idx=0)
                                          
        # Initialize segment embeddings (to differentiate cell and neighborhood
        # gene tokens)
        self.seg_embed = nn.Embedding(2 + 1, # incl. <pad>
                                      embed_dim,
                                      padding_idx=0)

        # Retrieve positional embeddings
        if pos_learnable:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, seq_len + (1 if has_cls else 0), embed_dim),
                requires_grad=True)
            trunc_normal_(self.pos_embed, std=self.init_std)
            if has_cls:
                self.pos_embed.data[0, 0, :] = 0
        else:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, seq_len + (1 if has_cls else 0), embed_dim),
                requires_grad=False)
            pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1],
                                                self.seq_len,
                                                cls_token=has_cls)
            self.pos_embed.data.copy_(
                torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Define decaying drop path rate (higher drop rate in deeper blocks)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Initialize encoder blocks and norm layer
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Initialize weights of layers
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        """
        Helper function to scale initialized layer weights.
        """
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        """
        Helper function to initialize layer weights.
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self,
                x: torch.Tensor,
                seg_label: torch.Tensor,
                masks: Optional[Union[list, torch.Tensor]]=None
                ) -> torch.Tensor:
        """
        Run encoder forward pass on a batch of input token sequences. For each 
        observation in the batch return only embeddings for tokens included in
        the masks.

        Parameters
        -----------
        x:
            Tensor containing input tokens with shape (BATCH_SIZE, SEQ_LEN) if
            no <cls> token is included and (BATCH_SIZE, SEQ_LEN+1) otherwise. 
        seg_label:
            Tensor containing segment labels to differentiate between cell and
            neighborhood gene tokens with shape (BATCH_SIZE, SEQ_LEN) if no
            <cls> token is included and (BATCH_SIZE, SEQ_LEN+1) otherwise.
        masks:
            List of N_MASKS tensors containing indices (within the sequence) of
            tokens to keep with shape (BATCH_SIZE, MASK_SIZE).

        Returns
        -----------
        x:
            Embeddings of input tokens included in the masks with shape (
            BATCH_SIZE * N_MASKS, MIN_MASK_SIZE, EMBED_DIM), where MIN_MASK_SIZE
            is minimum mask size in the batch.    
        """
        # Format masks
        if masks is not None:
            if not isinstance(masks, list):
                masks = [masks]

        # Get gene embeddings for sequence of gene tokens
        x = self.gene_embed(x)
        B, N, D = x.shape # B: BATCH_SIZE, N: SEQ_LEN (+1 if <cls>),
                          # D: EMBED_DIM
        
        # Add positional and segment embeddings to gene embedding
        x = x + self.pos_embed + self.seg_embed(seg_label)
              
        # Mask token embeddings if masks are provided
        if masks is not None:
            x = apply_masks(x, masks)
        
        # Run forward prop
        for i, blk in enumerate(self.blocks):
            x = blk(x)
        if self.norm is not None:
            x = self.norm(x)

        return x

    @torch.no_grad()
    def return_pos_emb(self,
                       x: torch.Tensor
                       ) -> torch.Tensor:
        """
        Return the positional embeddings for a batch of input token sequences.

        Parameters
        -----------
        x:
            Tensor containing input tokens with shape (BATCH_SIZE, SEQ_LEN) if
            no <cls> token is included and (BATCH_SIZE, SEQ_LEN+1) otherwise.

        Returns
        -----------
        pos_embed:
            Tensor containing the positional embeddings repeated across the
            batch dimension with shape (BATCH_SIZE, SEQ_LEN, EMBED_DIM) if
            no <cls> token is included and (BATCH_SIZE, SEQ_LEN+1, EMBED_DIM)
            otherwise.
        """
        # Repeat the positional embeddings across the batch dimension to match
        # the input shape
        pos_embed = self.pos_embed.repeat(x.shape[0], 1, 1)

        return pos_embed

    @torch.no_grad()
    def return_gene_emb(self,
                        x: torch.Tensor
                        ) -> torch.Tensor:
        """
        Return the gene embeddings for a batch of input tokens.

        Parameters
        -----------
        x:
            Tensor containing input tokens with shape (BATCH_SIZE, SEQ_LEN).

        Returns
        -----------
        gene_embed:
            Tensor containing the gene embeddings with shape (BATCH_SIZE,
            SEQ_LEN, EMBED_DIM).
        """
        # Retrieve gene embeddings
        gene_embeds = self.gene_embed(x)

        return gene_embeds

    @torch.no_grad()
    def return_multi_layer_emb(self,
                               x: torch.Tensor,
                               seg_label: torch.Tensor,
                               masks: Optional[Union[list, torch.Tensor]]=None
                               ) -> list:
        """
        Run encoder forward pass on a batch of input token sequences, applying
        masks if provided, and return a list containing the embeddings after
        each block.

        Parameters
        -----------
        x:
            Tensor containing input tokens with shape (BATCH_SIZE, SEQ_LEN) if
            no <cls> token is included and (BATCH_SIZE, SEQ_LEN+1) otherwise. 
        seg_label:
            Tensor containing segment labels to differentiate between cell and
            neighborhood gene tokens with shape (BATCH_SIZE, SEQ_LEN) if no
            <cls> token is included and (BATCH_SIZE, SEQ_LEN+1) otherwise.
        masks:
            List of N_MASKS tensors containing indices (within the sequence) of
            tokens to keep with shape (BATCH_SIZE, MASK_SIZE).

        Returns
        -----------
        emb_list:
            List containing the embeddings after each layer with shape (
            BATCH_SIZE * N_MASKS, MIN_MASK_SIZE, EMBED_DIM), where MIN_MASK_SIZE
            is minimum mask size in the batch. 
        """
        # Format masks
        if masks is not None:
            if not isinstance(masks, list):
                masks = [masks]

        # Get gene embeddings for sequence of gene tokens
        x = self.gene_embed(x)
        B, N, D = x.shape

        # Add positional and segment embeddings to gene embedding
        x = x + self.pos_embed + self.seg_embed(seg_label)

        # Mask token embeddings if masks are provided
        if masks is not None:
            x = apply_masks(x, masks)

        # Run forward prop and store embeddings after each block
        n_blocks = len(self.blocks)
        emb_list = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if (i == (n_blocks - 1)) and (self.norm is not None):
                x = self.norm(x)
            emb_list.append(x)

        return emb_list


class GeneTransformerPredictor(nn.Module):
    """
    GeneTransformerPredictor class to predict encoded targets from encoded
    contexts.
    
    Parameters
    -----------
    embed_dim:
        Dimension of the input embedding.
    seq_len:
        Length of the token sequences (w/o <cls> token).
    has_cls:
        If 'True', sequences include a <cls> token at the start.
    pos_learnable:
        If 'True', positional embeddings are learnable, otherwise use sin cos
        positional embeddings.
    predictor_embed_dim:
        Dimension of the embedding of the predictor.
    depth:
        Number of transformer blocks in the predictor.
    num_heads:
        Number of attention heads in the Attention module.
    mlp_ratio:
        Ratio to determine number of hidden dimensions in MLP module compared to
        input and output dimensions.
    qkv_bias:
        If 'True', include bias in query, key, and value layers of Attention
        module.
    qk_scale:
        Scaling factor for query and key vectors of Attention module.
    drop_rate:
        Dropout ratio in projection layer of Attention module and in layers of
        MLP module.
    attn_drop_rate:
        Dropout ratio in attention layer of Attention module.
    drop_path_rate:
        Probability for dropping paths in DropPath module.
    norm_layer:
        Normalization layer.
    init_std:
        Standard deviation for weight initialization.
    """
    def __init__(self,
                 embed_dim: int,
                 seq_len: int,
                 has_cls: bool=True,
                 pos_learnable: bool=False,
                 predictor_embed_dim: int=384,
                 depth: int=6,
                 num_heads: int=8,
                 mlp_ratio: float=4.0,
                 qkv_bias: bool=True,
                 qk_scale: Optional[float]=None,
                 drop_rate: float=0.0,
                 attn_drop_rate: float=0.0,
                 drop_path_rate: float=0.0,
                 norm_layer: torch.nn.modules.normalization=nn.LayerNorm,
                 init_std: float=0.02,
                 **kwargs
                 ):
        super().__init__()
        self.init_std = init_std

        # Initialize layer to project from encoder to predictor embed dim
        self.predictor_embed = nn.Linear(embed_dim,
                                         predictor_embed_dim,
                                         bias=True)

        # Initialize mask token embedding for prediction
        self.mask_token = nn.Parameter(torch.zeros(predictor_embed_dim))

        # Define decaying drop path rate (higher drop rate in deeper blocks)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Retrieve positional embedding
        if pos_learnable:
            self.predictor_pos_embed = nn.Parameter(
                torch.zeros(1,
                            seq_len + (1 if has_cls else 0),
                            predictor_embed_dim),
                requires_grad=True)
            trunc_normal_(self.predictor_pos_embed, std=self.init_std)
            if has_cls:
                self.predictor_pos_embed.data[0, 0, :] = 0
        else:
            self.predictor_pos_embed = nn.Parameter(
                torch.zeros(1, seq_len + (1 if has_cls else 0),
                predictor_embed_dim),
                requires_grad=False)
            predictor_pos_embed = get_1d_sincos_pos_embed(
                self.predictor_pos_embed.shape[-1],
                seq_len,
                cls_token=has_cls)
            self.predictor_pos_embed.data.copy_(
                torch.from_numpy(predictor_pos_embed).float().unsqueeze(0))
        
        # Initialize predictor blocks, norm layer, and predictor projection
        # layer to project back to encoder embedding size
        self.predictor_blocks = nn.ModuleList([
            Block(dim=predictor_embed_dim,
                  num_heads=num_heads,
                  mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias,
                  qk_scale=qk_scale,
                  drop=drop_rate,
                  attn_drop=attn_drop_rate,
                  drop_path=dpr[i],
                  norm_layer=norm_layer)
            for i in range(depth)])
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim,
                                        embed_dim,
                                        bias=True)

        # Initialize mask token weights
        trunc_normal_(self.mask_token, std=self.init_std)
        
        # Initialize layer weights
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        """
        Helper function to scale initialized layer weights.
        """
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        """
        Helper function to initialize layer weights.
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self,
                x: torch.Tensor, 
                masks_enc: Union[list, torch.Tensor],
                masks_pred: Union[list, torch.Tensor]
                ) -> torch.Tensor:
        """
        Run predictor forward pass for a batch of input tokens.

        Parameters
        -----------
        x:
            Embeddings from the encoder with shape (BATCH_SIZE*N_CONTEXT_MASKS,
            CONTEXT_MASK_SIZE, EMBED_DIM).
        masks_enc:
            List of N_CONTEXT_MASKS tensors containing indices (within the
            sequence) of tokens to keep with shape (BATCH_SIZE,
            CONTEXT_MASK_SIZE).
        masks_pred:
            List of N_TARGET_MASKS tensors containing indices (within the
            sequence) of tokens to keep with shape (BATCH_SIZE,
            TARGET_MASK_SIZE).

        Returns
        -----------
        x:
            Embeddings of tokens included in the target masks with shape (
            BATCH_SIZE * N_CONTEXT_MASKS * N_TARGET_MASKS, TARGET_MASK_SIZE,
            EMBED_DIM).   
        """
        assert (masks_enc is not None) and (masks_pred is not None), \
            'Cannot run predictor without index masks.'

        # Format masks
        if not isinstance(masks_enc, list):
            masks_enc = [masks_enc]
        if not isinstance(masks_pred, list):
            masks_pred = [masks_pred]

        # Retrieve batch size
        B = len(x) // len(masks_enc) # len(x) is BATCH_SIZE*N_CONTEXT_MASKS

        # Map from encoder dim to pedictor dim
        x = self.predictor_embed(x)

        # Add positional embeddings to tokens from context masks
        x_pos_embed = self.predictor_pos_embed.repeat(B, 1, 1)
        x += apply_masks(x_pos_embed, masks_enc) # only keep pos from encoder
                                                 # masks

        _, N_ctxt, D = x.shape # N_ctxt: CONTEXT_MASK_SIZE, D: PRED_EMBED_DIM

        # Create positional embeddings for tokens from target masks
        pos_embs = self.predictor_pos_embed.repeat(B, 1, 1)
        pos_embs = apply_masks(pos_embs, masks_pred) # only keep pos from
                                                     # predictor masks
        pos_embs = repeat_interleave_batch(
            pos_embs,
            B,
            repeat=len(masks_enc)) # repeat pos embeddings for all encoder masks
        
        # Repeat mask token for all batches, masks and positions from
        # predictor masks
        pred_tokens = self.mask_token.repeat(
            pos_embs.size(0), # BATCH_SIZE * N_CONTEXT_MASKS * N_TARGET_MASKS
            pos_embs.size(1), # TARGET_MASK_SIZE
            1)
                                             
        pred_tokens += pos_embs # add positional embeddings to mask tokens

        # Repeat context embeddings for all target masks
        x = x.repeat(len(masks_pred), 1, 1)

        # Concatenate context embeddings and mask tokens (both incl. pos
        # embedding)
        x = torch.cat([x, pred_tokens], dim=1)

        # Run forward prop
        for blk in self.predictor_blocks:
            x = blk(x)
        x = self.predictor_norm(x)

        # Return predictions for (target) mask tokens
        x = x[:, N_ctxt:]

        # Map back from predictor dim to encoder dim
        x = self.predictor_proj(x)

        return x


def gt_encoder(**kwargs):
    model = GeneTransformerEncoder(
        num_heads=8,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    return model


def gt_predictor(**kwargs):
    model = GeneTransformerPredictor(
        num_heads=8,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    return model
