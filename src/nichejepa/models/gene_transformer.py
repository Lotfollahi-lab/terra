"""
Adapted from Assran, M. et al. Self-supervised learning from images with a
Joint-Embedding Predictive Architecture. Proc. IEEE Comput. Soc. Conf. Comput.
Vis. Pattern Recognit. 15619–15629 (2023);
https://github.com/facebookresearch/ijepa/blob/main/src/models/vision_transformer.py
(05.06.2024).
"""

import math
from functools import partial
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from .modules import Attention, Block, DropPath, MLP
from .utils import (get_1d_sincos_pos_embed,
                    repeat_interleave_batch,
                    trunc_normal_)
from ..masks.utils import apply_masks


class GeneTransformerEncoder(nn.Module):
    """
    GeneTransformerEncoder class to encode contexts or targets.
    
    Parameters
    -----------
    vocab_size:
        Size of the token vocabulary. Includes <pad> token.
    seq_len:
        Length of the token sequences.
    n_special_tokens:
    n_segments:
    pos_learnable:
        If 'True', positional embeddings are learnable, otherwise use sin cos
        positional embeddings.
    seg_learnable:
        If 'True', segment embeddings are learnable, otherwise use fixed
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
                 n_special_tokens: int,
                 n_segments: int,
                 pos_learnable: bool=False,
                 seg_learnable: bool=False,
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
        self.n_special_tokens = n_special_tokens
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.init_std = init_std
        
        # Initialize gene embeddings
        self.gene_embed = nn.Embedding(vocab_size, # already includes <pad>
                                       embed_dim,
                                       padding_idx=0)
                                          
        # Initialize segment embeddings
        self.seg_embed = nn.Embedding(n_segments + 1, # add <pad>
                                      embed_dim,
                                      padding_idx=0)
        if not seg_learnable:
            # Prevent gradient updates and manually set embedding weights
            self.seg_embed.weight.requires_grad = False
            self.seg_embed.weight.copy_(
                torch.tensor(
                    [[0] * embed_dim,
                     [1] * embed_dim,
                     [2] * embed_dim],
                    dtype=torch.float32))

        # Initialize positional embeddings
        self.pos_embed = nn.Parameter(
            torch.zeros(1,
                        seq_len,
                        embed_dim))
        if pos_learnable:
            trunc_normal_(self.pos_embed, std=self.init_std)
            if n_special_tokens > 0:
                self.pos_embed.data[0, 0:n_special_tokens, :] = 0
        else:
            self.pos_embed.requires_grad = False
            pos_embed = get_1d_sincos_pos_embed(
                embed_dim=self.pos_embed.shape[-1],
                n_zero_pos=n_special_tokens,
                n_sincos_pos=seq_len-n_special_tokens)
            self.pos_embed.data.copy_(
                torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Define decaying drop path rate (higher drop rate in deeper blocks)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Initialize encoder blocks and norm layer
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim,
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
                masks: Optional[Union[List[torch.Tensor], torch.Tensor]]=None,
                masks_attention: Optional[torch.Tensor]=None 
                ) -> torch.Tensor:
        """
        Run encoder forward pass on a batch of input token sequences. For each 
        observation in the batch return only embeddings for tokens included in
        the masks.

        Parameters
        -----------
        x:
            Tensor containing input tokens with shape (BATCH_SIZE, SEQ_LEN).
        seg_label:
            Tensor containing segment labels with shape (BATCH_SIZE, SEQ_LEN).
        masks:
            List of N_MASKS tensors containing indices (within the sequence) of
            tokens to keep with shape (BATCH_SIZE, MASK_SIZE).
        masks_attention:
            Tensor containing input for mask attention 
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
        B, N, D = x.shape # B: BATCH_SIZE, N: SEQ_LEN, D: EMBED_DIM
        
        # Add positional and segment embeddings to gene embedding
        x = x + self.pos_embed + self.seg_embed(seg_label)
              
        # Mask token embeddings if masks are provided
        if masks is not None:
            x = apply_masks(x, masks)
        
        # Run forward prop
        for i, blk in enumerate(self.blocks):
            if masks_attention is not None:
               x = blk(x, masks=masks_attention)
            else:
               x = blk(x)
        if self.norm is not None:
            x = self.norm(x)

        return x

    @torch.no_grad()
    def return_pos_emb(self,
                       x: torch.Tensor,
                       ) -> torch.Tensor:
        """
        Return the positional embeddings for a batch of input token sequences.

        Parameters
        -----------
        x:
            Tensor containing input tokens with shape (BATCH_SIZE, SEQ_LEN).

        Returns
        -----------
        pos_embed:
            Tensor containing the positional embeddings repeated across the
            batch dimension with shape (BATCH_SIZE, SEQ_LEN, EMBED_DIM).
        """
        # Repeat the positional embeddings across the batch dimension to match
        # the input shape
        pos_embed = self.pos_embed.repeat(x.shape[0], 1, 1)

        return pos_embed

    @torch.no_grad()
    def return_gene_emb(self,
                        x: torch.Tensor,
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
                               masks: Optional[Union[
                                   List[torch.Tensor], torch.Tensor]]=None,
                               masks_attention: Optional[torch.Tensor]=None, 
                               ) -> List[torch.Tensor]:
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
        masks_attention:
            Tensor containing input for mask attention 

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
            if masks_attention is not None:
               x = blk(x, masks=masks_attention)
            else:
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
        Length of the token sequences.
    n_special_tokens:
    n_segments:
    pos_learnable:
        If 'True', positional embeddings are learnable, otherwise use sin cos
        positional embeddings.
    seg_learnable:
        If 'True', segment embeddings are learnable, otherwise use fixed
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
                 n_special_tokens: int,
                 n_segments: int,
                 pos_learnable: bool=False,
                 seg_learnable: bool=False,
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
        self.seq_len = seq_len
        self.n_special_tokens = n_special_tokens
        self.predictor_embed_dim = predictor_embed_dim
        self.num_heads = num_heads
        self.init_std = init_std

        # Initialize layer to project from encoder to predictor embed dim
        self.predictor_embed = nn.Linear(embed_dim,
                                         predictor_embed_dim,
                                         bias=True)

        # Initialize mask token embedding for prediction
        self.mask_token = nn.Parameter(torch.zeros(predictor_embed_dim))

        # Define decaying drop path rate (higher drop rate in deeper blocks)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Initialize segment embeddings
        self.seg_embed = nn.Embedding(n_segments + 1, # add <pad>
                                      embed_dim,
                                      padding_idx=0)
        if not seg_learnable:
            # Prevent gradient updates and manually set embedding weights
            self.seg_embed.weight.requires_grad = False
            self.seg_embed.weight.copy_(
                torch.tensor(
                    [[0] * embed_dim,
                     [1] * embed_dim,
                     [2] * embed_dim],
                    dtype=torch.float32))

        # Initialize positional embeddings
        self.predictor_pos_embed = nn.Parameter(
            torch.zeros(1,
                        seq_len,
                        predictor_embed_dim))
        if pos_learnable:
            trunc_normal_(self.predictor_pos_embed, std=self.init_std)
            if n_special_tokens > 0:
                self.predictor_pos_embed.data[0, 0:n_special_tokens, :] = 0
        else:
            self.predictor_pos_embed.requires_grad = False
            predictor_pos_embed = get_1d_sincos_pos_embed(
                embed_dim=self.predictor_pos_embed.shape[-1],
                n_zero_pos=n_special_tokens,
                n_sincos_pos=seq_len-n_special_tokens)
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
                seg_label: torch.Tensor,
                masks_enc: Union[List[torch.Tensor], torch.Tensor],
                masks_pred: Union[List[torch.Tensor], torch.Tensor]
                ) -> torch.Tensor:
        """
        Run predictor forward pass for a batch of input tokens.

        Parameters
        -----------
        x:
            Embeddings from the encoder with shape (BATCH_SIZE*N_CONTEXT_MASKS,
            CONTEXT_MASK_SIZE, EMBED_DIM).
        seg_label:
            Tensor containing segment labels with shape (BATCH_SIZE, SEQ_LEN).
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
        x_seg_embed = self.seg_embed(seg_label)
        x += apply_masks(x_pos_embed, masks_enc) # only keep pos from encoder
                                                 # masks

        x += apply_masks(x_seg_embed, masks_enc) # only keep seg from encoder
                                                 # masks

        _, N_ctxt, D = x.shape # N_ctxt: CONTEXT_MASK_SIZE, D: PRED_EMBED_DIM

        # Create positional embeddings for tokens from target masks
        pos_embs = self.predictor_pos_embed.repeat(B, 1, 1)
        seg_embs = self.seg_embed(seg_label)

        pos_embs = apply_masks(pos_embs, masks_pred) # only keep pos from
                                                     # predictor masks
        seg_embs = apply_masks(seg_embs, masks_pred) # only keep seg from
                                                     # predictor masks
          
        pos_embs = repeat_interleave_batch(
            pos_embs,
            B,
            repeat=len(masks_enc)) # repeat pos embeddings for all encoder masks
        seg_embs = repeat_interleave_batch(
            seg_embs,
            B,
            repeat=len(masks_enc)) # repeat seg embeddings for all encoder masks

        # Repeat mask token for all batches, masks and positions from
        # predictor masks
        pred_tokens = self.mask_token.repeat(
            pos_embs.size(0), # BATCH_SIZE * N_CONTEXT_MASKS * N_TARGET_MASKS
            pos_embs.size(1), # TARGET_MASK_SIZE
            1)

        # Add positional and segment embeddings to mask tokens                  
        pred_tokens += pos_embs + seg_embs

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
