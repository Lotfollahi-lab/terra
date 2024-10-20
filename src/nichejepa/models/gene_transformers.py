"""
Adapted from Assran, M. et al. Self-supervised learning from images with a
Joint-Embedding Predictive Architecture. Proc. IEEE Comput. Soc. Conf. Comput.
Vis. Pattern Recognit. 15619–15629 (2023);
https://github.com/facebookresearch/ijepa/blob/main/src/models/vision_transformer.py
(05.06.2024).
"""

import math
from abc import ABC, abstractmethod
from functools import partial
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from .modules import Attention, Block, CountProjection, DropPath, MLP
from .utils import (get_1d_sincos_pos_embed,
                    repeat_interleave_batch,
                    trunc_normal_)
from ..masks.utils import apply_masks


class GeneTransformerBaseEncoder(ABC, nn.Module):
    """
    GeneTransformerBaseEncoder class to encode contexts or targets.

    Parameters
    -----------
    vocab_size:
        Size of the token vocabulary. Includes <pad> token.
    seq_len:
        Length of the token sequences.
    n_special_tokens:
        Number of special tokens included in a token sequence.
    n_segments:
        Number of token segments within a token sequence.
    seg_learnable:
        If 'True', segment embeddings are learnable, otherwise fixed.
    embed_dim:
        Dimension of the encoder embedding.
    depth:
        Number of transformer blocks in the encoder.
    predictor_embed_dim:
        Dimension of the predictor embedding.
    predictor_depth:
        Number of transformer blocks in the predictor.
    num_heads:
        Number of attention heads in the Attention modules.
    mlp_ratio:
        Ratio to determine number of hidden dimensions in MLP modules compared
        to input and output dimensions.
    qkv_bias:
        If 'True', include bias in query, key, and value layers of Attention
        modules.
    qk_scale:
        Scaling factor for query and key vectors of Attention modules.
    drop_rate:
        Dropout ratio in projection layer of Attention modules and in layers of
        MLP modules.
    attn_drop_rate:
        Dropout ratio in attention layer of Attention modules.
    drop_path_rate:
        Probability for dropping paths in DropPath modules.
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
            
        # Initialize token embeddings
        self.token_embed = nn.Embedding(vocab_size, # already includes <pad>
                                        embed_dim,
                                        padding_idx=0)

        # Initialize segment embeddings
        self.seg_embed = nn.Embedding(
            n_segments + 2, # include <pad> and special tokens segment
            embed_dim,
            padding_idx=0)

        if not seg_learnable:
            # Prevent gradient updates and initialize with sincos embedding
            self.seg_embed.weight.requires_grad = False
            seg_embed = get_1d_sincos_pos_embed(
                embed_dim=embed_dim,
                n_zero_pos=0,
                n_sincos_pos=n_segments + 1)
            self.seg_embed.weight[1:].copy_(torch.from_numpy(seg_embed).float())

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

    @torch.no_grad()
    def return_token_emb(self,
                         tokens: torch.Tensor,
                         ) -> torch.Tensor:
        """
        Return the token embeddings for a batch of input tokens.

        Parameters
        -----------
        tokens:
            Tensor containing input tokens with shape (BATCH_SIZE, SEQ_LEN).

        Returns
        -----------
        token_embed:
            Tensor containing the token embeddings with shape (BATCH_SIZE,
            SEQ_LEN, EMBED_DIM).
        """
        # Retrieve token embeddings
        token_emb = self.token_embed(tokens)

        return token_emb

    @torch.no_grad()
    def return_seg_emb(self,
                       segments: torch.Tensor,
                       ) -> torch.Tensor:
        """
        Return the segment embeddings for a batch of input segments.

        Parameters
        -----------
        segments:
            Tensor containing input segments with shape (BATCH_SIZE, SEQ_LEN).

        Returns
        -----------
        seg_embed:
            Tensor containing the segment embeddings with shape (BATCH_SIZE,
            SEQ_LEN, EMBED_DIM).
        """
        # Retrieve segment embeddings
        seg_emb = self.seg_embed(x)

        return seg_emb

    @abstractmethod
    def forward(self) -> torch.Tensor:
        """
        Encoder-specific logic for forward pass.
        """
        pass

    @abstractmethod
    def return_multi_layer_emb() -> List[torch.Tensor]:
        """
        Encoder-specific logic for returning embeddings during inference.
        """
        pass


class GeneTransformerBasePredictor(ABC, nn.Module):
    """
    GeneTransformerBasePredictor class to predict encoded targets from encoded
    contexts.
    
    Parameters
    -----------
    embed_dim:
        Dimension of the predictor embedding.
    seq_len:
        Length of the token sequences.
    n_special_tokens:
        Number of special tokens included in a token sequence.
    n_segments:
        Number of token segments within a token sequence.
    seg_learnable:
        If 'True', segment embeddings are learnable, otherwise fixed.
    predictor_embed_dim:
        Dimension of the embedding of the predictor.
    depth:
        Number of transformer blocks in the predictor.
    num_heads:
        Number of attention heads in the Attention modules.
    mlp_ratio:
        Ratio to determine number of hidden dimensions in MLP modules compared
        to input and output dimensions.
    qkv_bias:
        If 'True', include bias in query, key, and value layers of Attention
        modules.
    qk_scale:
        Scaling factor for query and key vectors of Attention modules.
    drop_rate:
        Dropout ratio in projection layer of Attention module and in layers of
        MLP modules.
    attn_drop_rate:
        Dropout ratio in attention layer of Attention modules.
    drop_path_rate:
        Probability for dropping paths in DropPath modules.
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

    @abstractmethod
    def forward(self) -> torch.Tensor:
        """
        Predictor-specific logic for forward pass.
        """
        pass


class GeneTransformerRankEncoder(GeneTransformerBaseEncoder):
    """
    GeneTransformerRankEncoder class to encode contexts or targets using ranks
    based on gene expression counts.

    Parameters
    -----------
    pos_learnable:
        If 'True', positional embeddings are learnable, otherwise use sin cos
        positional embeddings.
    """
    def __init__(self,
                 pos_learnable: bool=False,
                 **base_encoder_kwargs,
                 ):
        super().__init__(**base_encoder_kwargs)

        # Initialize positional embeddings
        self.pos_embed = nn.Embedding(self.seq_len + 1, # include <pad>
                                      self.embed_dim,
                                      padding_idx=0)

        if not pos_learnable:
            # Prevent gradient updates and initialize with sincos embedding
            self.pos_embed.weight.requires_grad = False
            pos_embed = get_1d_sincos_pos_embed(
                embed_dim=self.embed_dim,
                n_zero_pos=0,
                n_sincos_pos=self.seq_len)
            self.pos_embed.weight[1:].copy_(torch.from_numpy(pos_embed).float())

    def forward(self,
                tokens: torch.Tensor,
                segments: torch.Tensor,
                positions: torch.Tensor,
                masks: Optional[Union[List[torch.Tensor], torch.Tensor]]=None,
                masks_attention: Optional[torch.Tensor]=None 
                ) -> torch.Tensor:
            """
            Run encoder forward pass on a batch of input token sequences. For
            each observation in the batch return only embeddings for tokens
            included in the masks.

            Parameters
            -----------
            tokens:
                Tensor containing input tokens with shape (BATCH_SIZE, SEQ_LEN).
            segments:
                Tensor containing segment labels with shape (BATCH_SIZE,
                SEQ_LEN).
            positions:
                Tensor containing position labels with shape (BATCH_SIZE,
                SEQ_LEN)
            masks:
                List of N_MASKS tensors containing indices (within the sequence)
                of tokens to keep with shape (BATCH_SIZE, MASK_SIZE).
            masks_attention:
                Tensor containing input for mask attention.
            
            Returns
            -----------
            x:
                Embeddings of input tokens included in the masks with shape (
                BATCH_SIZE * N_MASKS, MIN_MASK_SIZE, EMBED_DIM), where
                MIN_MASK_SIZE is minimum mask size in the batch.    
            """
            # Format masks
            if masks is not None:
                if not isinstance(masks, list):
                    masks = [masks]

            # Get token embeddings for sequence of tokens
            token_emb = self.token_embed(tokens)

            # Get segment embeddings
            seg_emb = self.seg_embed(segments)

            # Get positional embeddings
            pos_emb = self.pos_embed(positions)
            
            # Add positional and segment embeddings to token embeddings
            x = token_emb + seg_emb + pos_emb

            B, N, D = x.shape # B: BATCH_SIZE, N: SEQ_LEN, D: EMBED_DIM
                
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
    def return_multi_layer_emb(self,
                               tokens: torch.Tensor,
                               segments: torch.Tensor,
                               positions: torch.Tensor,
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
        tokens:
            Tensor containing input tokens with shape (BATCH_SIZE, SEQ_LEN). 
        segments:
            Tensor containing segment labels with shape (BATCH_SIZE, SEQ_LEN).
        positions:
            Tensor containing position labels with shape (BATCH_SIZE, SEQ_LEN).
        masks:
            List of N_MASKS tensors containing indices (within the sequence) of
            tokens to keep with shape (BATCH_SIZE, MASK_SIZE).
        masks_attention:
            Tensor containing input for mask attention.

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
        
        # Replace special tokens (except <cls> tokens) with pad tokens for
        # inference
        tokens[:, 2:self.n_special_tokens] = 0

        # Get token embeddings for sequence of tokens
        token_emb = self.token_embed(tokens)

        # Get segment embeddings
        seg_emb = self.seg_embed(segments)

        # Get positional embeddings
        pos_emb = self.pos_embed(positions)

        # Add positional and segment embeddings to token embeddings
        x = token_emb + seg_emb + pos_emb

        B, N, D = x.shape

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


class GeneTransformerCountEncoder(GeneTransformerBaseEncoder):
    """
    GeneTransformerCountEncoder class to encode contexts or targets using gene
    expression counts.
    """
    def __init__(self,
                 n_value_bins: int=100,
                 **base_encoder_kwargs
                 ):
        super().__init__(**base_encoder_kwargs)
        self.n_value_bins = n_value_bins

        # Initialize value embeddings
        self.value_embed = nn.Embedding(self.n_value_bins,
                                        self.embed_dim)
        self.special_value_embed = nn.Embedding(4, # temp
                                                self.embed_dim)

        self.count_projection = CountProjection(dim=self.n_value_bins)

    def forward(self,
                tokens: torch.Tensor,
                segments: torch.Tensor,
                counts: torch.Tensor,
                masks: Optional[Union[List[torch.Tensor], torch.Tensor]]=None,
                masks_attention: Optional[torch.Tensor]=None 
                ) -> torch.Tensor:
            """
            Run encoder forward pass on a batch of input token sequences. For
            each observation in the batch return only embeddings for tokens
            included in the masks.

            Parameters
            -----------
            tokens:
                Tensor containing input tokens with shape (BATCH_SIZE, SEQ_LEN).
            segments:
                Tensor containing segment labels with shape (BATCH_SIZE,
                SEQ_LEN).
            counts:
                Tensor containing the counts corresponding to gene tokens with
                shape (BATCH_SIZE, SEQ_LEN).
            masks:
                List of N_MASKS tensors containing indices (within the sequence)
                of tokens to keep with shape (BATCH_SIZE, MASK_SIZE).
            masks_attention:
                Tensor containing input for mask attention 
            Returns
            -----------
            x:
                Embeddings of input tokens included in the masks with shape (
                BATCH_SIZE * N_MASKS, MIN_MASK_SIZE, EMBED_DIM), where
                MIN_MASK_SIZE is minimum mask size in the batch.    
            """
            # Format masks
            if masks is not None:
                if not isinstance(masks, list):
                    masks = [masks]

            # Get token embeddings for sequence of tokens
            token_emb = self.token_embed(tokens)

            #value_embed = self.value_embed(
            #    torch.tensor(range(self.n_value_bins)).to(tokens.device))

            #special_value_embed = self.special_value_embed(
            #    torch.tensor(range(4)).to(tokens.device))

            a = self.count_projection(counts.unsqueeze(dim=-1))
            value_emb = torch.matmul(a, self.value_embed.weight)

            zero_counts_mask = counts == 0.0 # 0 counts have a separate
                                             # embedding
            value_emb[zero_counts_mask] = self.special_value_embed.weight[0, :].to(value_emb.dtype) # special_value_embed[0, :] # self.special_value_embed.weight[0, :]
            value_emb[:, 0:3, :] = self.special_value_embed.weight[1:4, :].to(value_emb.dtype) # special_value_embed[1:4, :] # self.special_value_embed.weight[1:4, :]

            # Get embeddings for segments
            seg_emb = self.seg_embed(segments)
            
            # Add token and segment embeddings to value embeddings
            x = token_emb + value_emb + seg_emb

            B, N, D = x.shape # B: BATCH_SIZE, N: SEQ_LEN, D: EMBED_DIM
                
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
    def return_multi_layer_emb(self,
                               tokens: torch.Tensor,
                               segments: torch.Tensor,
                               counts: torch.Tensor,
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
        tokens:
            Tensor containing input tokens with shape (BATCH_SIZE, SEQ_LEN). 
        segments:
            Tensor containing segment labels with shape (BATCH_SIZE, SEQ_LEN).
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

        # Get token embeddings for sequence of tokens
        token_emb = self.token_embed(tokens)

        a = self.count_projection(counts.unsqueeze(dim=-1))
        value_emb = torch.matmul(a, self.value_embed.weight)

        print(value_emb.shape)

        zero_counts_mask = counts == 0.0 # 0 counts have a separate
                                            # embedding
        value_emb[zero_counts_mask] = self.special_value_embed.weight[0, :].to(value_emb.dtype) # special_value_embed[0, :] # self.special_value_embed.weight[0, :]
        value_emb[:, 0:2, :] = self.special_value_embed.weight[1:3, :].to(value_emb.dtype) # special_value_embed[1:4, :] # self.special_value_embed.weight[1:4, :]
        value_emb[:, 2, :] = self.special_value_embed.weight[0, :].to(value_emb.dtype)

        # Get embeddings for segments
        seg_emb = self.seg_embed(segments)
        
        # Add token and segment embeddings to value embeddings
        x = token_emb + value_emb + seg_emb

        B, N, D = x.shape # B: BATCH_SIZE, N: SEQ_LEN, D: EMBED_DIM

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


class GeneTransformerRankPredictor(GeneTransformerBasePredictor):
    """
    GeneTransformerRankPredictor class.
    """
    def __init__(self,
                 **base_encoder_kwargs
                 ):
        super().__init__(**base_encoder_kwargs)

    def forward(self,
                z: torch.Tensor,
                segments: torch.Tensor,
                positions: torch.Tensor,
                enc_pos_embed: nn.Embedding,
                enc_seg_embed: nn.Embedding,
                masks_enc: Union[List[torch.Tensor], torch.Tensor],
                masks_pred: Union[List[torch.Tensor], torch.Tensor]
                ) -> torch.Tensor:
            """
            Run predictor forward pass for a batch of input tokens.

            Parameters
            -----------
            z:
                Embeddings from the encoder with shape (
                BATCH_SIZE*N_CONTEXT_MASKS, CONTEXT_MASK_SIZE, EMBED_DIM).
            segments:
                Tensor containing segment labels with shape (BATCH_SIZE,
                SEQ_LEN).
            positions:
                Tensor containing position labels with shape (BATCH_SIZE,
                SEQ_LEN).
            enc_pos_embed:
                Positional embeddings from the encoder.
            enc_seg_embed:
                Segment embeddings from the encoder.
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
            z:
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
            B = len(z) // len(masks_enc) # len(x) is BATCH_SIZE*N_CONTEXT_MASKS

            # Map from encoder dim to pedictor dim
            z = self.predictor_embed(z)

            # Add positional embeddings to tokens from context masks
            z_pos_embed = enc_pos_embed(positions)
            z_seg_embed = enc_seg_embed(segments)
            z += apply_masks(z_pos_embed, masks_enc) # only keep pos from encoder
                                                    # masks

            z += apply_masks(z_seg_embed, masks_enc) # only keep seg from encoder
                                                    # masks

            _, N_ctxt, D = z.shape # N_ctxt: CONTEXT_MASK_SIZE, D: PRED_EMBED_DIM

            # Create positional embeddings for tokens from target masks
            pos_embs = enc_pos_embed(positions)
            seg_embs = enc_seg_embed(segments)

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
            pred_tokens += seg_embs + pos_embs

            # Repeat context embeddings for all target masks
            z = z.repeat(len(masks_pred), 1, 1)

            # Concatenate context embeddings and mask tokens (both incl. pos
            # embedding)
            z = torch.cat([z, pred_tokens], dim=1)

            # Run forward prop
            for blk in self.predictor_blocks:
                z = blk(z)
            z = self.predictor_norm(z)

            # Return predictions for (target) mask tokens
            z = z[:, N_ctxt:]

            # Map back from predictor dim to encoder dim
            z = self.predictor_proj(z)

            return z


class GeneTransformerCountPredictor(GeneTransformerBasePredictor):
    """
    GeneTransformerCountPredictor class.
    """
    def __init__(self,
                 pos_learnable: bool=False,
                 **base_predictor_kwargs
                 ):
        super().__init__(**base_predictor_kwargs)

    def forward(self,
                z: torch.Tensor,
                segments: torch.Tensor,
                tokens: torch.Tensor,
                enc_token_embed: nn.Embedding,
                enc_seg_embed: nn.Embedding,
                masks_enc: Union[List[torch.Tensor], torch.Tensor],
                masks_pred: Union[List[torch.Tensor], torch.Tensor],
                ) -> torch.Tensor:
            """
            Run predictor forward pass for a batch of input tokens.

            Parameters
            -----------
            z:
                Embeddings from the encoder with shape (
                BATCH_SIZE*N_CONTEXT_MASKS, CONTEXT_MASK_SIZE, EMBED_DIM).
            segments:
                Tensor containing segment labels with shape (BATCH_SIZE,
                SEQ_LEN).
            tokens:
                Tensor containing tokens with shape (BATCH_SIZE, SEQ_LEN).
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
            z:
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
            B = len(z) // len(masks_enc) # len(x) is BATCH_SIZE*N_CONTEXT_MASKS

            # Map from encoder dim to pedictor dim
            z = self.predictor_embed(z)

            token_embs = enc_token_embed(tokens)
            seg_embs = enc_seg_embed(segments)

            # Add positional embeddings to tokens from context masks
            z_gene_embed = enc_token_embed(tokens)
            z_seg_embed = enc_seg_embed(segments)
            z += apply_masks(z_gene_embed, masks_enc) # only keep pos from encoder
                                                    # masks

            z += apply_masks(z_seg_embed, masks_enc) # only keep seg from encoder
                                                    # masks

            _, N_ctxt, D = z.shape # N_ctxt: CONTEXT_MASK_SIZE, D: PRED_EMBED_DIM

            # Create positional embeddings for tokens from target masks
            token_embs = apply_masks(token_embs, masks_pred) # only keep pos from
                                                        # predictor masks
            seg_embs = apply_masks(seg_embs, masks_pred) # only keep seg from
                                                        # predictor masks
            
            token_embs = repeat_interleave_batch(
                token_embs,
                B,
                repeat=len(masks_enc)) # repeat pos embeddings for all encoder masks
            seg_embs = repeat_interleave_batch(
                seg_embs,
                B,
                repeat=len(masks_enc)) # repeat seg embeddings for all encoder masks

            # Repeat mask token for all batches, masks and positions from
            # predictor masks
            pred_tokens = self.mask_token.repeat(
                token_embs.size(0), # BATCH_SIZE * N_CONTEXT_MASKS * N_TARGET_MASKS
                token_embs.size(1), # TARGET_MASK_SIZE
                1)

            # Add gene and segment embeddings to mask tokens                  
            pred_tokens += token_embs + seg_embs

            # Repeat context embeddings for all target masks
            z = z.repeat(len(masks_pred), 1, 1)

            # Concatenate context embeddings and mask tokens (both incl. pos
            # embedding)
            z = torch.cat([z, pred_tokens], dim=1)

            # Run forward prop
            for blk in self.predictor_blocks:
                z = blk(z)
            z = self.predictor_norm(z)

            # Return predictions for (target) mask tokens
            z = z[:, N_ctxt:]

            # Map back from predictor dim to encoder dim
            z = self.predictor_proj(z)

            return z


def init_gt_encoder(encoder_type: Literal['rank', 'counts'],
               **encoder_kwargs
               ) -> Union[GeneTransformerRankEncoder,
                          GeneTransformerCountEncoder]:
    if encoder_type == 'rank':
        model = GeneTransformerRankEncoder(
            num_heads=8,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **encoder_kwargs)
    elif encoder_type == 'counts':
        model = GeneTransformerCountEncoder(
            num_heads=8,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **encoder_kwargs)

    return model


def init_gt_predictor(predictor_type: Literal['rank', 'counts'],
                 **predictor_kwargs
                 ) -> Union[GeneTransformerRankPredictor,
                            GeneTransformerCountPredictor]:
    if predictor_type == 'rank':
        model = GeneTransformerRankPredictor(
            num_heads=8,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **predictor_kwargs)
    elif predictor_type == 'counts':
        model = GeneTransformerCountPredictor(
            num_heads=8,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **predictor_kwargs)        

    return model
