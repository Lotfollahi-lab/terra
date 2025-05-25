"""
Adapted from Assran, M. et al. Self-supervised learning from images with
a Joint-Embedding Predictive Architecture. Proc. IEEE Comput. Soc. Conf.
Comput. Vis. Pattern Recognit. 15619–15629 (2023);
https://github.com/facebookresearch/ijepa/blob/main/src/models/vision_transformer.py
(05.06.2024).
"""

import math
from abc import ABC, abstractmethod
from functools import partial
from typing import Literal

import numpy as np
import torch
import torch.nn as nn

from .modules import Attention, Block, ValueEmbWeightsProjection, MLP
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
        Ratio to determine number of hidden dimensions in MLP modules
        compared to input and output dimensions.
    qkv_bias:
        If `True`, include bias in query, key, and value layers of
        Attention modules.
    qk_scale:
        Scaling factor for query and key vectors of Attention modules.
    drop_rate:
        Dropout ratio in projection layer of Attention modules and in
        layers of MLP modules.
    attn_drop_rate:
        Dropout ratio in attention layer of Attention modules.
    norm_layer:
        Normalization layer.
    init_std:
        Standard deviation for weight initialization.
    use_flash_attention:
        If `True`, use flash_attention.
    use_layer_norm:
        If `True`, use layer normalization, else use dynamic tanh
        normalization.
    api_version:
        Version of the API to use.
    sep_gene_tokens_neb:
        If `True`, use separate gene tokens for neighborhood.
    """
    def __init__(self,
                 vocab_size: int,
                 seq_len: int,
                 n_special_tokens: int,
                 n_segments: int,
                 embed_dim: int = 768,
                 depth: int = 12,
                 predictor_embed_dim: int = 384,
                 predictor_depth: int = 12,
                 num_heads: int = 12,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 qk_scale: float | None = None,
                 drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0,
                 norm_layer: nn.modules.normalization=nn.LayerNorm,
                 init_std: float = 0.02,
                 use_flash_attention: bool = True,
                 use_layer_norm: bool = True,
                 api_version: Literal['v1', 'v2', 'v3'] = 'v3',
                 sep_gene_tokens_neb: bool = False,
                 **kwargs
                 ):
        super().__init__()
        self.seq_len = seq_len
        self.n_segments = n_segments
        self.n_special_tokens = n_special_tokens
        self.seq_len_cell = (seq_len - n_special_tokens)//n_segments
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.init_std = init_std
        self.api_version = api_version
        self.sep_gene_tokens_neb = sep_gene_tokens_neb
            
        # Initialize token embeddings
        self.token_embed = nn.Embedding(
            vocab_size + (vocab_size if sep_gene_tokens_neb else 0), # already includes <pad>
            embed_dim,
            padding_idx=0)

        # Initialize segment embeddings
        self.seg_embed = nn.Embedding(
            1 + n_segments + (105 if api_version == 'v1' else 0), # include <pad>
            embed_dim,
            padding_idx=0)
        
        # Prevent gradient updates and initialize with sincos embedding,
        # including special segments
        self.seg_embed.weight.requires_grad = False
        seg_embed = get_1d_sincos_pos_embed(
            embed_dim=embed_dim,
            n_zero_pos=0,
            n_sincos_pos=n_segments + (105 if api_version == 'v1' else 0))
        self.seg_embed.weight[1:].copy_(torch.from_numpy(seg_embed).float())

        # Initialize encoder blocks and norm layer
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim,
                  num_heads=num_heads,
                  mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias,
                  qk_scale=qk_scale,
                  drop=drop_rate,
                  act_layer=nn.GELU,
                  attn_drop=attn_drop_rate,
                  norm_layer=norm_layer,
                  use_flash_attention=use_flash_attention,
                  use_layer_norm=use_layer_norm)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Initialize weights of layers
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _rescale_blocks(self):
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

    @torch.no_grad()
    def return_token_emb(self,
                         tokens: torch.Tensor,
                         ) -> torch.Tensor:
        """
        Return the token embeddings for a batch of input tokens.

        Parameters
        -----------
        tokens:
            Tensor containing input tokens with shape (BATCH_SIZE,
            SEQ_LEN).

        Returns
        -----------
        token_embed:
            Tensor containing the token embeddings with shape
            (BATCH_SIZE, SEQ_LEN, EMBED_DIM).
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
            Tensor containing input segments with shape (BATCH_SIZE,
            SEQ_LEN).

        Returns
        -----------
        seg_embed:
            Tensor containing the segment embeddings with shape
            (BATCH_SIZE, SEQ_LEN, EMBED_DIM).
        """
        # Retrieve segment embeddings
        seg_emb = self.seg_embed(segments)

        return seg_emb

    @abstractmethod
    def forward(self) -> torch.Tensor:
        """
        Encoder-specific logic for forward pass.
        """
        pass

    @abstractmethod
    def return_layer_emb() -> list[torch.Tensor]:
        """
        Encoder-specific logic for returning layer-specific embeddings
        during inference.
        """
        pass


class GeneTransformerBasePredictor(ABC, nn.Module):
    """
    GeneTransformerBasePredictor class to predict encoded targets from
    encoded contexts.
    
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
    predictor_embed_dim:
        Dimension of the embedding of the predictor.
    depth:
        Number of transformer blocks in the predictor.
    num_heads:
        Number of attention heads in the Attention modules.
    mlp_ratio:
        Ratio to determine number of hidden dimensions in MLP modules
        compared to input and output dimensions.
    qkv_bias:
        If `True`, include bias in query, key, and value layers of
        Attention modules.
    qk_scale:
        Scaling factor for query and key vectors of Attention modules.
    drop_rate:
        Dropout ratio in projection layer of Attention module and in
        layers of MLP modules.
    attn_drop_rate:
        Dropout ratio in attention layer of Attention modules.
    norm_layer:
        Normalization layer.
    init_std:
        Standard deviation for weight initialization.
    use_flash_attention:
        If `True`, use flash_attention.
    use_layer_norm:
        If `True`, use layer normalization, else use dynamic tanh
        normalization.
    api_version:
        Version of the API to use.
    """
    def __init__(self,
                 embed_dim: int,
                 seq_len: int,
                 n_special_tokens: int,
                 n_segments: int,
                 predictor_embed_dim: int = 768,
                 depth: int = 6,
                 num_heads: int = 12,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 qk_scale: float | None = None,
                 drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0,
                 norm_layer: torch.nn.modules.normalization=nn.LayerNorm,
                 init_std: float = 0.02,
                 use_flash_attention: bool = True,
                 use_layer_norm: bool = True,
                 api_version: Literal['v1', 'v2', 'v3'] = 'v3',
                 **kwargs
                 ):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.n_special_tokens = n_special_tokens
        self.predictor_embed_dim = predictor_embed_dim
        self.num_heads = num_heads
        self.init_std = init_std
        self.api_version = api_version

        # Initialize positional embeddings
        self.pos_embed = nn.Embedding(self.seq_len + 1, # include <pad>
                                      predictor_embed_dim,
                                      padding_idx=0)

        # Prevent gradient updates and initialize with sincos
        # embedding
        self.pos_embed.weight.requires_grad = False
        pos_embed = get_1d_sincos_pos_embed(
            embed_dim=predictor_embed_dim,
            n_zero_pos=0,
            n_sincos_pos=self.seq_len)
        self.pos_embed.weight[1:].copy_(
            torch.from_numpy(pos_embed).float())

        # Initialize segment embeddings
        self.seg_embed = nn.Embedding(
            1 + n_segments + (105 if api_version == 'v1' else 0), # include <pad>
            predictor_embed_dim,
            padding_idx=0)
        
        # Prevent gradient updates and initialize with sincos embedding,
        # including special segments
        self.seg_embed.weight.requires_grad = False
        seg_embed = get_1d_sincos_pos_embed(
            embed_dim=predictor_embed_dim,
            n_zero_pos=0,
            n_sincos_pos=n_segments + (105 if api_version == 'v1' else 0))
        self.seg_embed.weight[1:].copy_(torch.from_numpy(seg_embed).float())

        # Initialize layer to project from enc to pred embed dim
        self.predictor_embed = nn.Linear(embed_dim,
                                         predictor_embed_dim,
                                         bias=True)
        self.token_embed_projection = nn.Linear(embed_dim,
                                                predictor_embed_dim,
                                                bias=True)

        # Initialize mask token embedding for prediction
        self.mask_token = nn.Parameter(torch.zeros(predictor_embed_dim))

        # Initialize predictor blocks, norm layer, and predictor
        # projection layer to project back to encoder embedding size
        self.predictor_blocks = nn.ModuleList([
            Block(dim=predictor_embed_dim,
                  num_heads=num_heads,
                  mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias,
                  qk_scale=qk_scale,
                  drop=drop_rate,
                  attn_drop=attn_drop_rate,
                  norm_layer=norm_layer,
                  use_flash_attention=use_flash_attention,
                  use_layer_norm=use_layer_norm)
            for i in range(depth)])
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim,
                                        embed_dim,
                                        bias=True)

        # Initialize mask token weights
        # trunc_normal_(self.mask_token, std=self.init_std)
        
        # Initialize layer weights
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _rescale_blocks(self):
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

    @abstractmethod
    def forward(self) -> torch.Tensor:
        """
        Predictor-specific logic for forward pass.
        """
        pass


class GeneTransformerRankEncoder(GeneTransformerBaseEncoder):
    """
    GeneTransformerRankEncoder class to encode contexts or targets using
    ranks based on gene expression counts.

    Parameters
    -----------
    pos_learnable:
        If `True`, positional embeddings are learnable, otherwise use
        sin cos positional embeddings.
    """
    def __init__(self,
                 pos_learnable: bool = False,
                 **base_encoder_kwargs,
                 ):
        super().__init__(**base_encoder_kwargs)

        # Initialize positional embeddings
        self.pos_embed = nn.Embedding(self.seq_len + 1, # include <pad>
                                      self.embed_dim,
                                      padding_idx=0)

        if not pos_learnable:
            # Prevent gradient updates and initialize with sincos
            # embedding
            self.pos_embed.weight.requires_grad = False
            pos_embed = get_1d_sincos_pos_embed(
                embed_dim=self.embed_dim,
                n_zero_pos=0,
                n_sincos_pos=self.seq_len)
            self.pos_embed.weight[1:].copy_(
                torch.from_numpy(pos_embed).float())

    def forward(self,
                positions: torch.Tensor,
                segments: torch.Tensor,
                tokens: torch.Tensor,
                masks: list[torch.Tensor] | torch.Tensor | None = None,
                masks_attention: torch.Tensor | None = None 
                ) -> torch.Tensor:
            """
            Run encoder forward pass on a batch of input token
            sequences. For each observation in the batch return only
            embeddings for tokens included in the masks.

            Parameters
            -----------
            positions:
                Tensor containing position labels with shape
                (BATCH_SIZE, SEQ_LEN).
            segments:
                Tensor containing segment labels with shape (BATCH_SIZE,
                SEQ_LEN).
            tokens:
                Tensor containing input tokens with shape (BATCH_SIZE,
                SEQ_LEN).
            masks:
                List of N_MASKS tensors containing indices (within the
                sequence) of tokens to keep with shape (BATCH_SIZE,
                MASK_SIZE).
            masks_attention:
                An attention tensor that controls how different tokens
                attend to each other within a sequence.
            
            Returns
            -----------
            x:
                Embeddings of input tokens included in the masks with
                shape (BATCH_SIZE * N_MASKS, MIN_MASK_SIZE, EMBED_DIM),
                where MIN_MASK_SIZE is minimum mask size in the batch.    
            """
            # Format masks
            if masks is not None:
                if not isinstance(masks, list):
                    masks = [masks]

            # Get positional, segment and token embeddings (excl.
            # special tokens)
            pos_emb = self.pos_embed(positions)
            seg_emb = self.seg_embed(segments)
            token_emb = self.token_embed(tokens)
            
            # Add positional and segment embeddings to token embeddings
            x = pos_emb + seg_emb + token_emb
            # B, N, D = x.shape # B: BATCH_SIZE, N: SEQ_LEN,
            # D: EMBED_DIM

            # Mask token embeddings if masks are provided
            if masks is not None:
                x = apply_masks(x, masks)
            
            # Run forward prop
            for i, blk in enumerate(self.blocks):
                x = blk(x, masks=masks_attention)
            if self.norm is not None:
                x = self.norm(x)

            return x, pos_emb, token_emb

    @torch.no_grad()
    def return_layer_emb(
            self,
            layer: int,
            positions: torch.Tensor,
            segments: torch.Tensor,
            tokens: torch.Tensor,
            masks: list[torch.Tensor] | torch.Tensor | None = None,
            masks_attention: torch.Tensor | None = None,
            pad_neighborhood: bool = False,
            ) -> list[torch.Tensor]:
        """
        Run encoder forward pass on a batch of input token sequences,
        applying masks if provided, and return the embeddings of a
        specific layer.

        Parameters
        -----------
        layer:
            Index of the specific layer to be returned
        positions:
            Tensor containing position labels with shape (BATCH_SIZE,
            SEQ_LEN).
        segments:
            Tensor containing segment labels with shape (BATCH_SIZE,
            SEQ_LEN).
        tokens:
            Tensor containing input tokens with shape (BATCH_SIZE,
            SEQ_LEN).
        masks:
            List of N_MASKS tensors containing indices (within the
            sequence) of tokens to keep with shape (BATCH_SIZE,
            MASK_SIZE).
        masks_attention:
            An attention tensor that controls how different tokens
            attend to each other within a sequence.

        Returns
        -----------
        x:
            Embeddings of a specific layer with shape (BATCH_SIZE *
            N_MASKS, MIN_MASK_SIZE, EMBED_DIM), where MIN_MASK_SIZE is
            minimum mask size in the batch. 
        """
        # Format masks
        if masks is not None:
                if not isinstance(masks, list):
                    masks = [masks]

        # Get positional, segment and token embeddings
        pos_emb = self.pos_embed(positions)
        seg_emb = self.seg_embed(segments)
        token_emb = self.token_embed(tokens)

        # Add positional and segment embeddings to token embeddings
        x = pos_emb + seg_emb + token_emb
        #B, N, D = x.shape

        # Pad special tokens
        x[:, :self.n_special_tokens, :] = 0 
        masks_attention[:,
                        :,
                        :
                        :self.n_special_tokens] = 0

        if pad_neighborhood:
            x[:, (self.n_special_tokens+self.seq_len_cell):, :] = 0

            masks_attention = masks_attention.expand(
                masks_attention.shape[0],
                1,
                masks_attention.shape[-1],
                masks_attention.shape[-1]).clone()

            # Mask neighborhood gene tokens for index cell gene tokens
            masks_attention[
                :,
                :,
                self.n_special_tokens:(self.n_special_tokens+self.seq_len_cell),
                (self.n_special_tokens+self.seq_len_cell):] = 0

        # Mask token embeddings if masks are provided
        if masks is not None:
            x = apply_masks(x, masks)

        # Run forward prop and store embeddings after each block
        n_blocks = len(self.blocks)
        for i, blk in enumerate(self.blocks):
            x = blk(x, masks=masks_attention)
            if (i == (n_blocks - 1)) and (self.norm is not None):
                x = self.norm(x)
            if i == (layer-1):
                break

        # Remove special tokens
        x = x[:, self.n_special_tokens:, :]

        return x


class GeneTransformerCountEncoder(GeneTransformerBaseEncoder):
    """
    GeneTransformerCountEncoder class to encode contexts or targets
    using gene expression counts.

    Parameters
    -----------
    count_encoding:
        Encoding module for counts. Can be either `value_bins`
        (scFoundation count encoding) or `mlp` (2 layer MLP).
    n_value_bins:
        Number of value bins if `value_bins` count encoding is used.    
    """
    def __init__(self,
                 n_special_values: int,
                 count_encoding: Literal['value_bins', 'mlp'] = 'mlp',
                 n_value_bins: int | None = 100,
                 pos_learnable: bool = False,
                 **base_encoder_kwargs
                 ):
        super().__init__(**base_encoder_kwargs)
        self.n_special_values = n_special_values
        self.count_encoding = count_encoding
        self.n_value_bins = n_value_bins

        # Initialize positional embeddings
        self.pos_embed = nn.Embedding(self.seq_len + 1, # include <pad>
                                      self.embed_dim,
                                      padding_idx=0)

        if not pos_learnable:
            # Prevent gradient updates and initialize with sincos
            # embedding
            self.pos_embed.weight.requires_grad = False
            pos_embed = get_1d_sincos_pos_embed(
                embed_dim=self.embed_dim,
                n_zero_pos=0,
                n_sincos_pos=self.seq_len)
            self.pos_embed.weight[1:].copy_(
                torch.from_numpy(pos_embed).float())

        # Initialize value embeddings and value embedding weight
        # projection layer
        if self.count_encoding == 'value_bins':
            self.value_embed = nn.Embedding(
                self.n_value_bins,
                self.embed_dim)
            self.special_value_embed = nn.Embedding(
                1 + (1 + self.n_special_values + 105 if self.api_version == 'v1' else 0), # include only <pad>
                self.embed_dim,
                padding_idx=0)
            self.value_emb_weights_projection = ValueEmbWeightsProjection(
                dim=self.n_value_bins)
        elif self.count_encoding == 'mlp':
            hidden_dim = int(self.embed_dim/2)
            self.value_embed = MLP(
                in_features=1, 
                hidden_features=hidden_dim,
                out_features=self.embed_dim,
                act_layer=nn.GELU)

    def forward(self,
                positions: torch.Tensor,
                segments: torch.Tensor,
                tokens: torch.Tensor,
                counts: torch.Tensor,
                masks: list[torch.Tensor] | torch.Tensor | None = None,
                masks_attention: torch.Tensor | None = None
                ) -> torch.Tensor:
        """
        Run encoder forward pass on a batch of cell graph sequences. For
        each observation in the batch return only embeddings for tokens
        included in the masks.

        Parameters
        -----------
        tokens:
            Tensor containing input gene tokens with shape (BATCH_SIZE,
            SEQ_LEN).
        segments:
            Tensor containing segment labels with shape (BATCH_SIZE,
            SEQ_LEN).
        counts:
            Tensor containing the counts corresponding to gene tokens
            with shape (BATCH_SIZE, SEQ_LEN).
        masks:
            List of N_MASKS tensors containing indices (within the
            sequence) of tokens to keep with shape (BATCH_SIZE,
            MASK_SIZE).
        masks_attention:
            An attention tensor that controls how different tokens
            attend to each other within a sequence.

        Returns
        -----------
        x:
            Embeddings of input tokens included in the masks with shape
            (BATCH_SIZE * N_MASKS, MIN_MASK_SIZE, EMBED_DIM), where
            MIN_MASK_SIZE is minimum mask size in the batch.    
        """
        
        # Format masks
        if masks is not None:
            if not isinstance(masks, list):
                masks = [masks]

        # Get embeddings for sequence of gene tokens and segments
        pos_emb = self.pos_embed(positions)
        seg_emb = self.seg_embed(segments)
        token_emb = self.token_embed(tokens)

        # Get value embeddings
        if self.count_encoding == 'value_bins':
            value_emb_weights = self.value_emb_weights_projection(
                counts.unsqueeze(dim=-1))
            value_emb = torch.matmul(
                value_emb_weights, self.value_embed.weight)
            zero_counts_mask = counts == 0.0 # assign pad to 0 counts
            zero_value_embed = self.special_value_embed(
                torch.tensor(0, device=tokens.device)).to(value_emb.dtype)
            value_emb[zero_counts_mask] = zero_value_embed
        elif self.count_encoding == 'mlp':
            value_emb = self.value_embed(counts.unsqueeze(dim=-1))           

        # Add gene token and segment embeddings to value embeddings
        x = pos_emb + seg_emb + token_emb + value_emb
        # B, N, D = x.shape # B: BATCH_SIZE, N: SEQ_LEN, D: EMBED_DIM

        # Mask token embeddings if masks are provided
        if masks is not None:
            x = apply_masks(x, masks)
        
        # Run forward prop
        for i, blk in enumerate(self.blocks):
            x = blk(x, masks=masks_attention)
        if self.norm is not None:
            x = self.norm(x)

        return x

    @torch.no_grad()
    def return_layer_emb(
            self,
            layer: int,
            positions: torch.Tensor,
            segments: torch.Tensor,
            tokens: torch.Tensor,
            counts: torch.Tensor,
            masks: list[torch.Tensor] | torch.Tensor | None = None,
            masks_attention: torch.Tensor | None = None,
            pad_neighborhood: bool = False,
            ) -> list[torch.Tensor]:
        """
        Run encoder forward pass on a batch of cell graph sequences,
        applying masks if provided, and return the embeddings of a
        specific layer.

        Parameters
        -----------
        layer:
            Index of the specific layer to be returned
        tokens:
            Tensor containing input gene tokens with shape (BATCH_SIZE,
            SEQ_LEN).
        segments:
            Tensor containing segment labels with shape (BATCH_SIZE,
            SEQ_LEN).
        counts:
            Tensor containing the counts corresponding to gene tokens
            with shape (BATCH_SIZE, SEQ_LEN).
        masks:
            List of N_MASKS tensors containing indices (within the
            sequence) of tokens to keep with shape (BATCH_SIZE,
            MASK_SIZE).
        masks_attention:
            An attention tensor that controls how different tokens
            attend to each other within a sequence.

        Returns
        -----------
        x:
            Embeddings of a specific layer with shape (BATCH_SIZE *
            N_MASKS, MIN_MASK_SIZE, EMBED_DIM), where MIN_MASK_SIZE is
            minimum mask size in the batch. 
        """

        # Format masks
        if masks is not None:
            if not isinstance(masks, list):
                masks = [masks]

        # Get embeddings for sequence of gene tokens and segments
        pos_emb = self.pos_embed(positions)
        seg_emb = self.seg_embed(segments)
        token_emb = self.token_embed(tokens)

        # Get value embeddings
        if self.count_encoding == 'value_bins':
            value_emb_weights = self.value_emb_weights_projection(
                counts.unsqueeze(dim=-1))
            value_emb = torch.matmul(
                value_emb_weights, self.value_embed.weight)
            zero_counts_mask = counts == 0.0 # assign padding to 0 counts
            zero_value_embed = self.special_value_embed(
                torch.tensor(0, device=tokens.device)).to(value_emb.dtype)
            value_emb[zero_counts_mask] = zero_value_embed
        elif self.count_encoding == 'mlp':
            value_emb = self.value_embed(counts.unsqueeze(dim=-1))  

        # Add gene token and segment embeddings to value embeddings
        x = pos_emb + seg_emb + token_emb + value_emb
        # B, N, D = x.shape # B: BATCH_SIZE, N: SEQ_LEN, D: EMBED_DIM

        # Pad special tokens
        x[:, :self.n_special_tokens, :] = 0
        masks_attention[:,
                        :,
                        :,
                        :self.n_special_tokens] = 0

        if pad_neighborhood:
            x[:, (self.n_special_tokens+self.seq_len_cell):, :] = 0

            masks_attention = masks_attention.expand(
                masks_attention.shape[0],
                1,
                masks_attention.shape[-1],
                masks_attention.shape[-1]).clone()

            # Mask neighborhood gene tokens for index cell gene tokens
            masks_attention[
                :,
                :,
                self.n_special_tokens:(self.n_special_tokens+self.seq_len_cell),
                (self.n_special_tokens+self.seq_len_cell):] = 0

        # Mask token embeddings if masks are provided
        if masks is not None:
            x = apply_masks(x, masks)

        # Run forward prop and store embeddings after each block
        n_blocks = len(self.blocks)
        for i, blk in enumerate(self.blocks):
            x = blk(x, masks=masks_attention)
            if (i == (n_blocks - 1)) and (self.norm is not None):
                x = self.norm(x)
            if i == (layer-1):
                break

        # Remove special tokens
        x = x[:, self.n_special_tokens:, :]

        return x


class GeneTransformerRankPredictor(GeneTransformerBasePredictor):
    """
    GeneTransformerRankPredictor class.
    """
    def __init__(self,
                 **base_encoder_kwargs
                 ):
        super().__init__(**base_encoder_kwargs)

        self.pos_embed_projection = nn.Linear(self.embed_dim,
                                              self.predictor_embed_dim,
                                              bias=True)

    def forward(self,
                z: torch.Tensor,
                pos_embed: torch.Tensor,
                segments: torch.Tensor,
                token_embed: torch.Tensor,
                masks_enc: list[torch.Tensor] | torch.Tensor,
                masks_pred: list[torch.Tensor] | torch.Tensor,
                masks_attention: torch.Tensor | None = None,
                ) -> torch.Tensor:
            """
            Run predictor forward pass for a batch of input tokens.

            Parameters
            -----------
            z:
                Embeddings from the encoder with shape (
                BATCH_SIZE*N_CONTEXT_MASKS, CONTEXT_MASK_SIZE,
                EMBED_DIM).
            pos_embed:
                Tensor containing positional embedding with shape
                (BATCH_SIZE, SEQ_LEN, EMBED_DIM).
            seg_embed:
                Tensor containing segment embeddings with shape
                (BATCH_SIZE, SEQ_LEN, EMBED_DIM).
            token_emb:
                Tensor containing token embeddings with shape
                (BATCH_SIZE, SEQ_LEN, EMBED_DIM).
            masks_enc:
                List of N_CONTEXT_MASKS tensors containing indices
                (within the sequence) of tokens to keep with shape
                (BATCH_SIZE, CONTEXT_MASK_SIZE).
            masks_pred:
                List of N_TARGET_MASKS tensors containing indices
                (within the sequence) of tokens to keep with shape
                (BATCH_SIZE, TARGET_MASK_SIZE).
            masks_attention:
                An attention mask that controls how different tokens
                attend to each other within a sequence.

            Returns
            -----------
            z:
                Embeddings of tokens included in the target masks with
                shape (BATCH_SIZE * N_CONTEXT_MASKS * N_TARGET_MASKS,
                TARGET_MASK_SIZE, EMBED_DIM).   
            """
            assert (masks_enc is not None) and (masks_pred is not None), \
                'Cannot run predictor without index masks.'

            # Format masks
            if not isinstance(masks_enc, list):
                masks_enc = [masks_enc]
            if not isinstance(masks_pred, list):
                masks_pred = [masks_pred]

            # Retrieve batch size
            B = len(z)

            # MLP projection layer
            z = self.predictor_embed(z)
            pos_embed = self.pos_embed_projection(pos_embed)
            token_embed = self.token_embed_projection(token_embed)

            # Get segment and special value embeddings
            seg_embed = self.seg_embed(segments)

            # Add positional embeddings to tokens from context masks
            # (only keep context mask indices and sum positional and
            # segment embeddings without token embeddings)
            z += apply_masks(pos_embed, masks_enc)
            z += apply_masks(seg_embed, masks_enc)
            _, N_ctxt, D = z.shape # N_ctxt: CONTEXT_MASK_SIZE, D: EMBED_DIM

            # Create positional embeddings for tokens from target masks
            # (only keep target mask indices and sum positional and
            # segment embeddings without token embeddings; the latter
            # are to be predicted)
            pos_embs = apply_masks(pos_embed, masks_pred)
            seg_embs = apply_masks(seg_embed, masks_pred)

            # Repeat mask token for all batches, masks and positions
            # from predictor masks
            pred_tokens = self.mask_token.repeat(
                pos_embs.size(0), # BATCH_SIZE * N_TARGET_MASKS
                pos_embs.size(1), # TARGET_MASK_SIZE
                1)

            # Add positional and segment embeddings to mask tokens                  
            pred_tokens += pos_embs + seg_embs

            # Repeat context embeddings for all target masks
            z = z.repeat(len(masks_pred), 1, 1)

            # Concatenate mask tokens and context embeddings of gene
            # tokens
            z = torch.cat([
                pred_tokens, # target gene tokens (incl. special tokens)
                # z # context gene tokens (incl. special tokens)
                z[:, self.n_special_tokens:, :] # context gene tokens (excl. special tokens)
                ], dim=1)

            # Run forward prop
            for blk in self.predictor_blocks:
                z = blk(z, masks=masks_attention)
            z = self.predictor_norm(z)

            # Return predictions for (target) mask tokens
            z = z[:, :pred_tokens.size(1), :]

            # MLP projection layer
            z = self.predictor_proj(z)

            return z


class GeneTransformerCountPredictor(GeneTransformerBasePredictor):
    """
    GeneTransformerCountPredictor class.

    Parameters
    -----------
    n_special_values:
        Number of special values (used for special tokens) for value
        embedding.
    """
    def __init__(
        self,
        n_special_values: int,
        **base_predictor_kwargs
        ):
        
        super().__init__(**base_predictor_kwargs)

        # Initialize special value embedding
        # Include <pad> and zero expression
        # TODO: why is n_special_tokens needed?
        self.n_special_values = n_special_values
        self.special_value_embed = nn.Embedding(
            2 + self.n_special_values + self.n_special_tokens,
            self.predictor_embed_dim,
            padding_idx=0)

    def forward(
        self,
        z: torch.Tensor,
        positions: torch.Tensor,
        segments: torch.Tensor,
        counts: torch.Tensor,
        masks_enc: list[torch.Tensor] | torch.Tensor,
        masks_pred: list[torch.Tensor] | torch.Tensor,
        masks_attention: torch.Tensor | None = None,
        ) -> torch.Tensor:
        """
        Run predictor forward pass for a batch of input tokens.

        Parameters
        -----------
        z:
            Embeddings from the encoder with shape (
            BATCH_SIZE*N_CONTEXT_MASKS, CONTEXT_MASK_SIZE, EMBED_DIM).
        token_embed:
            Token embeddings from the encoder.
        segments:
            Tensor containing segment labels with shape (BATCH_SIZE,
            SEQ_LEN).
        counts:
            Tensor containing counts with shape (BATCH_SIZE, SEQ_LEN).
        masks_enc:
            List of N_CONTEXT_MASKS tensors containing indices (within
            the sequence) of tokens to keep with shape (BATCH_SIZE,
            CONTEXT_MASK_SIZE).
        masks_pred:
            List of N_TARGET_MASKS tensors containing indices (within
            the sequence) of tokens to keep with shape (BATCH_SIZE,
            TARGET_MASK_SIZE).
        masks_attention:
            An attention mask that controls how different tokens attend
            to each other within a sequence.

        Returns
        -----------
        z:
            Embeddings of tokens included in the target masks with
            shape (BATCH_SIZE * N_CONTEXT_MASKS * N_TARGET_MASKS,
            TARGET_MASK_SIZE, EMBED_DIM).   
        """
        assert (masks_enc is not None) and (masks_pred is not None), \
            'Cannot run predictor without index masks.'

        # Format masks
        if not isinstance(masks_enc, list):
            masks_enc = [masks_enc]
        if not isinstance(masks_pred, list):
            masks_pred = [masks_pred]

        # Retrieve batch size
        B = len(z)

        # MLP projection layer
        z = self.predictor_embed(z)

        pos_embed = self.pos_embed(positions)

        # Get segment and special value embeddings
        seg_embed = self.seg_embed(segments)
        sp_value_embed = self.special_value_embed(
            counts[:, :self.n_special_tokens].int())

        # Add positional embeddings to tokens from context masks (only
        # keep context mask indices and sum positional and segment
        # embeddings without token embeddings)
        z += apply_masks(pos_embed, masks_enc)
        z += apply_masks(seg_embed, masks_enc)
        _, N_ctxt, D = z.shape # N_ctxt: CONTEXT_MASK_SIZE, D: EMBED_DIM

        # Create "positional" embeddings for tokens from target masks
        # (only keep target mask indices and sum token and segment
        # embeddings without value embeddings; the latter are to be
        # predicted)
        pos_embs = apply_masks(pos_embed, masks_pred)
        seg_embs = apply_masks(seg_embed, masks_pred)

        # Repeat mask token for all batches, masks and "positions" from
        # predictor masks
        pred_tokens = self.mask_token.repeat(
            pos_embs.size(0), # BATCH_SIZE * N_TARGET_MASKS
            pos_embs.size(1), # TARGET_MASK_SIZE
            1)

        # Add gene and segment embeddings to mask tokens                  
        pred_tokens += pos_embs + seg_embs 

        # Repeat context embeddings for all target masks
        z = z.repeat(len(masks_pred), 1, 1)

        # Concatenate mask tokens and context embeddings of gene tokens
        z = torch.cat([
            pred_tokens, # target gene tokens (excl. special tokens)
            z # context gene tokens (incl. special tokens)
            #z[:, self.n_special_tokens:, :] # context gene tokens (excl. special tokens)
            ], dim=1)

        # Run forward prop
        for blk in self.predictor_blocks:
            z = blk(z, masks=masks_attention)
        z = self.predictor_norm(z)

        # Return predictions for (target) mask tokens
        z = z[:, :pred_tokens.size(1), :]

        # MLP projection layer
        z = self.predictor_proj(z)

        return z


def init_gt_encoder(
        encoder_type: Literal['rank', 'counts'],
        **encoder_kwargs
        ) -> GeneTransformerRankEncoder | GeneTransformerCountEncoder:
    if encoder_type == 'rank':
        model = GeneTransformerRankEncoder(
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **encoder_kwargs)
    elif encoder_type == 'counts':
        model = GeneTransformerCountEncoder(
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **encoder_kwargs)

    return model


def init_gt_predictor(
        predictor_type: Literal['rank', 'counts'],
        n_special_values: int | None = None,
        **predictor_kwargs
        ) -> GeneTransformerRankPredictor | GeneTransformerCountPredictor:
    if predictor_type == 'rank':
        model = GeneTransformerRankPredictor(
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **predictor_kwargs)
    elif predictor_type == 'counts':
        model = GeneTransformerCountPredictor(
            n_special_values=n_special_values,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **predictor_kwargs)        

    return model
