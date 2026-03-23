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
from typing import Dict, Literal, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .modules import Attention, Block, MLP, ValueEmbWeightsProjection
from .utils import (get_1d_sincos_pos_embed,
                    get_1d_sincos_pos_embed_from_coord,
                    repeat_interleave_batch,
                    trunc_normal_)
from ..masks.utils import apply_masks

DEBUG = False


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
    cell_pos_enc:
        Cell position encoding. Either `segment` if cells are ranked
        based on distance to index cell or `coords` if relative
        positions to index cell are used.
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
                 cell_pos_enc: Literal['segment', 'coord'],
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
                 nz_spc: bool = False,
                 **kwargs
                 ):
        super().__init__()
        self.seq_len = seq_len
        self.n_segments = n_segments
        self.n_special_tokens = n_special_tokens
        self.seq_len_cell = (seq_len - n_special_tokens)//n_segments
        self.cell_pos_enc = cell_pos_enc
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.init_std = init_std
        self.api_version = api_version
        self.sep_gene_tokens_neb = sep_gene_tokens_neb
        self.nz_spc = nz_spc
            
        # Initialize token embeddings
        self.token_embed = nn.Embedding(
            vocab_size + (vocab_size if sep_gene_tokens_neb else 0), # already includes <pad>
            embed_dim,
            padding_idx=0)

        # Initialize segment embeddings
        if self.cell_pos_enc == 'segment':
            self.seg_embed = nn.Embedding(
                1 + n_segments + (105 if api_version == 'v1' else 0) + (self.n_special_tokens if self.nz_spc else 0), # include <pad>
                embed_dim,
                padding_idx=0)
        
            # Prevent gradient updates and initialize with sincos embedding,
            # including special segments
            self.seg_embed.weight.requires_grad = False
            seg_embed = get_1d_sincos_pos_embed(
                embed_dim=embed_dim,
                n_zero_pos=0,
                n_sincos_pos=n_segments + (105 if api_version == 'v1' else 0) + (self.n_special_tokens if self.nz_spc else 0))
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

        # Compute omega for segment embedding: 1 / 10000^{2i/dim}
        if self.cell_pos_enc == 'coord':
            self.coord_omega = torch.arange(
                embed_dim // 4, dtype=torch.float32)
            self.coord_omega = 1.0 / (
                10000 ** (self.coord_omega / (embed_dim / 4)))

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

    def _compute_layer_emb(
            self,
            x: torch.Tensor,
            attn_base: torch.Tensor | None,
            layers: Sequence[int],
            masks: list[torch.Tensor] | torch.Tensor | None,
            n_included_cells: int,
            ignore_spc_tokens: bool = False) -> dict[int, torch.Tensor]:
        """
        Helper function to return embeddings for either full context or
        masked cell context.
        """
        layers: list[int] = sorted({int(l) for l in layers})
        max_layer: int = max(layers)

        # Format masks
        if masks is not None and not isinstance(masks, list):
            masks = [masks]

        x = x.clone()

        attn = None
        if attn_base is not None:
            if attn_base.size(2) == 1: # [B, 1, 1, L] -> [B, 1, L, L]
                attn = attn_base.expand(-1, -1, attn_base.size(-1), -1).clone()
            else:
                attn = attn_base.clone()
            #if ignore_spc_tokens:
            #    # Never attend to special tokens
            #    if self.n_special_tokens > 0:
            #        attn[:, :, :, :self.n_special_tokens] = False
            # Block attention from included cell queries to excluded cell keys
            if n_included_cells:
                attn[
                    :,
                    :,
                    self.n_special_tokens: (self.n_special_tokens + self.seq_len_cell * n_included_cells),
                    (self.n_special_tokens + self.seq_len_cell * n_included_cells):] = False

        if n_included_cells:
            x[:, (self.n_special_tokens + self.seq_len_cell * n_included_cells):, :] = 0

        # Mask token embeddings if masks are provided
        if masks is not None:
            x = apply_masks(x, masks)

        #print("MASKS:")
        #print(masks)
        #print(x)

        if DEBUG:
            zero_rows = (x.abs().sum(dim=-1) == 0)   # (B, S)
            percentage_zero_rows = zero_rows.float().mean()
            print("Fraction of preblock X all-zero rows:", percentage_zero_rows)

        # Run forward prop and store embeddings for each specified layer
        out: dict[int, torch.Tensor] = {}
        valid_mask = x.ne(0).any(dim=-1)
        for i, blk in enumerate(self.blocks, start=1):
            x = blk(x, masks=attn)
            if i == len(self.blocks) and (self.norm is not None):
                # Apply norm only for last layer as in training
                x = self.norm(x)
                if DEBUG:
                    x = x * valid_mask.unsqueeze(-1)
                    zero_rows = (x.abs().sum(dim=-1) == 0)   # (B, S)

                    percentage_zero_rows = zero_rows.float().mean()
                    print("Fraction of afterblock X all-zero rows:", percentage_zero_rows)                
            if i in layers:
                # Remove special tokens from output
                out[i] = x[:, self.n_special_tokens:, :]
            if i == max_layer:
                break

        return out

    def _get_seg_emb(
            self,
            batch: dict):
        """
        Helper function to get segment embeddings.
        """
        if self.cell_pos_enc == 'segment':
            seg_emb = self.seg_embed(batch['segments'])
        elif self.cell_pos_enc == 'coord':
            rel_x_coord_emb = get_1d_sincos_pos_embed_from_coord(
                embed_dim=self.embed_dim // 2,
                omega=self.coord_omega,
                coord=batch['rel_x_coords'])
            rel_y_coord_emb = get_1d_sincos_pos_embed_from_coord(
                embed_dim=self.embed_dim // 2,
                omega=self.coord_omega,
                coord=batch['rel_y_coords'])
            seg_emb = torch.cat(
                [rel_x_coord_emb, rel_y_coord_emb], dim=-1)
        else:
            raise ValueError(
                f"Unknown cell_pos_enc: {self.cell_pos_enc}.")

        return seg_emb

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
    def return_layer_emb() -> tuple[
            dict[int, torch.Tensor], dict[int, torch.Tensor] | None]:
        """
        Encoder-specific logic for returning embeddings from multiple
        layers during inference.
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
    cell_pos_enc:
        Cell position encoding. Either `segment` if cells are ranked
        based on distance to index cell or `coords` if relative
        positions to index cell are used.
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
                 cell_pos_enc: Literal['segment', 'coord'],
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
                 nz_spc: bool = False,
                 new_spc: bool = False,
                 **kwargs
                 ):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.n_special_tokens = n_special_tokens
        self.cell_pos_enc = cell_pos_enc
        self.predictor_embed_dim = predictor_embed_dim
        self.num_heads = num_heads
        self.init_std = init_std
        self.api_version = api_version
        self.nz_spc = nz_spc
        self.new_spc = new_spc

        # Initialize segment embeddings
        if self.cell_pos_enc == 'segment':
            self.seg_embed = nn.Embedding(
                1 + n_segments + (105 if api_version == 'v1' else 0) + (1 if self.nz_spc else 0), # include <pad>
                predictor_embed_dim,
                padding_idx=0)
            
            # Prevent gradient updates and initialize with sincos embedding,
            # including special segments
            self.seg_embed.weight.requires_grad = False
            seg_embed = get_1d_sincos_pos_embed(
                embed_dim=predictor_embed_dim,
                n_zero_pos=0,
                n_sincos_pos=n_segments + (105 if api_version == 'v1' else 0) + (1 if self.nz_spc else 0))
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

        # Initialize mask token weights (not used)
        # trunc_normal_(self.mask_token, std=self.init_std)
        
        # Initialize layer weights
        self.apply(self._init_weights)
        self._rescale_blocks()

        # Compute omega for segment embedding: 1 / 10000^{2i/dim}
        if self.cell_pos_enc == 'coord':
            self.coord_omega = torch.arange(
                predictor_embed_dim // 4, dtype=torch.float32)
            self.coord_omega = 1.0 / (
                10000 ** (self.coord_omega / (predictor_embed_dim / 4)))

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

    def _get_seg_emb(
            self,
            batch: dict):
        """
        Helper function to get segment embeddings.
        """
        if self.cell_pos_enc == 'segment':
            seg_emb = self.seg_embed(batch['segments'])
        elif self.cell_pos_enc == 'coord':
            rel_x_coord_emb = get_1d_sincos_pos_embed_from_coord(
                embed_dim=self.predictor_embed_dim // 2,
                omega=self.coord_omega,
                coord=batch['rel_x_coords'])
            rel_y_coord_emb = get_1d_sincos_pos_embed_from_coord(
                embed_dim=self.predictor_embed_dim // 2,
                omega=self.coord_omega,
                coord=batch['rel_y_coords'])
            seg_emb = torch.cat(
                [rel_x_coord_emb, rel_y_coord_emb], dim=-1)
        else:
            raise ValueError(
                f"Unknown cell_pos_enc: {self.cell_pos_enc}.")

        return seg_emb

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
                batch: dict[torch.Tensor],
                masks: list[torch.Tensor] | torch.Tensor | None = None,
                masks_attention: torch.Tensor | None = None 
                ) -> tuple[torch.Tensor, dict]:
            """
            Run encoder forward pass on a batch of input token
            sequences. For each observation in the batch return only
            embeddings for tokens included in the masks.

            Parameters
            -----------
            batch:
                Dictionary containing sequence:
                - positions: Tensor containing positions with shape
                (BATCH_SIZE, SEQ_LEN).
                - segments: Tensor containing segment labels with shape
                (BATCH_SIZE, SEQ_LEN).
                - tokens: Tensor containing input gene tokens with shape
                (BATCH_SIZE, SEQ_LEN).
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
            batch:
                Updated sequence dictionary. Here, no updates are done but for
                API consistency this is kept.  
            """
            # Format masks
            if masks is not None:
                if not isinstance(masks, list):
                    masks = [masks]

            # Get positional, segment and token embeddings (excl.
            # special tokens)
            pos_emb = self.pos_embed(batch['positions'])
            seg_emb = self._get_seg_emb(batch)

            token_emb = self.token_embed(batch['tokens'])
            
            # Add positional and segment embeddings to token embeddings
            x = seg_emb + pos_emb + token_emb
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

            return x, token_emb

    @torch.inference_mode()
    def return_layer_emb(
            self,
            layers: Sequence[int],
            batch: Mapping[str, torch.Tensor],
            masks: list[torch.Tensor] | torch.Tensor | None = None,
            masks_attention: torch.Tensor | None = None,
            need_cell_only_context: bool = True,
            ignore_spc_tokens: bool = True,
            ) -> tuple[
                dict[int, torch.Tensor], dict[int, torch.Tensor] | None]:
        """
        Run encoder forward pass on a batch of cell graph sequences,
        applying masks if provided, and return the embeddings for
        multiple layers.

        Parameters
        -----------
        layers:
            1-based indices of returned layers (e.g., [4, 8, 12]).
        batch:
            Dictionary containing:
            - 'segments' (if cell_pos_enc == 'segment'): Tensor containing
              segment labels with shape (B, N).
            - 'rel_x_coords' (if cell_pos_enc == 'coord'): Tensor containing
              relative x-coordinates with shape (B, N).
            - 'rel_y_coords' (if cell_pos_enc == 'coord'): Tensor containing
              relative y-coordinates with shape (B, N).
            - 'positions': Tensor containing positions with shape(B, N).
            - 'tokens': Tensor containing input gene tokens with shape
              (B, N).
        masks:
            List of N_MASKS tensors containing indices (within the
            sequence) of tokens to keep with shape (B, M).
        masks_attention:
            An attention tensor that controls how different tokens
            attend to each other within a sequence.
        need_cell_only_context:
            If `True`, also run a second pass where queries in
            [0:seq_len_cell) cannot attend to keys in [seq_len_cell:).

        Returns
        -----------
        (full_ctx, cell_only_ctx)
          full_ctx : {layer_idx: Tensor[B, L_no_special, D]}
          cell_only_ctx : {layer_idx: Tensor[B, L_no_special, D]} or None
        """
        if not layers:
            raise ValueError(
                "Layers must be a non-empty sequence of positive integers.")

        # Get embeddings for sequence of gene tokens, positions and segments
        token_emb = self.token_embed(batch["tokens"])
        pos_emb = self.pos_embed(batch['positions'])
        seg_emb = self._get_seg_emb(batch)

        # Add segment and positional embeddings to token embeddings
        x = seg_emb + pos_emb + token_emb # [B, L, D]

        # Remove special token contents
        if ignore_spc_tokens:
            if self.n_special_tokens:
                x[:, :self.n_special_tokens, :] = 0

        full_ctx: dict[int, torch.Tensor] = self._compute_layer_emb(
            x,
            masks_attention,
            layers,
            masks,
            cell_only=False,
            ignore_spc_tokens=ignore_spc_tokens)
        cell_only_ctx: dict[int, torch.Tensor] = self._compute_layer_emb(
            x,
            masks_attention,
            layers,
            masks,
            cell_only=True,
            ignore_spc_tokens=ignore_spc_tokens) if need_cell_only_context else None

        return full_ctx, cell_only_ctx


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
                 mlp_bias: bool = True,
                 n_value_bins: int | None = 100,
                 **base_encoder_kwargs
                 ):
        super().__init__(**base_encoder_kwargs)
        self.n_special_values = n_special_values
        self.count_encoding = count_encoding
        self.mlp_bias = mlp_bias
        self.n_value_bins = n_value_bins

        # Initialize value embeddings and value embedding weight
        # projection layer
        if self.count_encoding == 'value_bins':
            self.value_embed = nn.Embedding(
                self.n_value_bins,
                self.embed_dim)
            if not self.nz_spc:
                self.special_value_embed = nn.Embedding(
                    1 + 1 + self.n_special_values + (105 if self.api_version == 'v1' else 0), # include only <pad>
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
                bias=self.mlp_bias,
                act_layer=nn.GELU)
        if self.nz_spc:
            self.special_value_embed = nn.Embedding(
                1 + 1 + self.n_special_values + (105 if self.api_version == 'v1' else 0),
                self.embed_dim,
                padding_idx=0)

    def forward(self,
                batch: dict[torch.Tensor],
                masks: list[torch.Tensor] | torch.Tensor | None = None,
                masks_attention: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, dict]:
        """
        Run encoder forward pass on a batch of cell graph sequences. For
        each observation in the batch return only embeddings for tokens
        included in the masks.

        Parameters
        -----------
        batch:
            Dictionary containing sequence:
            - segments: Tensor containing segment labels with shape
              (BATCH_SIZE, SEQ_LEN).
            - tokens: Tensor containing input gene tokens with shape
              (BATCH_SIZE, SEQ_LEN).
            - counts: Tensor containing the counts corresponding to gene
              tokens with shape (BATCH_SIZE, SEQ_LEN).
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
        batch:
            Updated sequence dictionary with added token embeddings. 
        """
        
        # Format masks
        if masks is not None:
            if not isinstance(masks, list):
                masks = [masks]

        # Get embeddings for sequence of gene tokens and segments
        seg_emb = self._get_seg_emb(batch)
        token_emb = self.token_embed(batch['tokens'])

        # Get value embeddings
        if self.count_encoding == 'value_bins':
            # [B, L, BINS] x [BINS, D] -> [B, L, D]
            value_emb_weights = self.value_emb_weights_projection(
                batch['values'].unsqueeze(-1))
            value_emb = value_emb_weights @ self.value_embed.weight

            if self.nz_spc:
                # Assign padding value embedding to 0 counts 
                zero_counts_mask = batch['values'] == 0.0
                zero_value_embed = self.special_value_embed(
                    torch.tensor(0, device=batch['tokens'].device)).to(value_emb.dtype)
                value_emb[zero_counts_mask] = zero_value_embed

                # Assign special value embeddings to special tokens
                sp_value_embed = self.special_value_embed(
                    batch['values'][:, :self.n_special_tokens].int()).to(
                        value_emb.dtype)
                value_emb[:, :self.n_special_tokens, :] = sp_value_embed
            else:
                # Assign padding to 0 counts
                zero_counts_mask = (batch['values'] == 0)
                if zero_counts_mask.any():
                    value_emb = torch.where(
                        zero_counts_mask.unsqueeze(-1),
                        self.special_value_embed.weight[0].expand_as(value_emb),
                        value_emb)

        elif self.count_encoding == 'mlp':
            value_emb = self.value_embed(batch['values'].unsqueeze(-1))

            if self.nz_spc:
                # Assign special value embeddings to special tokens
                sp_value_embed = self.special_value_embed(
                    batch['values'][:, :self.n_special_tokens].int()).to(
                        value_emb.dtype)
                value_emb[:, :self.n_special_tokens, :] = sp_value_embed
        else:
            raise ValueError(
                f"Unknown count_encoding: {self.count_encoding}.")

        # Add gene token and segment embeddings to value embeddings
        x = seg_emb + token_emb + value_emb
        # B, N, D = x.shape # B: BATCH_SIZE, N: SEQ_LEN, D: EMBED_DIM

        # Mask token embeddings if masks are provided
        if masks is not None:
            x = apply_masks(x, masks)
        
        # Run forward prop
        for i, blk in enumerate(self.blocks):
            x = blk(x, masks=masks_attention)
        if self.norm is not None:
            x = self.norm(x)

        return x, token_emb

    @torch.inference_mode()
    def return_layer_emb(
            self,
            layers: Sequence[int],
            batch: Mapping[str, torch.Tensor],
            masks: list[torch.Tensor] | torch.Tensor | None = None,
            masks_attention: torch.Tensor | None = None,
            need_cell_only_context: bool = True,
            ignore_spc_tokens: bool = True,
            ) -> tuple[
                dict[int, torch.Tensor], dict[int, torch.Tensor] | None]:
        """
        Run encoder forward pass on a batch of cell graph sequences,
        applying masks if provided, and return the embeddings for
        multiple layers.

        Parameters
        -----------
        layers:
            1-based indices of returned layers (e.g., [4, 8, 12]).
        batch:
            Dictionary containing:
            - 'segments' (if cell_pos_enc == 'segment'): Tensor containing
              segment labels with shape (B, N).
            - 'rel_x_coords' (if cell_pos_enc == 'coord'): Tensor containing
              relative x-coordinates with shape (B, N).
            - 'rel_y_coords' (if cell_pos_enc == 'coord'): Tensor containing
              relative y-coordinates with shape (B, N).
            - 'tokens': Tensor containing input gene tokens with shape
              (B, N).
            - 'values': Tensor containing the counts corresponding to gene
              tokens with shape (B, N).
        masks:
            List of N_MASKS tensors containing indices (within the
            sequence) of tokens to keep with shape (B, M).
        masks_attention:
            An attention tensor that controls how different tokens
            attend to each other within a sequence.
        need_cell_only_context:
            If `True`, also run a second pass where queries in
            [0:seq_len_cell) cannot attend to keys in [seq_len_cell:).

        Returns
        -----------
        (full_ctx, cell_only_ctx)
          full_ctx : {layer_idx: Tensor[B, L_no_special, D]}
          cell_only_ctx : {layer_idx: Tensor[B, L_no_special, D]} or None
        """
        if not layers:
            raise ValueError(
                "Layers must be a non-empty sequence of positive integers.")

        # Get embeddings for sequence of gene tokens and segments
        token_emb = self.token_embed(batch['tokens'])
        seg_emb = self._get_seg_emb(batch)

        # Get value embeddings
        if self.count_encoding == 'value_bins':
            # [B, L, BINS] x [BINS, D] -> [B, L, D]
            value_emb_weights = self.value_emb_weights_projection(
                batch['values'].unsqueeze(-1))
            value_emb = value_emb_weights @ self.value_embed.weight

            if self.nz_spc:
                # Assign padding value embedding to 0 counts 
                zero_counts_mask = batch['values'] == 0.0
                zero_value_embed = self.special_value_embed(
                    torch.tensor(0, device=batch['tokens'].device)).to(value_emb.dtype)
                value_emb[zero_counts_mask] = zero_value_embed

                # Assign special value embeddings to special tokens
                sp_value_embed = self.special_value_embed(
                    batch['values'][:, :self.n_special_tokens].int()).to(
                        value_emb.dtype)
                value_emb[:, :self.n_special_tokens, :] = sp_value_embed
            else:
                # Assign padding to 0 counts
                zero_counts_mask = (batch['values'] == 0)
                if zero_counts_mask.any():
                    value_emb = torch.where(
                        zero_counts_mask.unsqueeze(-1),
                        self.special_value_embed.weight[0].expand_as(value_emb),
                        value_emb)

        elif self.count_encoding == 'mlp':
            value_emb = self.value_embed(batch['values'].unsqueeze(-1))

            if self.nz_spc:
                # Assign special value embeddings to special tokens
                sp_value_embed = self.special_value_embed(
                    batch['values'][:, :self.n_special_tokens].int()).to(
                        value_emb.dtype)
                value_emb[:, :self.n_special_tokens, :] = sp_value_embed
        else:
            raise ValueError(
                f"Unknown count_encoding: {self.count_encoding}.")

        # Add segment and gene token embeddings to value embeddings
        x = seg_emb + token_emb + value_emb # [B, L, D]

        # Remove special token contents
        # if ignore_spc_tokens:
        #    if self.n_special_tokens:
        #        x[:, :self.n_special_tokens, :] = 0

        full_ctx: dict[int, torch.Tensor] = self._compute_layer_emb(
            x,
            masks_attention,
            layers,
            masks,
            cell_only=False,
            ignore_spc_tokens=ignore_spc_tokens)
        cell_only_ctx: dict[int, torch.Tensor] = self._compute_layer_emb(
            x,
            masks_attention,
            layers,
            masks,
            cell_only=True,
            ignore_spc_tokens=ignore_spc_tokens) if need_cell_only_context else None

        return full_ctx, cell_only_ctx


class GeneTransformerCombinedEncoder(GeneTransformerBaseEncoder):
    """
    GeneTransformerCombinedEncoder class to encode contexts or targets
    using gene tokens and gene expression counts.

    Parameters
    -----------
    count_encoding:
        Encoding module for counts. Can be either `value_bins`
        (scFoundation count encoding) or `mlp` (2 layer MLP).
    n_value_bins:
        Number of value bins if `value_bins` count encoding is used.
    pos_learnable:
        If `True`, positional embeddings are learnable, otherwise use
        sin cos positional embeddings.
    """
    def __init__(self,
                 n_special_values: int,
                 count_encoding: Literal['value_bins', 'mlp'] = 'mlp',
                 mlp_bias: bool = True,
                 n_value_bins: int | None = 100,
                 pos_learnable: bool = False,
                 **base_encoder_kwargs
                 ):
        super().__init__(**base_encoder_kwargs)
        self.n_special_values = n_special_values
        self.count_encoding = count_encoding
        self.mlp_bias = mlp_bias
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
            if not self.nz_spc:
                self.special_value_embed = nn.Embedding(
                    1 + 1 + self.n_special_values + (105 if self.api_version == 'v1' else 0), # include only <pad>
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
                bias=self.mlp_bias,
                act_layer=nn.GELU)
        if self.nz_spc:
            self.special_value_embed = nn.Embedding(
                1 + 1 + self.n_special_values + (105 if self.api_version == 'v1' else 0),
                self.embed_dim,
                padding_idx=0)

    def forward(self,
                batch: dict[torch.Tensor],
                masks: list[torch.Tensor] | torch.Tensor | None = None,
                masks_attention: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, dict]:
        """
        Run encoder forward pass on a batch of cell graph sequences. For
        each observation in the batch return only embeddings for tokens
        included in the masks.

        Parameters
        -----------
        batch:
            Dictionary containing sequence:
            - positions: Tensor containing positions with shape
              (BATCH_SIZE, SEQ_LEN).
            - segments: Tensor containing segment labels with shape
              (BATCH_SIZE, SEQ_LEN).
            - tokens: Tensor containing input gene tokens with shape
              (BATCH_SIZE, SEQ_LEN).
            - counts: Tensor containing the counts corresponding to gene
              tokens with shape (BATCH_SIZE, SEQ_LEN).
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
        batch:
            Updated sequence dictionary. Here, no updates are done but for
            API consistency this is kept.  
        """
        
        # Format masks
        if masks is not None:
            if not isinstance(masks, list):
                masks = [masks]

        # Get embeddings for positions, segments and gene tokens
        pos_emb = self.pos_embed(batch['positions'])
        seg_emb = self._get_seg_emb(batch)
        token_emb = self.token_embed(batch['tokens'])

        # Get value embeddings
        if self.count_encoding == 'value_bins':
            # [B, L, BINS] x [BINS, D] -> [B, L, D]
            value_emb_weights = self.value_emb_weights_projection(
                batch['values'].unsqueeze(-1))
            value_emb = value_emb_weights @ self.value_embed.weight

            if self.nz_spc:
                # Assign padding value embedding to 0 counts 
                zero_counts_mask = batch['values'] == 0.0
                zero_value_embed = self.special_value_embed(
                    torch.tensor(0, device=batch['tokens'].device)).to(value_emb.dtype)
                value_emb[zero_counts_mask] = zero_value_embed

                # Assign special value embeddings to special tokens
                sp_value_embed = self.special_value_embed(
                    batch['values'][:, :self.n_special_tokens].int()).to(
                        value_emb.dtype)
                value_emb[:, :self.n_special_tokens, :] = sp_value_embed
            else:
                # Assign padding to 0 counts
                zero_counts_mask = (batch['values'] == 0)
                if zero_counts_mask.any():
                    value_emb = torch.where(
                        zero_counts_mask.unsqueeze(-1),
                        self.special_value_embed.weight[0].expand_as(value_emb),
                        value_emb)

        elif self.count_encoding == 'mlp':
            value_emb = self.value_embed(batch['values'].unsqueeze(-1))

            if self.nz_spc:
                # Assign special value embeddings to special tokens
                sp_value_embed = self.special_value_embed(
                    batch['values'][:, :self.n_special_tokens].int()).to(
                        value_emb.dtype)
                value_emb[:, :self.n_special_tokens, :] = sp_value_embed
        else:
            raise ValueError(
                f"Unknown count_encoding: {self.count_encoding}.")

        # Add positional, segment, and gene embeddings to value embeddings
        x = seg_emb + pos_emb + token_emb + value_emb
        # B, N, D = x.shape # B: BATCH_SIZE, N: SEQ_LEN, D: EMBED_DIM

        # Mask token embeddings if masks are provided
        if masks is not None:
            x = apply_masks(x, masks)
        
        # Run forward prop
        for i, blk in enumerate(self.blocks):
            x = blk(x, masks=masks_attention)
        if self.norm is not None:
            x = self.norm(x)

        return x, token_emb

    @torch.inference_mode()
    def return_layer_emb(
            self,
            layers: Sequence[int],
            batch: Mapping[str, torch.Tensor],
            masks: list[torch.Tensor] | torch.Tensor | None = None,
            masks_attention: torch.Tensor | None = None,
            n_included_cells_list: list[int] = [],
            ignore_spc_tokens: bool = True,
            ) -> dict[int, dict[int, torch.Tensor]]:
        """
        Run encoder forward pass on a batch of cell graph sequences,
        applying masks if provided, and return the embeddings for
        multiple layers.

        Parameters
        -----------
        layers:
            1-based indices of returned layers (e.g., [4, 8, 12]).
        batch:
            Dictionary containing:
            - 'segments' (if cell_pos_enc == 'segment'): Tensor containing
              segment labels with shape (B, N).
            - 'rel_x_coords' (if cell_pos_enc == 'coord'): Tensor containing
              relative x-coordinates with shape (B, N).
            - 'rel_y_coords' (if cell_pos_enc == 'coord'): Tensor containing
              relative y-coordinates with shape (B, N).
            - 'positions': Tensor containing positions with shape(B, N).
            - 'tokens': Tensor containing input gene tokens with shape
              (B, N).
            - 'values': Tensor containing the counts corresponding to gene
              tokens with shape (B, N).
        masks:
            List of N_MASKS tensors containing indices (within the
            sequence) of tokens to keep with shape (B, M).
        masks_attention:
            An attention tensor that controls how different tokens
            attend to each other within a sequence.
        need_cell_only_context:
            If `True`, also run a second pass where queries in
            [0:seq_len_cell) cannot attend to keys in [seq_len_cell:).

        Returns
        -----------
        (full_ctx, cell_only_ctx)
          full_ctx : {layer_idx: Tensor[B, L_no_special, D]}
          cell_only_ctx : {layer_idx: Tensor[B, L_no_special, D]} or None
        """
        if not layers:
            raise ValueError(
                "Layers must be a non-empty sequence of positive integers.")

        # Get embeddings for sequence of gene tokens, positions and segments
        token_emb = self.token_embed(batch['tokens'])
        pos_emb = self.pos_embed(batch['positions'])
        seg_emb = self._get_seg_emb(batch)
        
        if DEBUG:
            proportion_zero = (batch['tokens'] == 0).float().mean()
            print("Proportion of zero tokens:", proportion_zero)

        # Get value embeddings
        if self.count_encoding == 'value_bins':
            # [B, L, BINS] x [BINS, D] -> [B, L, D]
            value_emb_weights = self.value_emb_weights_projection(
                batch['values'].unsqueeze(-1))
            value_emb = value_emb_weights @ self.value_embed.weight

            if self.nz_spc:
                # Assign padding value embedding to 0 counts 
                zero_counts_mask = batch['values'] == 0.0
                zero_value_embed = self.special_value_embed(
                    torch.tensor(0, device=batch['tokens'].device)).to(value_emb.dtype)
                value_emb[zero_counts_mask] = zero_value_embed

                # Assign special value embeddings to special tokens
                sp_value_embed = self.special_value_embed(
                    batch['values'][:, :self.n_special_tokens].int()).to(
                        value_emb.dtype)
                value_emb[:, :self.n_special_tokens, :] = sp_value_embed
            else:
                # Assign padding to 0 counts
                zero_counts_mask = (batch['values'] == 0)
                if zero_counts_mask.any():
                    value_emb = torch.where(
                        zero_counts_mask.unsqueeze(-1),
                        self.special_value_embed.weight[0].expand_as(value_emb),
                        value_emb)

        elif self.count_encoding == 'mlp':
            value_emb = self.value_embed(batch['values'].unsqueeze(-1))

            if self.nz_spc:
                # Assign special value embeddings to special tokens
                sp_value_embed = self.special_value_embed(
                    batch['values'][:, :self.n_special_tokens].int()).to(
                        value_emb.dtype)
                value_emb[:, :self.n_special_tokens, :] = sp_value_embed
        else:
            raise ValueError(
                f"Unknown count_encoding: {self.count_encoding}.")

        # Add segment, positional, and gene embeddings to value embeddings
        x = seg_emb + pos_emb + token_emb + value_emb # [B, L, D]

        if DEBUG:

            def zero_row_percentage(t):
                # t: (B, L, D)
                zero_rows = t.eq(0).all(dim=-1)   # (B, L)
                return zero_rows.float().mean()   # scalar        

            print("seg_emb zero rows:", zero_row_percentage(seg_emb))
            print("pos_emb zero rows:", zero_row_percentage(pos_emb))
            print("token_emb zero rows:", zero_row_percentage(token_emb))
            print("value_emb zero rows:", zero_row_percentage(value_emb))

        # Remove special token contents
        #if ignore_spc_tokens:
        #    if self.n_special_tokens:
        #        x[:, :self.n_special_tokens, :] = 0

        ctx_layer_gene_emb = {}
        for n_included_cells in n_included_cells_list:
            layer_gene_emb: dict[int, torch.Tensor] = self._compute_layer_emb(
                x,
                masks_attention,
                layers,
                masks,
                n_included_cells=n_included_cells,
                ignore_spc_tokens=ignore_spc_tokens)
            ctx_layer_gene_emb[n_included_cells] = layer_gene_emb

        return ctx_layer_gene_emb


class GeneTransformerRankPredictor(GeneTransformerBasePredictor):
    """
    GeneTransformerRankPredictor class.

    Parameters
    -----------
    pos_learnable:
        If `True`, positional embeddings are learnable, otherwise use
        sin cos positional embeddings.
    """
    def __init__(self,
                 pos_learnable: bool = False,
                 **base_predictor_kwargs
                 ):
        super().__init__(**base_predictor_kwargs)

        # Initialize positional embeddings
        self.pos_embed = nn.Embedding(self.seq_len + 1, # include <pad>
                                      self.predictor_embed_dim,
                                      padding_idx=0)

        if not pos_learnable:
            # Prevent gradient updates and initialize with sincos embedding
            self.pos_embed.weight.requires_grad = False
            pos_embed = get_1d_sincos_pos_embed(
                embed_dim=self.predictor_embed_dim,
                n_zero_pos=0,
                n_sincos_pos=self.seq_len)
            self.pos_embed.weight[1:].copy_(
                torch.from_numpy(pos_embed).float())

    def forward(self,
                z: torch.Tensor,
                token_emb: torch.Tensor,
                batch: dict[torch.Tensor],
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
            token_emb:
            batch:
                Dictionary containing sequence:
                - positions: Tensor containing positions with shape
                  (BATCH_SIZE, SEQ_LEN).
                - segments: Tensor containing segment labels with shape
                  (BATCH_SIZE, SEQ_LEN).
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

            # Get positional and segment embeddings
            pos_embed = self.pos_embed(batch['positions'])
            seg_embed = self._get_seg_emb(batch)

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

            if self.new_spc:
                # Concatenate mask tokens and context embeddings of gene tokens
                z = torch.cat([
                    z[:, :self.n_special_tokens, :],
                    pred_tokens[:, self.n_special_tokens:, :], # target gene tokens (incl. special tokens)
                    z[:, self.n_special_tokens:, :] # context gene tokens (excl. special tokens)
                    ], dim=1)
            else:
                # Concatenate mask tokens and context embeddings of gene tokens
                z = torch.cat([
                    pred_tokens, # target gene tokens (incl. special tokens)
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


class GeneTransformerCountPredictor(GeneTransformerBasePredictor):
    """
    GeneTransformerCountPredictor class.
    """
    def __init__(
        self,
        **base_predictor_kwargs
        ):
        
        super().__init__(**base_predictor_kwargs)

    def forward(
        self,
        z: torch.Tensor,
        token_emb: torch.Tensor,
        batch: dict[torch.Tensor],
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
        token_emb:
            Token embeddings from the encoder.
        batch:
            Dictionary containing sequence:
            - segments: Tensor containing segment labels with shape
              (BATCH_SIZE, SEQ_LEN).
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

        # Get gene and segment embeddings
        token_embed = self.token_embed_projection(token_emb)
        seg_embed = self._get_seg_emb(batch)

        # Add positional embeddings to tokens from context masks (only
        # keep context mask indices and sum positional and segment
        # embeddings without token embeddings)
        z += apply_masks(token_embed, masks_enc)
        z += apply_masks(seg_embed, masks_enc)
        _, N_ctxt, D = z.shape # N_ctxt: CONTEXT_MASK_SIZE, D: EMBED_DIM

        # Create "positional" embeddings for tokens from target masks
        # (only keep target mask indices and sum token and segment
        # embeddings without value embeddings; the latter are to be
        # predicted)
        token_embs = apply_masks(token_embed, masks_pred)
        seg_embs = apply_masks(seg_embed, masks_pred)

        # Repeat mask token for all batches, masks and "positions" from
        # predictor masks
        pred_tokens = self.mask_token.repeat(
            seg_embs.size(0), # BATCH_SIZE * N_TARGET_MASKS
            seg_embs.size(1), # TARGET_MASK_SIZE
            1)

        # Add gene and segment embeddings to mask tokens                  
        pred_tokens += token_embs + seg_embs

        # Repeat context embeddings for all target masks
        z = z.repeat(len(masks_pred), 1, 1)

        if self.new_spc:
            # Concatenate mask tokens and context embeddings of gene tokens
            z = torch.cat([
                z[:, :self.n_special_tokens, :],
                pred_tokens[:, self.n_special_tokens:, :], # target gene tokens (incl. special tokens)
                z[:, self.n_special_tokens:, :] # context gene tokens (excl. special tokens)
                ], dim=1)
        else:
            # Concatenate mask tokens and context embeddings of gene tokens
            z = torch.cat([
                pred_tokens, # target gene tokens (incl. special tokens)
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


class GeneTransformerCombinedPredictor(GeneTransformerBasePredictor):
    """
    GeneTransformerCombinedPredictor class.

    Parameters
    -----------
    predict_gene:
        If `True`, predict gene given rank, otherwise predict rank given
        gene.
    pos_learnable:
        If `True`, positional embeddings are learnable, otherwise use
        sin cos positional embeddings.
    """
    def __init__(
        self,
        predict_gene: bool = True,
        pos_learnable: bool = False,
        **base_predictor_kwargs
        ):
        
        super().__init__(**base_predictor_kwargs)
        self.predict_gene = predict_gene

        if self.predict_gene:
            # Initialize positional embeddings
            self.pos_embed = nn.Embedding(self.seq_len + 1, # include <pad>
                                          self.predictor_embed_dim,
                                          padding_idx=0)
            
            if not pos_learnable:
                # Prevent gradient updates and initialize with sincos
                # embedding
                self.pos_embed.weight.requires_grad = False
                pos_embed = get_1d_sincos_pos_embed(
                    embed_dim=self.predictor_embed_dim,
                    n_zero_pos=0,
                    n_sincos_pos=self.seq_len)
                self.pos_embed.weight[1:].copy_(
                    torch.from_numpy(pos_embed).float())

    def forward(
        self,
        z: torch.Tensor,
        token_emb: torch.Tensor,
        batch: dict[torch.Tensor],
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
        token_emb:
            Token embeddings from the encoder.
        batch:
            Dictionary containing sequence:
            - positions: Tensor containing positions with shape
              (BATCH_SIZE, SEQ_LEN).
            - segments: Tensor containing segment labels with shape
              (BATCH_SIZE, SEQ_LEN).
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

        if self.predict_gene:
            # Get positional embeddings
            pos_embed = self.pos_embed(batch['positions'])
        else:
            # Get gene embeddings
            token_embed = self.token_embed_projection(token_emb)

        # Get segment embeddings
        seg_embed = self._get_seg_emb(batch)

        # Add positional or gene embeddings to tokens from context masks (only
        # keep context mask indices and sum positional or gene and segment
        # embeddings without token embeddings)
        if self.predict_gene:
            z += apply_masks(pos_embed, masks_enc)
        else:
            z += apply_masks(token_embed, masks_enc)
        z += apply_masks(seg_embed, masks_enc)
        _, N_ctxt, D = z.shape # N_ctxt: CONTEXT_MASK_SIZE, D: EMBED_DIM

        # Create "positional" embeddings for tokens from target masks
        # (only keep target mask indices and sum token and segment
        # embeddings without value embeddings; the latter are to be
        # predicted)
        if self.predict_gene:
            pos_embs = apply_masks(pos_embed, masks_pred)
        else:
            token_embs = apply_masks(token_embed, masks_pred)
        seg_embs = apply_masks(seg_embed, masks_pred)

        # Repeat mask token for all batches, masks and "positions" from
        # predictor masks
        pred_tokens = self.mask_token.repeat(
            seg_embs.size(0), # BATCH_SIZE * N_TARGET_MASKS
            seg_embs.size(1), # TARGET_MASK_SIZE
            1)

        # Add position and segment embeddings to mask tokens
        if self.predict_gene:                  
            pred_tokens += pos_embs + seg_embs
        else:
            pred_tokens += token_embs + seg_embs

        # Repeat context embeddings for all target masks
        z = z.repeat(len(masks_pred), 1, 1)

        if self.new_spc:
            # Concatenate mask tokens and context embeddings of gene tokens
            z = torch.cat([
                z[:, :self.n_special_tokens, :],
                pred_tokens[:, self.n_special_tokens:, :], # target gene tokens (incl. special tokens)
                z[:, self.n_special_tokens:, :] # context gene tokens (excl. special tokens)
                ], dim=1)
        else:
            # Concatenate mask tokens and context embeddings of gene tokens
            z = torch.cat([
                pred_tokens, # target gene tokens (incl. special tokens)
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
        encoder_type: Literal['rank', 'counts', 'combined'],
        **encoder_kwargs
        ) -> GeneTransformerRankEncoder | GeneTransformerCountEncoder | GeneTransformerCombinedEncoder:
    """
    Initialize GeneTransformerEncoder based on encoder type.
    """
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
    elif encoder_type == 'combined':
        model = GeneTransformerCombinedEncoder(
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **encoder_kwargs)

    return model


def init_gt_predictor(
        predictor_type: Literal['rank', 'counts', 'combined'],
        n_special_values: int | None = None,
        **predictor_kwargs
        ) -> GeneTransformerRankPredictor | GeneTransformerCountPredictor | GeneTransformerCombinedPredictor:
    """
    Initialize GeneTransformerPredictor based on predictor type.
    """
    if predictor_type == 'rank':
        model = GeneTransformerRankPredictor(
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **predictor_kwargs)
    elif predictor_type == 'counts':
        model = GeneTransformerCountPredictor(
            n_special_values=n_special_values,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **predictor_kwargs)
    elif predictor_type == 'combined':
        model = GeneTransformerCombinedPredictor(
            n_special_values=n_special_values,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **predictor_kwargs)      

    return model