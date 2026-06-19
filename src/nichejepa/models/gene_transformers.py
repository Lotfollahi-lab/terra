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
from typing import Dict, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .adaln import zero_init_adaln_modulations
from .modules import Attention, Block, MLP, ValueEmbWeightsProjection
from .protein_init import (build_protein_init_token_embedding,
                           build_warm_start_token_embedding)
from .rope2d import RoPE2D


# Sentinel used by encoder/predictor _compute_cond when AdaLN is off:
# a None cond signals to Block.forward() that it should use its plain
# LayerNorm path (assuming Block was constructed without AdaLN). This
# keeps the conditioning-aware codepath uniformly None-able rather
# than requiring a parallel "no-cond" forward branch.
_NO_COND = None
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
        Cell position encoding. One of:
        - ``segment``: cells ranked by NN distance, fixed sincos by rank.
        - ``coord``: relative (x, y) sincos'd independently.
        - ``polar``: (log(1+r), theta) from (rel_x, rel_y), each sincos'd.
        - ``alibi``: segment input encoding + per-head distance-decaying
          attention bias.
        Older docstring text kept for reference: `segment` if cells are ranked
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
                 cell_pos_enc: Literal[
                     'none', 'segment', 'coord', 'polar', 'alibi',
                     'polar+alibi', 'laplacian', 'rope'],
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
                 protein_init_kwargs: dict | None = None,
                 laplacian_k: int = 8,
                 laplacian_sigma: float = 1.0,
                 rope_freq_scale: float = math.pi,
                 rope_rotation_augment: bool = True,
                 adaln_kwargs: dict | None = None,
                 rda_kwargs: dict | None = None,
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

        # Read-Depth-Aware (RDA) conditioning: scFoundation-style
        # per-cell depth embedding broadcast across each cell's
        # gene-token positions. Zero-initialized output head makes
        # this a no-op at step 0 even when ``enabled=True``, so
        # backward-compat is preserved.
        self.rda_enabled = bool(rda_kwargs) and rda_kwargs.get(
            'enabled', False)
        if self.rda_enabled:
            from .rda import DepthConditioning
            self.rda_module = DepthConditioning(
                embed_dim=embed_dim,
                hidden_dim=int((rda_kwargs or {}).get(
                    'hidden_dim', 32)),
                use_target_depth=bool((rda_kwargs or {}).get(
                    'use_target_depth', False)),
            )
        else:
            self.rda_module = None

        # Initialize token embeddings. When `protein_init_kwargs` is
        # provided, gene-token rows are sourced from a frozen ESM
        # protein-embedding matrix and adapted into `embed_dim` by a
        # learnable linear projection (UCE-style); special tokens and
        # missing-gene Ensembl IDs go through a separate learnable
        # embedding table. Otherwise a plain learnable nn.Embedding is
        # used, as before.
        effective_vocab_size = (
            vocab_size + (vocab_size if sep_gene_tokens_neb else 0))
        if protein_init_kwargs is not None:
            _pi_extra = {}
            if "gene_id_prefixes" in protein_init_kwargs:
                _pi_extra["gene_id_prefixes"] = tuple(
                    protein_init_kwargs["gene_id_prefixes"])
            mode = protein_init_kwargs.get("mode", "routing")
            if mode == "warm_start":
                # Identical architecture to baseline: a plain
                # nn.Embedding whose gene rows are pre-initialized
                # with PCA-reduced ESM. No routing module, no
                # LayerNorm, no projection in the forward path.
                self.token_embed = build_warm_start_token_embedding(
                    token_dict=protein_init_kwargs["token_dict"],
                    embedding_path=protein_init_kwargs["embedding_path"],
                    mapping_path=protein_init_kwargs["mapping_path"],
                    effective_vocab_size=effective_vocab_size,
                    embed_dim=embed_dim,
                    sep_gene_tokens_neb=sep_gene_tokens_neb,
                    padding_idx=0,
                    target_std=protein_init_kwargs.get(
                        "warm_start_target_std", 1.0),
                    seed=protein_init_kwargs.get("warm_start_seed", 0),
                    **_pi_extra,
                )
            elif mode == "routing":
                self.token_embed = build_protein_init_token_embedding(
                    token_dict=protein_init_kwargs["token_dict"],
                    embedding_path=protein_init_kwargs["embedding_path"],
                    mapping_path=protein_init_kwargs["mapping_path"],
                    effective_vocab_size=effective_vocab_size,
                    embed_dim=embed_dim,
                    sep_gene_tokens_neb=sep_gene_tokens_neb,
                    padding_idx=0,
                    init_std=init_std,
                    proj_bias=protein_init_kwargs.get("proj_bias", False),
                    use_layer_norm=protein_init_kwargs.get(
                        "use_layer_norm", True),
                    freeze_esm=protein_init_kwargs.get("freeze_esm", True),
                    **_pi_extra,
                )
            else:
                raise ValueError(
                    f"protein_init mode={mode!r} not recognized. "
                    "Expected 'routing' or 'warm_start'.")
        else:
            self.token_embed = nn.Embedding(
                effective_vocab_size,  # already includes <pad>
                embed_dim,
                padding_idx=0)

        # Initialize segment embeddings. Used by 'segment' mode (the
        # full positional signal) and also by 'alibi' mode (where it
        # still carries useful ordinal nearest-neighbor info at the
        # input; the distance-dependent signal is added at attention
        # time via the alibi bias). 'polar+alibi' uses polar input
        # encoding instead, so no segment table is needed there.
        if self.cell_pos_enc in ('segment', 'alibi'):
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

        # 2D rotary position embedding (RoPE). Shared across all
        # blocks of the encoder. Applied to q / k inside each
        # Attention module before SDPA, so attention logits depend
        # only on the relative cell positions. With 'rope' mode the
        # additive seg_emb contribution is zero -- all the positional
        # signal flows through attention.
        if self.cell_pos_enc == 'rope':
            head_dim = embed_dim // num_heads
            self.rope = RoPE2D(
                head_dim=head_dim,
                freq_scale=rope_freq_scale,
                rotation_augment=rope_rotation_augment,
            )
        else:
            self.rope = None

        # Adaptive LayerNorm (AdaLN) for per-batch conditioning. When
        # enabled, every transformer block's LayerNorms become AdaLN
        # modulated by a per-cell conditioning embedding looked up
        # from a batch ID at forward time. Default off: cond_dim None
        # -> Block uses plain LayerNorm and behaviour is unchanged.
        self.adaln_enabled = bool(adaln_kwargs) and adaln_kwargs.get(
            'enabled', False)
        if self.adaln_enabled:
            from .batch_labels import resolve_label_spec
            self.adaln_batch_embed_dim = int(adaln_kwargs.get(
                'batch_embed_dim', 64))
            # Resolve multi-key spec. Falls back to single-key
            # (legacy ``values[:, 0]`` path) if no ``keys`` set.
            spec = resolve_label_spec(adaln_kwargs)
            if not spec['keys']:
                # Legacy single-slot using values[:, 0]. CAVEAT: at
                # inference with ``pad_special_tokens=True`` (the
                # default for embed_dataset / infer.py), the spv_*
                # positions of the values tensor are zeroed, so
                # AdaLN would silently condition on batch_label=0
                # for every cell -- different from training. Migrate
                # to the metadata key path
                # (``adaln.keys=['batch_value']``) for inference-safe
                # behavior. Rank-0 gate so DDP runs don't print
                # n_ranks * n_modules duplicate warnings.
                import os as _adaln_os
                if int(_adaln_os.environ.get('RANK', '0')) == 0:
                    import logging as _adaln_logging
                    _adaln_logging.getLogger(
                        "nichejepa.models.gene_transformers"
                    ).warning(
                        "[AdaLN] using LEGACY single-key path "
                        "(reads values[:, 0]). At inference with "
                        "pad_special_tokens=True this silently "
                        "conditions on batch_label=0. Recommend "
                        "migrating to "
                        "``adaln.keys=['batch_value']`` + "
                        "matching n_classes / offsets.")
                self.adaln_keys = [None]
                self.adaln_offsets = [0]
                self.adaln_n_classes_per_key = [
                    int(adaln_kwargs['n_batches'])]
            else:
                if not spec['n_classes'] or any(
                        n <= 0 for n in spec['n_classes']):
                    raise ValueError(
                        "AdaLN multi-key spec: ``n_classes`` must be "
                        "set per key (one positive int per "
                        f"key). Got n_classes={spec['n_classes']} "
                        f"for keys={spec['keys']}.")
                self.adaln_keys = spec['keys']
                self.adaln_offsets = spec['offsets']
                self.adaln_n_classes_per_key = spec['n_classes']
            # One embedding table per key; per-key lookups are summed
            # to form the AdaLN cond vector. This is the same
            # additive-bias-by-slot structure as special_token_moe
            # but used as conditioning input instead of output bias.
            self.batch_emb_tables = nn.ModuleList([
                nn.Embedding(n, self.adaln_batch_embed_dim)
                for n in self.adaln_n_classes_per_key
            ])
            _block_cond_dim = self.adaln_batch_embed_dim
        else:
            self.adaln_keys = []
            self.adaln_offsets = []
            self.adaln_n_classes_per_key = []
            self.adaln_batch_embed_dim = 0
            self.batch_emb_tables = None
            _block_cond_dim = None

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
                  use_layer_norm=use_layer_norm,
                  rope=self.rope,
                  cond_dim=_block_cond_dim)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Initialize weights of layers
        self.apply(self._init_weights)
        self._rescale_blocks()
        # `self.apply(_init_weights)` above visits every nn.Linear --
        # including each AdaLN's modulation hypernetwork -- and re-
        # initializes it via trunc_normal_(self.init_std). That
        # overwrites the zero-init from AdaLN.__init__ and breaks the
        # AdaLN-at-step-0 == LayerNorm invariant. Restore it here.
        # No-op when AdaLN is disabled (no AdaLN modules to find).
        zero_init_adaln_modulations(self)

        # Compute omega for coord-based positional sincos: 1 / 10000^{2i/dim}
        # Used by 'coord' (sincos of rel_x, rel_y), 'polar' (sincos of
        # log(1+r), theta), and 'polar+alibi' (polar input + alibi bias).
        if self.cell_pos_enc in ('coord', 'polar', 'polar+alibi'):
            self.coord_omega = torch.arange(
                embed_dim // 4, dtype=torch.float32)
            self.coord_omega = 1.0 / (
                10000 ** (self.coord_omega / (embed_dim / 4)))

        # ALiBi-style attention bias: per-head slope buffer that scales
        # the negative pairwise distance into an additive attention
        # logit term. Computed once at construction and broadcast over
        # the batch dimension in the forward. Used by 'alibi' (with
        # segment input) and 'polar+alibi' (with polar input).
        if self.cell_pos_enc in ('alibi', 'polar+alibi'):
            slopes = self._get_alibi_slopes(num_heads)
            self.register_buffer("alibi_slopes", slopes, persistent=False)

        # Laplacian PE: per-batch eigendecomposition of the spatial-
        # graph Laplacian over the n_segments cells, with the bottom-k
        # non-trivial eigenvectors projected to embed_dim.
        if self.cell_pos_enc == 'laplacian':
            # Cap k at n_segments - 1 since the trivial zero
            # eigenvector is skipped and there are at most n_segments
            # eigenvectors total.
            self.laplacian_k = min(int(laplacian_k), n_segments - 1)
            if self.laplacian_k < 1:
                raise ValueError(
                    f"laplacian_k must be >= 1 (got {laplacian_k}). "
                    f"With n_segments={n_segments}, the max usable k is "
                    f"{n_segments - 1}.")
            self.laplacian_sigma = float(laplacian_sigma)
            # Learnable projection from eigenvector space to embed_dim.
            self.laplacian_proj = nn.Linear(
                self.laplacian_k, embed_dim, bias=False)

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
            # Affine-less LayerNorm (elementwise_affine=False) has
            # None .bias / .weight -- e.g. the LayerNorm inside AdaLN,
            # whose affine is applied externally as (gamma, beta) from
            # the modulation MLP. Guard against the None case.
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def _compute_layer_emb(
            self,
            x: torch.Tensor,
            attn_base: torch.Tensor | None,
            layers: Sequence[int],
            masks: list[torch.Tensor] | torch.Tensor | None,
            cell_only: bool,
            ignore_spc_tokens: bool = False,
            coords: torch.Tensor | None = None,
            cond: torch.Tensor | None = None,
            ) -> dict[int, torch.Tensor]:
        """
        Helper function to return embeddings for either full context or
        cell only context. ``coords`` (for RoPE) and ``cond`` (for
        AdaLN) are forwarded unchanged to each Block.
        """
        layers: list[int] = sorted({int(l) for l in layers})
        max_layer: int = max(layers)

        # Format masks
        if masks is not None and not isinstance(masks, list):
            masks = [masks]

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
            # Optionally block cross-attention from cell queries to
            # neighborhood keys
            if cell_only:
                attn[
                    :,
                    :,
                    :(self.n_special_tokens+self.seq_len_cell),
                    (self.n_special_tokens+self.seq_len_cell):] = False

        if cell_only:
            x[:, (self.n_special_tokens+self.seq_len_cell):, :] = 0

            #print('model cell attn')
            #torch.set_printoptions(profile="full")
            #print(attn.shape)
            #print(attn[0, :, 1, :])
            #print(attn[0, :, 65, :])

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
        if DEBUG:
            valid_mask = x.ne(0).any(dim=-1)
        for i, blk in enumerate(self.blocks, start=1):
            x = blk(x, masks=attn, coords=coords, cond=cond)
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
        if self.cell_pos_enc == 'none':
            # No spatial encoding at all. Return zeros so the forward's
            # ``x = seg_emb + pos_emb + token_emb + ...`` addition is a
            # no-op for the seg term. Shape matches token_emb's
            # (B, L, embed_dim). Float32 is fine -- autocast handles
            # the mixed-precision add downstream.
            tokens = batch['tokens']
            return torch.zeros(
                tokens.size(0), tokens.size(1), self.embed_dim,
                device=tokens.device, dtype=torch.float32,
            )
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
        elif self.cell_pos_enc in ('polar', 'polar+alibi'):
            # 'polar+alibi' reuses the polar input branch -- the only
            # additional behavior is in _compute_attention_bias which
            # also computes the per-head ALiBi distance bias for the
            # attention layers.
            # Reparameterize (rel_x, rel_y) -> (log(1 + r), theta) so
            # distance (which biology cares about) and angle (which is
            # naturally periodic and well-suited to sincos) are the
            # two encoded channels. log1p handles r=0 and r in
            # [10 um, 1 mm] with even resolution. Special-token
            # positions (rel_x = -inf) are detected and the resulting
            # log_r / theta are set back to -inf so that
            # `get_1d_sincos_pos_embed_from_coord`'s existing -inf
            # handling zeroes them out -- same behavior as the
            # 'coord' branch for special tokens.
            rel_x = batch['rel_x_coords']
            rel_y = batch['rel_y_coords']
            special_mask = torch.isneginf(rel_x) | torch.isneginf(rel_y)
            zero = torch.zeros_like(rel_x)
            rel_x_safe = torch.where(special_mask, zero, rel_x)
            rel_y_safe = torch.where(special_mask, zero, rel_y)
            r = torch.sqrt(rel_x_safe ** 2 + rel_y_safe ** 2)
            theta = torch.atan2(rel_y_safe, rel_x_safe)
            log_r = torch.log1p(r)
            neg_inf = torch.full_like(log_r, float('-inf'))
            log_r = torch.where(special_mask, neg_inf, log_r)
            theta = torch.where(special_mask, neg_inf, theta)
            r_emb = get_1d_sincos_pos_embed_from_coord(
                embed_dim=self.embed_dim // 2,
                omega=self.coord_omega,
                coord=log_r)
            theta_emb = get_1d_sincos_pos_embed_from_coord(
                embed_dim=self.embed_dim // 2,
                omega=self.coord_omega,
                coord=theta)
            seg_emb = torch.cat([r_emb, theta_emb], dim=-1)
        elif self.cell_pos_enc == 'alibi':
            # ALiBi mode: keep the ordinal segment encoding at input
            # (cheap and helpful) and add a per-head distance bias at
            # attention time -- see _compute_attention_bias.
            seg_emb = self.seg_embed(batch['segments'])
        elif self.cell_pos_enc == 'laplacian':
            seg_emb = self._compute_laplacian_pe(batch)
        elif self.cell_pos_enc == 'rope':
            # RoPE is applied inside Attention directly via q/k
            # rotation. The additive segment contribution is zero so
            # that token + pos + value embeddings are preserved at
            # the input.
            tokens = batch['tokens']
            return torch.zeros(
                tokens.size(0), tokens.size(1), self.embed_dim,
                device=tokens.device, dtype=torch.float32,
            )
        else:
            raise ValueError(
                f"Unknown cell_pos_enc: {self.cell_pos_enc}.")

        return seg_emb

    def _build_coords(self, batch: Mapping[str, torch.Tensor]
                      ) -> torch.Tensor | None:
        """Build a ``(B, L, 2)`` per-token coordinate tensor for RoPE.
        Returns ``None`` if the encoder is not in RoPE mode, so the
        forward can pass the result through unconditionally.
        Sentinel ``-inf`` values for special tokens / pad are kept;
        Attention sanitizes them to ``(0, 0)`` (identity rotation)
        before applying RoPE.
        """
        if self.cell_pos_enc != 'rope':
            return None
        return torch.stack(
            [batch['rel_x_coords'], batch['rel_y_coords']], dim=-1)

    def _apply_rda(self,
                   x: torch.Tensor,
                   batch: Mapping[str, torch.Tensor],
                   ) -> torch.Tensor:
        """Add the per-cell depth embedding to the input ``x``.

        No-op (returns ``x`` unchanged) when RDA is disabled. When
        enabled, computes per-cell totals from ``batch['values']``
        using the standard sequence layout and broadcasts a
        zero-initialized depth embedding across each cell's
        gene-token block. The zero-init makes RDA-at-step-0
        identical to the no-RDA baseline.
        """
        if not self.rda_enabled:
            return x
        from .rda import build_depth_embedding
        depth_emb = build_depth_embedding(
            self.rda_module,
            batch,
            n_special_tokens=self.n_special_tokens,
            n_cells=self.n_segments,
            seq_len_cell=self.seq_len_cell,
            seq_len=self.seq_len,
        )
        return x + depth_emb.to(x.dtype)

    def reset_adaln_range_check(self) -> None:
        """Re-arm the one-shot AdaLN label range check on the next
        ``_compute_cond`` call. Useful when reusing the same model
        instance across train -> inference within one process (e.g.
        notebooks): the range check would otherwise be skipped at
        inference because the flag persists from training. Call
        this between phases to get a fresh diagnostic.
        """
        if hasattr(self, '_adaln_label_range_logged'):
            self._adaln_label_range_logged = False

    def _compute_cond(self,
                      batch: Mapping[str, torch.Tensor],
                      ) -> torch.Tensor | None:
        """Return the per-cell conditioning embedding for AdaLN.

        With multi-key AdaLN, the conditioning is the **sum** of
        per-key embedding lookups (one table per metadata key).
        Single-key reduces to the original behavior.

        Shape: ``(B, adaln_batch_embed_dim)`` when AdaLN is on,
        ``_NO_COND`` (None sentinel) otherwise.

        Each key's lookup index is range-checked against its
        embedding table size; mismatches raise actionable errors
        rather than silently clamping (which would collapse
        out-of-range cells to a single row).
        """
        if not self.adaln_enabled:
            return _NO_COND
        from .batch_labels import extract_batch_label

        # Range-check labels only on the first call (one-shot CUDA
        # sync). After that, trust the spec and trade defensive
        # checks for throughput. CUDA's own indexSelect would surface
        # OOB later anyway as an async assertion -- but with a less
        # actionable error message. The first-batch sync gives a
        # clean Python traceback for misconfig diagnosis.
        do_range_check = not getattr(
            self, "_adaln_label_range_logged", False)

        cond = None
        for slot_i, (key, offset, n_cls, emb) in enumerate(zip(
                self.adaln_keys,
                self.adaln_offsets,
                self.adaln_n_classes_per_key,
                self.batch_emb_tables,
        )):
            label = extract_batch_label(
                batch, key=key, offset=offset)
            if do_range_check:
                with torch.no_grad():
                    max_obs = int(label.max().item())
                    min_obs = int(label.min().item())
                if max_obs >= n_cls or min_obs < 0:
                    raise RuntimeError(
                        f"AdaLN slot {slot_i} (key={key!r}) "
                        f"batch_label out of range: observed "
                        f"[{min_obs}, {max_obs}] but "
                        f"n_classes[{slot_i}]={n_cls}. Set the "
                        "matching ``n_classes`` entry to at least "
                        f"{max_obs + 1}, or set ``offsets"
                        f"[{slot_i}]`` to {min_obs} so the lookup "
                        "is 0-indexed.")
            this = emb(label)
            cond = this if cond is None else cond + this

        # One-time multi-slot diagnostic log on the first call.
        # Combined with the range check above so we pay the CUDA
        # sync at most once per process.
        if do_range_check:
            import logging as _logging
            _logging.getLogger(
                "nichejepa.models.gene_transformers"
            ).info(
                "AdaLN multi-key diagnostic on first batch: "
                "keys=%s, n_classes=%s, offsets=%s. cond_dim=%d. "
                "Subsequent steps skip the range check for speed; "
                "any OOB would surface as a CUDA indexSelect "
                "assertion.",
                self.adaln_keys, self.adaln_n_classes_per_key,
                self.adaln_offsets, self.adaln_batch_embed_dim,
            )
            self._adaln_label_range_logged = True

        return cond

    def _compute_laplacian_pe(self, batch: dict) -> torch.Tensor:
        """Per-token Laplacian positional encoding.

        Builds a per-batch-element Gaussian-kernel adjacency over the
        ``n_segments`` cells in the neighborhood, computes the
        normalized Laplacian, takes the bottom ``laplacian_k``
        non-trivial eigenvectors, and projects them to ``embed_dim``
        via a learnable linear layer. The resulting per-cell PE is
        then broadcast to every token of that cell via the
        ``segments`` tensor. Special-token positions get zero PE.

        Assumes the standard sequence layout used by the cell-graph /
        cell-neighborhood tokenizers: ``n_special_tokens`` special
        tokens followed by ``n_segments`` cells of ``seq_len_cell``
        tokens each, with segment IDs in ``1 .. n_segments`` (zero
        reserved for special / padding).
        """
        rel_x = batch['rel_x_coords']   # (B, L)
        rel_y = batch['rel_y_coords']
        segments = batch['segments']    # (B, L) long
        B, L = rel_x.shape
        device = rel_x.device
        n_cells = self.n_segments

        # 1. Pull one (rel_x, rel_y) per cell from the first token of
        #    each cell's contiguous block.
        first_token_idx = (
            self.n_special_tokens
            + torch.arange(n_cells, device=device) * self.seq_len_cell
        )  # (n_cells,)
        if first_token_idx.max().item() >= L:
            raise RuntimeError(
                "Laplacian PE: computed first-token indices exceed "
                f"sequence length (max={first_token_idx.max().item()}, "
                f"L={L}). Check that n_special_tokens + n_segments * "
                "seq_len_cell == seq_len.")
        cell_x = rel_x[:, first_token_idx]  # (B, n_cells)
        cell_y = rel_y[:, first_token_idx]

        # Cells whose representative token is a "missing" segment
        # (rel = -inf, e.g. dropped via masking) are excluded from
        # the graph by zeroing their adjacency rows/cols.
        cell_missing = (
            torch.isneginf(cell_x) | torch.isneginf(cell_y)
        )  # (B, n_cells)
        cell_x = torch.where(cell_missing, torch.zeros_like(cell_x), cell_x)
        cell_y = torch.where(cell_missing, torch.zeros_like(cell_y), cell_y)

        # 2. Gaussian-kernel adjacency. Zero diagonal (no self-edge);
        #    zero rows/cols for missing cells.
        positions = torch.stack([cell_x, cell_y], dim=-1)   # (B, n_cells, 2)
        diff = positions.unsqueeze(2) - positions.unsqueeze(1)  # (B, c, c, 2)
        dist_sq = (diff ** 2).sum(dim=-1)                   # (B, n_cells, n_cells)
        sigma = max(self.laplacian_sigma, 1e-6)
        adj = torch.exp(-dist_sq / (2.0 * sigma * sigma))   # (B, n_cells, n_cells)
        eye = torch.eye(n_cells, device=device).unsqueeze(0)
        adj = adj * (1.0 - eye)
        valid = (~cell_missing).to(adj.dtype)               # (B, n_cells)
        adj = adj * valid.unsqueeze(1) * valid.unsqueeze(2)

        # 3. Normalized Laplacian L = I - D^{-1/2} A D^{-1/2}.
        deg = adj.sum(dim=-1).clamp(min=1e-8)               # (B, n_cells)
        deg_inv_sqrt = 1.0 / torch.sqrt(deg)
        # Zero out the normalization for cells with no edges so they
        # contribute identity rows (eigvecs there are arbitrary; we
        # zero them out via cell_missing downstream).
        deg_inv_sqrt = torch.where(
            cell_missing, torch.zeros_like(deg_inv_sqrt), deg_inv_sqrt)
        norm_adj = adj * deg_inv_sqrt.unsqueeze(-1) * deg_inv_sqrt.unsqueeze(-2)
        lap = eye - norm_adj                                # (B, n_cells, n_cells)

        # 4. Eigendecomposition. No grad through positions -- they're
        #    not learnable parameters, only the projection is.
        with torch.no_grad():
            # eigh returns ascending eigenvalues. Skip the smallest
            # (trivial zero/near-zero eigenvalue) and take the next k.
            _, eigvecs = torch.linalg.eigh(lap)              # (B, c, c)
            pe = eigvecs[:, :, 1: self.laplacian_k + 1].contiguous()
            # Resolve sign ambiguity by convention: first non-missing
            # entry of each eigenvector is forced positive. (Eigvecs
            # of a real symmetric matrix are defined up to sign.)
            ref = pe[:, 0:1, :]                              # (B, 1, k)
            signs = torch.where(
                ref >= 0,
                torch.ones_like(ref),
                -torch.ones_like(ref),
            )
            pe = pe * signs
            # Zero out missing cells' PE rows.
            pe = pe * (~cell_missing).unsqueeze(-1).to(pe.dtype)

        # 5. Project k-dim eigenvectors into embed_dim, then broadcast
        #    each cell's PE to every token of that cell.
        cell_pe = self.laplacian_proj(pe)                    # (B, n_cells, D)

        # Per-position cell index. Mirrors the cell-position
        # *extraction* above (which also uses the sequence layout):
        # positions in ``[n_special, n_special + n_cells * seq_len_cell)``
        # belong to cell ``(pos - n_special) // seq_len_cell``.
        # Doing this layout-based rather than segments-based is
        # important because ``segments`` does NOT always range over
        # ``[0, n_cells]`` -- with ``nz_spc=True`` the dataset puts
        # the special tokens at segment IDs like ``2, 3, ...``, which
        # would otherwise overflow ``n_cells`` and trigger an
        # out-of-bounds in ``torch.gather``.
        pos_idx = torch.arange(L, device=device)
        cell_idx_per_pos = (
            (pos_idx - self.n_special_tokens)
            .div(self.seq_len_cell, rounding_mode='floor')
            .clamp(min=0, max=n_cells - 1)
        )                                                    # (L,)
        cell_idx_per_pos = cell_idx_per_pos.unsqueeze(0).expand(B, L)
        cell_idx_exp = cell_idx_per_pos.unsqueeze(-1).expand(
            B, L, cell_pe.size(-1))
        per_token_pe = torch.gather(cell_pe, dim=1, index=cell_idx_exp)

        # Zero PE for (a) positions in the special-token prefix, and
        # (b) any in-cell pad tokens (token == 0 -> segment == 0 in
        # nz_spc=False, but with nz_spc=True we can't rely on the
        # segments value; use the original tokens tensor if it's in
        # the batch). The position-based special-token mask handles
        # the prefix uniformly.
        in_special_region = (pos_idx < self.n_special_tokens).view(1, L, 1)
        zero_mask = in_special_region.expand(B, L, 1)
        if 'tokens' in batch:
            pad_mask = (batch['tokens'] == 0).unsqueeze(-1)
            zero_mask = zero_mask | pad_mask
        per_token_pe = torch.where(
            zero_mask, torch.zeros_like(per_token_pe), per_token_pe)
        return per_token_pe

    @torch.no_grad()
    def compute_laplacian_diagnostic(self, batch: dict) -> dict:
        """Return a dict of summary statistics on the spatial graph used
        by Laplacian PE. Useful for tuning ``laplacian_sigma`` -- a
        well-scaled adjacency has ``adj_offdiag_mean`` roughly in
        ``[0.2, 0.7]``. Lower means the kernel is collapsing (sigma too
        small relative to distances); higher means everyone is
        effectively connected to everyone (sigma too large).

        Returned keys (all scalars):
            laplacian/sigma
            laplacian/k
            laplacian/adj_offdiag_mean
            laplacian/adj_offdiag_min
            laplacian/adj_offdiag_max
            laplacian/dist_offdiag_mean
            laplacian/dist_offdiag_median
            laplacian/dist_offdiag_max
            laplacian/eigval_min_nontrivial
            laplacian/eigval_max
            laplacian/spectral_gap   (= eigval[k+1] - eigval[1])
        Returns ``{}`` if the encoder is not in laplacian mode.
        """
        if self.cell_pos_enc != 'laplacian':
            return {}
        rel_x = batch['rel_x_coords']
        rel_y = batch['rel_y_coords']
        device = rel_x.device
        n_cells = self.n_segments

        first_token_idx = (
            self.n_special_tokens
            + torch.arange(n_cells, device=device) * self.seq_len_cell
        )
        cell_x = rel_x[:, first_token_idx]
        cell_y = rel_y[:, first_token_idx]
        cell_missing = (
            torch.isneginf(cell_x) | torch.isneginf(cell_y)
        )
        cell_x = torch.where(cell_missing, torch.zeros_like(cell_x), cell_x)
        cell_y = torch.where(cell_missing, torch.zeros_like(cell_y), cell_y)

        positions = torch.stack([cell_x, cell_y], dim=-1)
        diff = positions.unsqueeze(2) - positions.unsqueeze(1)
        dist = (diff ** 2).sum(dim=-1).clamp(min=0.0).sqrt()  # (B, c, c)
        sigma = max(self.laplacian_sigma, 1e-6)
        adj = torch.exp(-(dist ** 2) / (2.0 * sigma * sigma))
        eye = torch.eye(n_cells, device=device).unsqueeze(0).bool()
        offdiag_mask = ~eye  # (1, c, c) -> broadcasts over batch

        adj_off = adj[offdiag_mask.expand_as(adj)]
        dist_off = dist[offdiag_mask.expand_as(dist)]

        # Spectrum on the actual normalized Laplacian we'd use.
        valid = (~cell_missing).to(adj.dtype)
        adj_masked = adj * (1.0 - eye.to(adj.dtype))
        adj_masked = adj_masked * valid.unsqueeze(1) * valid.unsqueeze(2)
        deg = adj_masked.sum(dim=-1).clamp(min=1e-8)
        deg_inv_sqrt = torch.where(
            cell_missing, torch.zeros_like(deg), 1.0 / torch.sqrt(deg))
        norm_adj = (adj_masked
                    * deg_inv_sqrt.unsqueeze(-1)
                    * deg_inv_sqrt.unsqueeze(-2))
        lap = (eye.to(adj.dtype) - norm_adj)
        eigvals, _ = torch.linalg.eigh(lap)  # (B, c) ascending
        # Skip the smallest (~0) eigenvalue; use means across the batch.
        nontrivial = eigvals[:, 1:]
        eig_min = nontrivial[:, 0].mean()
        eig_max = nontrivial[:, -1].mean()
        # Spectral gap between the k-th and (k+1)-th eigenvalue tells
        # you whether keeping more eigenvectors would still carry
        # signal. Index laplacian_k-1 of nontrivial corresponds to
        # the last kept eigenvalue; +1 is the first dropped one.
        k = self.laplacian_k
        if nontrivial.size(1) > k:
            gap = (nontrivial[:, k] - nontrivial[:, k - 1]).mean()
        else:
            gap = nontrivial[:, -1].mean() - nontrivial[:, 0].mean()

        return {
            "laplacian/sigma": float(self.laplacian_sigma),
            "laplacian/k": int(self.laplacian_k),
            "laplacian/adj_offdiag_mean": float(adj_off.mean().item()),
            "laplacian/adj_offdiag_min": float(adj_off.min().item()),
            "laplacian/adj_offdiag_max": float(adj_off.max().item()),
            "laplacian/dist_offdiag_mean": float(dist_off.mean().item()),
            "laplacian/dist_offdiag_median": float(dist_off.median().item()),
            "laplacian/dist_offdiag_max": float(dist_off.max().item()),
            "laplacian/eigval_min_nontrivial": float(eig_min.item()),
            "laplacian/eigval_max": float(eig_max.item()),
            "laplacian/spectral_gap_at_k": float(gap.item()),
        }

    @staticmethod
    def _get_alibi_slopes(num_heads: int) -> torch.Tensor:
        """ALiBi per-head slopes (Press et al. 2022). Geometric
        progression starting at 2^(-8/H), so different heads attend
        at different spatial scales: short heads see only nearby
        cells, long heads see across the whole neighborhood.
        Handles non-power-of-2 ``num_heads`` by interpolating between
        the closest powers of 2 (the standard ALiBi recipe).
        """
        def _slopes_for_power_of_2(n: int):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            return [start * (start ** i) for i in range(n)]

        if math.log2(num_heads).is_integer():
            slopes = _slopes_for_power_of_2(num_heads)
        else:
            closest = 2 ** math.floor(math.log2(num_heads))
            slopes = (_slopes_for_power_of_2(closest)
                      + _slopes_for_power_of_2(2 * closest)[0::2]
                      [: num_heads - closest])
        return torch.tensor(slopes, dtype=torch.float32)

    def _compute_alibi_bias(self, batch: dict) -> torch.Tensor:
        """Return per-head additive attention bias of shape
        ``(B, num_heads, L, L)`` based on pairwise spatial distance
        between cells. Bias is `-slope[h] * ||cell_i - cell_j||`, so
        nearby cells stay near 0 and far cells receive a strong
        negative bias that softmax suppresses. Special tokens
        (rel_x = -inf) get zero bias to/from everywhere -- they are
        spatially "anywhere" and should be free to attend globally.
        """
        rel_x = batch['rel_x_coords']  # (B, L)
        rel_y = batch['rel_y_coords']  # (B, L)
        special_mask = torch.isneginf(rel_x) | torch.isneginf(rel_y)  # (B, L)
        zero = torch.zeros_like(rel_x)
        rel_x = torch.where(special_mask, zero, rel_x)
        rel_y = torch.where(special_mask, zero, rel_y)

        dx = rel_x.unsqueeze(2) - rel_x.unsqueeze(1)  # (B, L, L)
        dy = rel_y.unsqueeze(2) - rel_y.unsqueeze(1)
        dist = torch.sqrt(dx * dx + dy * dy + 1e-12)  # (B, L, L)

        # Zero out distance whenever either side is a special token.
        not_special = (~special_mask).to(dist.dtype)
        dist = dist * not_special.unsqueeze(2) * not_special.unsqueeze(1)

        # (B, H, L, L)
        slopes = self.alibi_slopes.to(dist.device, dist.dtype)
        bias = -slopes.view(1, -1, 1, 1) * dist.unsqueeze(1)
        return bias

    @staticmethod
    def _slice_alibi_by_indices(
            alibi: torch.Tensor, keep_indices: torch.Tensor
            ) -> torch.Tensor:
        """Slice a full-sequence alibi bias ``(B, H, L, L)`` down to
        the ``keep_indices`` positions, returning
        ``(B, H, M, M)`` where ``M = keep_indices.size(1)``. Used when
        the encoder applies a JEPA mask to its input -- the bias has
        to be sliced consistently so the per-token attention scores
        match the per-token positions in the reduced sequence.
        """
        B, H, L, _ = alibi.shape
        M = keep_indices.size(1)
        row_idx = keep_indices.unsqueeze(1).unsqueeze(-1).expand(B, H, M, L)
        alibi = torch.gather(alibi, 2, row_idx)
        col_idx = keep_indices.unsqueeze(1).unsqueeze(2).expand(B, H, M, M)
        alibi = torch.gather(alibi, 3, col_idx)
        return alibi

    def _compute_attention_bias(
            self,
            batch: dict,
            masks_attention: Optional[torch.Tensor] = None,
            keep_indices: Optional[torch.Tensor] = None,
            ) -> Optional[torch.Tensor]:
        """Combine the optional existing attention mask with the ALiBi
        bias when ``cell_pos_enc == 'alibi'``. For other modes this is
        a pass-through (returns ``masks_attention`` unchanged), so
        existing configs are unaffected.

        ``keep_indices`` (the JEPA mask, shape ``(B, M)``) is used to
        slice the full-sequence ALiBi bias down to the masked
        sub-sequence. This is needed by the context encoder, which
        applies ``apply_masks`` before running blocks. The target
        encoder passes ``keep_indices=None``.

        Handles three input ``masks_attention`` formats:
        - ``None``  : just return the (possibly sliced) ALiBi bias.
        - ``bool``  : -inf where False, +alibi everywhere else.
        - ``float`` : add directly.
        """
        if self.cell_pos_enc not in ('alibi', 'polar+alibi'):
            return masks_attention

        alibi = self._compute_alibi_bias(batch)  # (B, H, L, L)
        if keep_indices is not None:
            alibi = self._slice_alibi_by_indices(alibi, keep_indices)

        if masks_attention is None:
            return alibi
        if masks_attention.dtype == torch.bool:
            ma = masks_attention
            if keep_indices is not None:
                # masks_attention is (B, 1, 1, L); slice the key axis.
                idx = keep_indices.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, M)
                ma = torch.gather(ma, 3, idx)
            mask_f = alibi.clone()
            mask_f = mask_f.masked_fill(~ma, float('-inf'))
            return mask_f
        return masks_attention + alibi

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
        Cell position encoding. One of:
        - ``segment``: cells ranked by NN distance, fixed sincos by rank.
        - ``coord``: relative (x, y) sincos'd independently.
        - ``polar``: (log(1+r), theta) from (rel_x, rel_y), each sincos'd.
        - ``alibi``: segment input encoding + per-head distance-decaying
          attention bias.
        Older docstring text kept for reference: `segment` if cells are ranked
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
                 cell_pos_enc: Literal[
                     'none', 'segment', 'coord', 'polar', 'alibi',
                     'polar+alibi', 'laplacian', 'rope'],
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
                 rope_freq_scale: float = math.pi,
                 rope_rotation_augment: bool = True,
                 adaln_kwargs: dict | None = None,
                 special_token_moe_kwargs: dict | None = None,
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

        # Special-token-routed bias head (lightweight MoE). Per-slot
        # additive biases on the predictor output, routed by one or
        # more values columns. Zero-init -> step-0 no-op, so enabling
        # at train start is bitwise backward-compatible. Multiple
        # routing slots (e.g. batch + assay) sum their bias
        # contributions -- this is equivalent to a per-(batch, assay)
        # bias but with O(n_batches + n_assays) params instead of
        # O(n_batches * n_assays).
        self.special_token_moe_enabled = (
            bool(special_token_moe_kwargs)
            and special_token_moe_kwargs.get('enabled', False)
        )
        if self.special_token_moe_enabled:
            from .special_token_moe import (
                SpecialTokenMoE, _as_list)
            # Accept scalar or list for each field; both metadata-key
            # routing (``routing_keys``) and sequence-index routing
            # (``routing_indices``) are supported. Key-mode is
            # preferred because it works regardless of
            # ``special_tokens``; index-mode is kept for backward
            # compat.
            routing_keys = _as_list(
                special_token_moe_kwargs.get('routing_keys'))
            routing_indices = _as_list(
                special_token_moe_kwargs.get('routing_indices'))
            if not routing_indices:
                # Backward-compat: also accept legacy 'routing_index'.
                routing_indices = _as_list(
                    special_token_moe_kwargs.get('routing_index'))
            if not routing_keys and not routing_indices:
                raise ValueError(
                    "special_token_moe.enabled=True but neither "
                    "routing_keys (metadata-field names, recommended) "
                    "nor routing_indices/routing_index (values "
                    "columns) is set.")
            # Determine slot count from whichever list is set; the
            # other defaults to all-None of matching length.
            if routing_keys:
                n_slots = len(routing_keys)
                if routing_indices and len(routing_indices) != n_slots:
                    raise ValueError(
                        "special_token_moe: routing_keys and "
                        "routing_indices must have the same length "
                        "if both are provided.")
                if not routing_indices:
                    routing_indices = [None] * n_slots
            else:
                n_slots = len(routing_indices)
                routing_keys = [None] * n_slots

            n_experts = _as_list(
                special_token_moe_kwargs.get('n_experts'))
            if not n_experts:
                raise ValueError(
                    "special_token_moe.enabled=True but n_experts "
                    "is not set.")
            routing_offsets = _as_list(
                special_token_moe_kwargs.get('routing_offsets'),
                fallback_len=n_slots)
            if not routing_offsets:
                routing_offsets = _as_list(
                    special_token_moe_kwargs.get('routing_offset'),
                    fallback_len=n_slots)

            if not (n_slots == len(n_experts) == len(routing_offsets)):
                raise ValueError(
                    "special_token_moe: routing_keys/"
                    "routing_indices, n_experts, and routing_offsets "
                    "must all have the same length. Got "
                    f"n_slots={n_slots}, n_experts={n_experts}, "
                    f"routing_offsets={routing_offsets}.")

            self.special_token_moe_routing_keys = [
                None if x is None else str(x) for x in routing_keys]
            self.special_token_moe_routing_indices = [
                None if x is None else int(x) for x in routing_indices]
            self.special_token_moe_routing_offsets = [
                int(x) for x in routing_offsets]
            self.special_token_moe = SpecialTokenMoE(
                n_experts_per_slot=[int(x) for x in n_experts],
                embed_dim=predictor_embed_dim,
            )
        else:
            self.special_token_moe_routing_keys = []
            self.special_token_moe_routing_indices = []
            self.special_token_moe_routing_offsets = []
            self.special_token_moe = None

        # Initialize segment embeddings. Used by 'segment' mode (full
        # positional signal) and by 'alibi' / 'laplacian' modes at the
        # predictor side. The 'alibi' attention bias and 'laplacian'
        # PE are encoder-only; the predictor side just uses segment
        # input encoding (cheap, carries ordinal NN info, sufficient
        # for slotting target tokens in). 'polar+alibi' uses polar on
        # the predictor side too.
        if self.cell_pos_enc in ('segment', 'alibi', 'laplacian'):
            # IMPORTANT: with nz_spc=True, the dataset assigns segment
            # IDs [1..n_special_tokens] to the special-token positions
            # and [n_special_tokens+1..n_special_tokens+n_segments] to
            # the gene-token cells, so the max segment ID observed is
            # n_special_tokens + n_segments. The table size must
            # therefore add n_special_tokens (not 1) when nz_spc is
            # on. Previously the predictor used `+ 1`, which was an
            # off-by-(n_special_tokens-1) sized table -- fine for the
            # default n_special_tokens=1 case, but caused
            # indexSelectLargeIndex CUDA assertions as soon as more
            # than one special token was used (e.g.
            # special_tokens=['batch', 'assay']). Mirror the encoder's
            # sizing exactly.
            self.seg_embed = nn.Embedding(
                1 + n_segments + (105 if api_version == 'v1' else 0) + (self.n_special_tokens if self.nz_spc else 0), # include <pad>
                predictor_embed_dim,
                padding_idx=0)

            # Prevent gradient updates and initialize with sincos embedding,
            # including special segments
            self.seg_embed.weight.requires_grad = False
            seg_embed = get_1d_sincos_pos_embed(
                embed_dim=predictor_embed_dim,
                n_zero_pos=0,
                n_sincos_pos=n_segments + (105 if api_version == 'v1' else 0) + (self.n_special_tokens if self.nz_spc else 0))
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

        # 2D rotary position embedding, shared across predictor blocks.
        if self.cell_pos_enc == 'rope':
            pred_head_dim = predictor_embed_dim // num_heads
            self.rope = RoPE2D(
                head_dim=pred_head_dim,
                freq_scale=rope_freq_scale,
                rotation_augment=rope_rotation_augment,
            )
        else:
            self.rope = None

        # AdaLN conditioning. Mirrors the encoder's setup: a per-batch
        # embedding lookup table whose output is fed to every Block's
        # two AdaLN modules. Predictor uses its own embedding table
        # (predictor_embed_dim != encoder_embed_dim in general), so
        # there's no parameter sharing between encoder.batch_emb_tables
        # and predictor.batch_emb_tables. They're updated independently
        # by gradient descent.
        self.adaln_enabled = bool(adaln_kwargs) and adaln_kwargs.get(
            'enabled', False)
        if self.adaln_enabled:
            from .batch_labels import resolve_label_spec
            self.adaln_batch_embed_dim = int(adaln_kwargs.get(
                'batch_embed_dim', 64))
            # Resolve multi-key spec. Falls back to single-key
            # (legacy ``values[:, 0]`` path) if no ``keys`` set.
            spec = resolve_label_spec(adaln_kwargs)
            if not spec['keys']:
                # Legacy single-slot using values[:, 0]. CAVEAT: at
                # inference with ``pad_special_tokens=True`` (the
                # default for embed_dataset / infer.py), the spv_*
                # positions of the values tensor are zeroed, so
                # AdaLN would silently condition on batch_label=0
                # for every cell -- different from training. Migrate
                # to the metadata key path
                # (``adaln.keys=['batch_value']``) for inference-safe
                # behavior. Rank-0 gate so DDP runs don't print
                # n_ranks * n_modules duplicate warnings.
                import os as _adaln_os
                if int(_adaln_os.environ.get('RANK', '0')) == 0:
                    import logging as _adaln_logging
                    _adaln_logging.getLogger(
                        "nichejepa.models.gene_transformers"
                    ).warning(
                        "[AdaLN] using LEGACY single-key path "
                        "(reads values[:, 0]). At inference with "
                        "pad_special_tokens=True this silently "
                        "conditions on batch_label=0. Recommend "
                        "migrating to "
                        "``adaln.keys=['batch_value']`` + "
                        "matching n_classes / offsets.")
                self.adaln_keys = [None]
                self.adaln_offsets = [0]
                self.adaln_n_classes_per_key = [
                    int(adaln_kwargs['n_batches'])]
            else:
                if not spec['n_classes'] or any(
                        n <= 0 for n in spec['n_classes']):
                    raise ValueError(
                        "AdaLN multi-key spec: ``n_classes`` must be "
                        "set per key (one positive int per "
                        f"key). Got n_classes={spec['n_classes']} "
                        f"for keys={spec['keys']}.")
                self.adaln_keys = spec['keys']
                self.adaln_offsets = spec['offsets']
                self.adaln_n_classes_per_key = spec['n_classes']
            # One embedding table per key; per-key lookups are summed
            # to form the AdaLN cond vector. This is the same
            # additive-bias-by-slot structure as special_token_moe
            # but used as conditioning input instead of output bias.
            self.batch_emb_tables = nn.ModuleList([
                nn.Embedding(n, self.adaln_batch_embed_dim)
                for n in self.adaln_n_classes_per_key
            ])
            _block_cond_dim = self.adaln_batch_embed_dim
        else:
            self.adaln_keys = []
            self.adaln_offsets = []
            self.adaln_n_classes_per_key = []
            self.adaln_batch_embed_dim = 0
            self.batch_emb_tables = None
            _block_cond_dim = None

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
                  use_layer_norm=use_layer_norm,
                  rope=self.rope,
                  cond_dim=_block_cond_dim)
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
        # `self.apply(_init_weights)` above visits every nn.Linear --
        # including each AdaLN's modulation hypernetwork -- and re-
        # initializes it via trunc_normal_(self.init_std). That
        # overwrites the zero-init from AdaLN.__init__ and breaks the
        # AdaLN-at-step-0 == LayerNorm invariant. Restore it here.
        # No-op when AdaLN is disabled (no AdaLN modules to find).
        zero_init_adaln_modulations(self)

        # Compute omega for coord-based positional sincos. Used by
        # 'coord' (sincos of rel_x, rel_y), 'polar' / 'polar+alibi'
        # (sincos of log(1+r), theta).
        if self.cell_pos_enc in ('coord', 'polar', 'polar+alibi'):
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
            # Affine-less LayerNorm (elementwise_affine=False) has
            # None .bias / .weight -- e.g. the LayerNorm inside AdaLN,
            # whose affine is applied externally as (gamma, beta) from
            # the modulation MLP. Guard against the None case.
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def _get_seg_emb(
            self,
            batch: dict):
        """
        Helper function to get segment embeddings.
        """
        if self.cell_pos_enc == 'none':
            # No spatial encoding (predictor side). Mirrors the
            # encoder's 'none' branch: return zeros of the right
            # shape so the predictor's additive sum is unaffected.
            tokens = batch['tokens']
            return torch.zeros(
                tokens.size(0), tokens.size(1), self.predictor_embed_dim,
                device=tokens.device, dtype=torch.float32,
            )
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
        elif self.cell_pos_enc in ('polar', 'polar+alibi'):
            # Same polar reparameterization as the encoder. For
            # 'polar+alibi' the alibi attention bias is encoder-only;
            # the predictor sees just the polar input encoding.
            rel_x = batch['rel_x_coords']
            rel_y = batch['rel_y_coords']
            special_mask = torch.isneginf(rel_x) | torch.isneginf(rel_y)
            zero = torch.zeros_like(rel_x)
            rel_x_safe = torch.where(special_mask, zero, rel_x)
            rel_y_safe = torch.where(special_mask, zero, rel_y)
            r = torch.sqrt(rel_x_safe ** 2 + rel_y_safe ** 2)
            theta = torch.atan2(rel_y_safe, rel_x_safe)
            log_r = torch.log1p(r)
            neg_inf = torch.full_like(log_r, float('-inf'))
            log_r = torch.where(special_mask, neg_inf, log_r)
            theta = torch.where(special_mask, neg_inf, theta)
            r_emb = get_1d_sincos_pos_embed_from_coord(
                embed_dim=self.predictor_embed_dim // 2,
                omega=self.coord_omega,
                coord=log_r)
            theta_emb = get_1d_sincos_pos_embed_from_coord(
                embed_dim=self.predictor_embed_dim // 2,
                omega=self.coord_omega,
                coord=theta)
            seg_emb = torch.cat([r_emb, theta_emb], dim=-1)
        elif self.cell_pos_enc in ('alibi', 'laplacian'):
            # On the predictor side, both 'alibi' (encoder-only
            # attention bias) and 'laplacian' (encoder-only PE
            # because computing it on masked subsets is non-trivial)
            # fall back to segment-style input encoding.
            seg_emb = self.seg_embed(batch['segments'])
        elif self.cell_pos_enc == 'rope':
            # Predictor mirrors the encoder: RoPE is applied inside
            # Attention via q/k rotation; the additive contribution
            # at the input is zero.
            tokens = batch['tokens']
            return torch.zeros(
                tokens.size(0), tokens.size(1), self.predictor_embed_dim,
                device=tokens.device, dtype=torch.float32,
            )
        else:
            raise ValueError(
                f"Unknown cell_pos_enc: {self.cell_pos_enc}.")

        return seg_emb

    def _build_coords(self, batch: Mapping[str, torch.Tensor]
                      ) -> torch.Tensor | None:
        """``(B, L, 2)`` per-token coordinate tensor for RoPE in the
        predictor. Returns ``None`` unless ``cell_pos_enc == 'rope'``.
        """
        if self.cell_pos_enc != 'rope':
            return None
        return torch.stack(
            [batch['rel_x_coords'], batch['rel_y_coords']], dim=-1)

    def _apply_special_token_moe(self,
                                 z: torch.Tensor,
                                 batch: Mapping[str, torch.Tensor],
                                 n_pred_masks: int,
                                 ) -> torch.Tensor:
        """Add the summed per-slot special-token biases to the
        predictor output ``z``.

        ``z`` has shape ``(B * n_pred_masks, L_target, D)``. The
        total bias is ``(B, D)`` per-cell (sum of per-slot lookups);
        we repeat across the ``n_pred_masks`` outer-batch dimension
        and broadcast across the target-length dimension before
        adding. No-op when SpecialTokenMoE is disabled.
        """
        if not self.special_token_moe_enabled:
            return z
        from .special_token_moe import extract_special_token_indices
        indices_per_slot = extract_special_token_indices(
            batch,
            routing_indices=self.special_token_moe_routing_indices,
            routing_offsets=self.special_token_moe_routing_offsets,
            routing_keys=self.special_token_moe_routing_keys,
        )                                                    # list of (B,)
        bias = self.special_token_moe(indices_per_slot)     # (B, D)
        bias = bias.repeat(n_pred_masks, 1).unsqueeze(1)    # (B*M, 1, D)
        return z + bias.to(z.dtype)

    def _compute_cond(self,
                      batch: Mapping[str, torch.Tensor],
                      ) -> torch.Tensor | None:
        """Return the per-cell conditioning embedding for AdaLN. The
        predictor's tables are independent of the encoder's; both
        are sized identically (mirror config). Multi-key: sum of
        per-key embedding lookups. See encoder ``_compute_cond``
        docstring for the multi-key semantics.
        """
        if not self.adaln_enabled:
            return None
        from .batch_labels import extract_batch_label

        # One-shot range check on the first call (one CUDA sync per
        # slot, once per process). After that, trust the spec for
        # throughput; OOB would surface as an async CUDA assertion.
        do_range_check = not getattr(
            self, "_adaln_label_range_logged", False)

        cond = None
        for slot_i, (key, offset, n_cls, emb) in enumerate(zip(
                self.adaln_keys,
                self.adaln_offsets,
                self.adaln_n_classes_per_key,
                self.batch_emb_tables,
        )):
            label = extract_batch_label(
                batch, key=key, offset=offset)
            if do_range_check:
                with torch.no_grad():
                    max_obs = int(label.max().item())
                    min_obs = int(label.min().item())
                if max_obs >= n_cls or min_obs < 0:
                    raise RuntimeError(
                        f"AdaLN (predictor) slot {slot_i} "
                        f"(key={key!r}) batch_label out of range: "
                        f"observed [{min_obs}, {max_obs}] but "
                        f"n_classes[{slot_i}]={n_cls}.")
            this = emb(label)
            cond = this if cond is None else cond + this

        if do_range_check:
            self._adaln_label_range_logged = True

        return cond

    def reset_adaln_range_check(self) -> None:
        """Re-arm the one-shot AdaLN label range check on the next
        ``_compute_cond`` call. Mirrors the encoder method so a
        caller iterating both can do ``encoder.reset...();
        predictor.reset...()`` symmetrically.
        """
        if hasattr(self, '_adaln_label_range_logged'):
            self._adaln_label_range_logged = False

    def _compose_predictor_coords(
            self,
            coords: torch.Tensor,
            masks_enc: list[torch.Tensor],
            masks_pred: list[torch.Tensor],
            ) -> torch.Tensor:
        """Mirror the context / target concatenation applied to ``z``
        for the coordinate tensor, so RoPE sees the correct per-token
        ``(x, y)`` of every token the predictor blocks see.

        Layout (matches the existing predictor forward concat):
          ``new_spc=False`` : ``[ coords_pred ; coords_ctx_repeated ]``
          ``new_spc=True``  : ``[ coords_ctx[:, :n_spc] ;
                                   coords_pred[:, n_spc:] ;
                                   coords_ctx[:, n_spc:] ]``
        """
        coords_ctx = apply_masks(coords, masks_enc)
        coords_pred = apply_masks(coords, masks_pred)
        coords_ctx = coords_ctx.repeat(len(masks_pred), 1, 1)
        if self.new_spc:
            return torch.cat([
                coords_ctx[:, :self.n_special_tokens, :],
                coords_pred[:, self.n_special_tokens:, :],
                coords_ctx[:, self.n_special_tokens:, :],
            ], dim=1)
        return torch.cat([coords_pred, coords_ctx], dim=1)

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

            # Compute the per-batch attention bias (ALiBi if enabled,
            # otherwise just passes ``masks_attention`` through). When
            # ``masks`` is provided we slice the bias to match the
            # reduced sequence length post-apply_masks.
            _keep_idx = (
                masks[0] if (masks is not None and len(masks) == 1) else None
            )
            attn_bias = self._compute_attention_bias(
                batch,
                masks_attention=masks_attention,
                keep_indices=_keep_idx,
            )

            # Build (B, L, 2) coords for RoPE; None for other modes.
            coords = self._build_coords(batch)

            # Per-cell conditioning for AdaLN (or None when off).
            cond = self._compute_cond(batch)

            # Mask token embeddings if masks are provided
            if masks is not None:
                x = apply_masks(x, masks)
                if coords is not None:
                    coords = apply_masks(coords, masks)

            # Run forward prop
            for i, blk in enumerate(self.blocks):
                x = blk(x, masks=attn_bias, coords=coords, cond=cond)
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
        #if ignore_spc_tokens:
        #    if self.n_special_tokens:
        #        x[:, :self.n_special_tokens, :] = 0

        # Inference path: build coords (for RoPE) and cond (for AdaLN)
        # so models trained with these features can also be embedded.
        coords = self._build_coords(batch)
        cond = self._compute_cond(batch)

        full_ctx: dict[int, torch.Tensor] = self._compute_layer_emb(
            x,
            masks_attention,
            layers,
            masks,
            cell_only=False,
            ignore_spc_tokens=ignore_spc_tokens,
            coords=coords,
            cond=cond)
        cell_only_ctx: dict[int, torch.Tensor] = self._compute_layer_emb(
            x,
            masks_attention,
            layers,
            masks,
            cell_only=True,
            ignore_spc_tokens=ignore_spc_tokens,
            coords=coords,
            cond=cond) if need_cell_only_context else None

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
        # RDA depth conditioning (no-op when disabled).
        x = self._apply_rda(x, batch)
        # B, N, D = x.shape # B: BATCH_SIZE, N: SEQ_LEN, D: EMBED_DIM

        # Compute attention bias (passes through unless ALiBi is on).
        _keep_idx = (
            masks[0] if (masks is not None and len(masks) == 1) else None
        )
        attn_bias = self._compute_attention_bias(
            batch,
            masks_attention=masks_attention,
            keep_indices=_keep_idx,
        )

        coords = self._build_coords(batch)

        # Per-cell conditioning for AdaLN (or None when off).
        cond = self._compute_cond(batch)

        # Mask token embeddings if masks are provided
        if masks is not None:
            x = apply_masks(x, masks)
            if coords is not None:
                coords = apply_masks(coords, masks)

        # Run forward prop
        for i, blk in enumerate(self.blocks):
            x = blk(x, masks=attn_bias, coords=coords, cond=cond)
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
        # RDA depth conditioning (no-op when disabled).
        x = self._apply_rda(x, batch)

        # Remove special token contents
        #if ignore_spc_tokens:
        #    if self.n_special_tokens:
        #        x[:, :self.n_special_tokens, :] = 0

        # Inference path: build coords (for RoPE) and cond (for AdaLN)
        # so models trained with these features can also be embedded.
        coords = self._build_coords(batch)
        cond = self._compute_cond(batch)

        full_ctx: dict[int, torch.Tensor] = self._compute_layer_emb(
            x,
            masks_attention,
            layers,
            masks,
            cell_only=False,
            ignore_spc_tokens=ignore_spc_tokens,
            coords=coords,
            cond=cond)
        cell_only_ctx: dict[int, torch.Tensor] = self._compute_layer_emb(
            x,
            masks_attention,
            layers,
            masks,
            cell_only=True,
            ignore_spc_tokens=ignore_spc_tokens,
            coords=coords,
            cond=cond) if need_cell_only_context else None

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
                 gene_panel_cond: bool = False,
                 gene_panel_pos: int | None = None,
                 **base_encoder_kwargs
                 ):
        super().__init__(**base_encoder_kwargs)
        self.n_special_values = n_special_values
        # Gene-panel-size conditioning. When enabled, the special-token slot
        # at ``gene_panel_pos`` carries a CONTINUOUS (effective) panel-size
        # scalar -- read from the non-maskable ``batch['panel_size']`` field --
        # embedded by a dedicated MLP, instead of the discrete
        # ``special_value_embed`` lookup. Panel size is an ordered numeric
        # quantity, so a continuous embedding preserves magnitude/ordering and
        # generalizes to unseen panel sizes. Off by default -> no new params
        # and behaviour is unchanged.
        self.gene_panel_cond = bool(gene_panel_cond)
        self.gene_panel_pos = gene_panel_pos
        self.panel_size_embed = None
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

        # Dedicated continuous embedding for the gene-panel-size scalar.
        if self.gene_panel_cond:
            if self.gene_panel_pos is None:
                raise ValueError(
                    "gene_panel_cond=True requires gene_panel_pos (the "
                    "sequence index of the 'gene_panel' special token).")
            self.panel_size_embed = MLP(
                in_features=1,
                hidden_features=int(self.embed_dim / 2),
                out_features=self.embed_dim,
                bias=self.mlp_bias,
                act_layer=nn.GELU)

    def _apply_panel_size_emb(self,
                              value_emb: torch.Tensor,
                              batch: Mapping[str, torch.Tensor],
                              ) -> torch.Tensor:
        """Override the gene-panel special-token slot's value embedding with a
        continuous embedding of the (normalized) panel-size scalar.

        No-op unless gene-panel conditioning is enabled. The scalar is read
        from ``batch['panel_size']`` ([B] or [B, 1]) -- a non-maskable field,
        so the panel-size signal survives the collator's special-token padding
        (which zeroes the in-sequence special slots at inference)."""
        if self.panel_size_embed is None or batch.get('panel_size') is None:
            return value_emb
        ps = batch['panel_size'].to(value_emb.dtype).reshape(-1, 1)  # [B, 1]
        value_emb[:, self.gene_panel_pos, :] = self.panel_size_embed(ps)
        return value_emb

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

        # Continuous gene-panel-size conditioning (no-op when disabled).
        value_emb = self._apply_panel_size_emb(value_emb, batch)

        # Add positional, segment, and gene embeddings to value embeddings
        x = seg_emb + pos_emb + token_emb + value_emb
        # RDA depth conditioning (no-op when disabled).
        x = self._apply_rda(x, batch)
        # B, N, D = x.shape # B: BATCH_SIZE, N: SEQ_LEN, D: EMBED_DIM

        # Compute attention bias (passes through unless ALiBi is on).
        _keep_idx = (
            masks[0] if (masks is not None and len(masks) == 1) else None
        )
        attn_bias = self._compute_attention_bias(
            batch,
            masks_attention=masks_attention,
            keep_indices=_keep_idx,
        )

        coords = self._build_coords(batch)

        # Per-cell conditioning for AdaLN (or None when off).
        cond = self._compute_cond(batch)

        # Mask token embeddings if masks are provided
        if masks is not None:
            x = apply_masks(x, masks)
            if coords is not None:
                coords = apply_masks(coords, masks)

        # Run forward prop
        for i, blk in enumerate(self.blocks):
            x = blk(x, masks=attn_bias, coords=coords, cond=cond)
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

        # Continuous gene-panel-size conditioning (no-op when disabled).
        value_emb = self._apply_panel_size_emb(value_emb, batch)

        # Add segment, positional, and gene embeddings to value embeddings
        x = seg_emb + pos_emb + token_emb + value_emb # [B, L, D]
        # RDA depth conditioning (no-op when disabled).
        x = self._apply_rda(x, batch)

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

        # Inference path: build coords (for RoPE) and cond (for AdaLN)
        # so models trained with these features can also be embedded.
        coords = self._build_coords(batch)
        cond = self._compute_cond(batch)

        full_ctx: dict[int, torch.Tensor] = self._compute_layer_emb(
            x,
            masks_attention,
            layers,
            masks,
            cell_only=False,
            ignore_spc_tokens=ignore_spc_tokens,
            coords=coords,
            cond=cond)
        cell_only_ctx: dict[int, torch.Tensor] = self._compute_layer_emb(
            x,
            masks_attention,
            layers,
            masks,
            cell_only=True,
            ignore_spc_tokens=ignore_spc_tokens,
            coords=coords,
            cond=cond) if need_cell_only_context else None

        return full_ctx, cell_only_ctx


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

            # Build coords for RoPE (per-token (x, y) matching the
            # concatenated context+target sequence the predictor sees).
            # Returns None for non-rope modes; blk passes it through.
            coords_full = self._build_coords(batch)
            if coords_full is not None:
                pred_coords = self._compose_predictor_coords(
                    coords_full, masks_enc, masks_pred)
            else:
                pred_coords = None

            # Per-cell AdaLN conditioning. Per-cell label is constant
            # across all tokens, so repeating n_pred_masks times
            # matches the outer batch dim after target-mask
            # repetition.
            cond_full = self._compute_cond(batch)
            if cond_full is not None:
                pred_cond = cond_full.repeat(len(masks_pred), 1)
            else:
                pred_cond = None

            # Run forward prop
            for blk in self.predictor_blocks:
                z = blk(z, masks=masks_attention,
                        coords=pred_coords, cond=pred_cond)
            z = self.predictor_norm(z)

            # Return predictions for (target) mask tokens
            z = z[:, :pred_tokens.size(1), :]

            # MLP projection layer
            z = self.predictor_proj(z)

            # Special-token MoE: summed per-slot additive bias on the
            # predictor output (no-op when disabled).
            z = self._apply_special_token_moe(z, batch, len(masks_pred))

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

        # Build coords for RoPE (per-token (x, y) for the concatenated
        # context+target sequence the predictor sees). None for non-rope
        # modes; Block.forward passes through unchanged in that case.
        coords_full = self._build_coords(batch)
        if coords_full is not None:
            pred_coords = self._compose_predictor_coords(
                coords_full, masks_enc, masks_pred)
        else:
            pred_coords = None

        # Per-cell AdaLN conditioning. The predictor's concatenated
        # sequence (context + target) has shape (B * n_pred_masks,
        # ..., D). The per-cell label is constant across all tokens
        # of a given batch element, so we just repeat it n_pred_masks
        # times to match the outer batch dim of z.
        cond_full = self._compute_cond(batch)
        if cond_full is not None:
            pred_cond = cond_full.repeat(len(masks_pred), 1)
        else:
            pred_cond = None

        # Run forward prop
        for blk in self.predictor_blocks:
            z = blk(z, masks=masks_attention,
                    coords=pred_coords, cond=pred_cond)
        z = self.predictor_norm(z)

        # Return predictions for (target) mask tokens
        z = z[:, :pred_tokens.size(1), :]

        # MLP projection layer
        z = self.predictor_proj(z)

        # Special-token MoE: summed per-slot additive bias on the
        # predictor output (no-op when disabled).
        z = self._apply_special_token_moe(z, batch, len(masks_pred))

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

        # Build coords for RoPE (per-token (x, y) for the concatenated
        # context+target sequence the predictor sees). None for non-rope
        # modes; Block.forward passes through unchanged in that case.
        coords_full = self._build_coords(batch)
        if coords_full is not None:
            pred_coords = self._compose_predictor_coords(
                coords_full, masks_enc, masks_pred)
        else:
            pred_coords = None

        # Per-cell AdaLN conditioning. The predictor's concatenated
        # sequence (context + target) has shape (B * n_pred_masks,
        # ..., D). The per-cell label is constant across all tokens
        # of a given batch element, so we just repeat it n_pred_masks
        # times to match the outer batch dim of z.
        cond_full = self._compute_cond(batch)
        if cond_full is not None:
            pred_cond = cond_full.repeat(len(masks_pred), 1)
        else:
            pred_cond = None

        # Run forward prop
        for blk in self.predictor_blocks:
            z = blk(z, masks=masks_attention,
                    coords=pred_coords, cond=pred_cond)
        z = self.predictor_norm(z)

        # Return predictions for (target) mask tokens
        z = z[:, :pred_tokens.size(1), :]

        # MLP projection layer
        z = self.predictor_proj(z)

        # Special-token MoE: summed per-slot additive bias on the
        # predictor output (no-op when disabled).
        z = self._apply_special_token_moe(z, batch, len(masks_pred))

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