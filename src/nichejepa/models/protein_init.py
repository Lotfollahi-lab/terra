"""
Protein-embedding initialization for the gene-token embedding layer.

UCE-style (Rosen et al. 2023): the gene-token embedding is sourced from
a frozen per-gene protein embedding (ESM-2 / ESM-C) and adapted into the
encoder's embed_dim by a learnable linear projection. Special tokens and
gene tokens missing from the protein file use a separate learnable
embedding table.

Inputs expected:
    - protein matrix : torch.Tensor of shape (N_proteins, esm_dim),
      saved via torch.save.
    - mapping        : JSON file mapping {ensembl_gene_id: row_index}.

The aligned matrix is built once at module-init time and shaped to
(effective_vocab_size, esm_dim) so it can be looked up directly by
NicheJEPA's token IDs without per-batch translation.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn

from .utils import trunc_normal_

logger = logging.getLogger(__name__)

# Ensembl gene-ID prefixes per species. NicheJEPA's token dictionaries
# key gene tokens by Ensembl gene ID; the precompute script writes
# protein embeddings using the same identifier. We accept multiple
# prefixes so a single helper works for human + mouse (most common) and
# can be extended without changing call sites (e.g. zebrafish ENSDARG).
DEFAULT_ENSEMBL_GENE_ID_PREFIXES: Tuple[str, ...] = ("ENSG", "ENSMUSG")


def load_protein_embeddings(
        embedding_path: str | Path,
        mapping_path: str | Path,
        ) -> Tuple[torch.Tensor, Dict[str, int]]:
    """Load the ESM embedding matrix and its Ensembl-row mapping.

    Parameters
    ----------
    embedding_path:
        Path to a torch-saved tensor of shape (N_proteins, esm_dim).
    mapping_path:
        Path to a JSON file mapping ``{ensembl_gene_id: row_index}``.

    Returns
    -------
    matrix:
        Float32 tensor on CPU.
    mapping:
        Dict from Ensembl gene ID to the matrix row index.
    """
    matrix = torch.load(str(embedding_path), map_location="cpu", weights_only=True)
    if not isinstance(matrix, torch.Tensor):
        raise TypeError(
            f"{embedding_path}: expected a torch.Tensor, got {type(matrix)}.")
    if matrix.dim() != 2:
        raise ValueError(
            f"{embedding_path}: expected a 2-D tensor, got shape {tuple(matrix.shape)}.")
    matrix = matrix.float()

    with open(mapping_path, "r") as f:
        mapping = json.load(f)
    if not isinstance(mapping, dict):
        raise TypeError(
            f"{mapping_path}: expected a JSON object (dict), got {type(mapping)}.")
    if not all(isinstance(v, int) for v in mapping.values()):
        raise TypeError(
            f"{mapping_path}: all row indices must be int.")
    if mapping and max(mapping.values()) >= matrix.shape[0]:
        raise ValueError(
            f"{mapping_path}: max row index {max(mapping.values())} >= "
            f"matrix rows {matrix.shape[0]}.")
    return matrix, mapping


def build_aligned_protein_matrix(
        token_dict: Dict[str, int],
        protein_matrix: torch.Tensor,
        ensembl_to_row: Dict[str, int],
        effective_vocab_size: int,
        sep_gene_tokens_neb: bool = False,
        gene_id_prefixes: Sequence[str] = DEFAULT_ENSEMBL_GENE_ID_PREFIXES,
        ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """Build a per-token-ID aligned ESM matrix.

    Iterates over ``token_dict`` and, for each key that starts with
    one of ``gene_id_prefixes`` (default: human ``ENSG`` + mouse
    ``ENSMUSG``) and is present in ``ensembl_to_row``, copies the
    corresponding protein-embedding row into ``aligned[token_id]`` and
    flips ``gene_mask[token_id]`` to True. Special-token keys (``<pad>``,
    ``cls_*``, ``spt_*``, ``spv_*``, etc.) and Ensembl IDs not present
    in the protein file leave their rows as zero with ``gene_mask=False``;
    the routed embedding module (see ``ProteinInitTokenEmbedding``)
    sends those token IDs through a separate learnable table.

    When ``sep_gene_tokens_neb`` is enabled, the encoder doubles its
    vocab so that gene token ``t`` in the cell context has a distinct
    token ID ``t + base_vocab_size`` in the neighborhood context. Since
    both IDs refer to the same protein, we mirror the first half of the
    aligned matrix into the second half.

    Parameters
    ----------
    token_dict:
        NicheJEPA token-name → token-id mapping (from the pickled
        ``token_dictionary_*.pkl``).
    protein_matrix:
        Tensor of shape (N_proteins, esm_dim).
    ensembl_to_row:
        Mapping from Ensembl gene ID to its row in ``protein_matrix``.
    effective_vocab_size:
        Size of the encoder's token-embedding table. This is
        ``len(token_dict)`` when ``sep_gene_tokens_neb=False``, and
        ``2 * len(token_dict)`` when True.
    sep_gene_tokens_neb:
        Mirrors the encoder flag of the same name.

    Returns
    -------
    aligned:
        Float32 tensor of shape (effective_vocab_size, esm_dim). Rows
        with no protein embedding are zero.
    gene_mask:
        Bool tensor of shape (effective_vocab_size,). True where the
        token ID has a real protein embedding.
    stats:
        Coverage diagnostics for logging.
    """
    if protein_matrix.dim() != 2:
        raise ValueError(
            f"protein_matrix must be 2-D, got shape {tuple(protein_matrix.shape)}.")
    base_vocab_size = len(token_dict)
    expected_eff = base_vocab_size * (2 if sep_gene_tokens_neb else 1)
    if effective_vocab_size != expected_eff:
        raise ValueError(
            f"effective_vocab_size={effective_vocab_size} does not match "
            f"len(token_dict)={base_vocab_size} with "
            f"sep_gene_tokens_neb={sep_gene_tokens_neb} (expected {expected_eff}).")

    # Normalize to a tuple so str.startswith() can match any of them.
    prefixes = tuple(gene_id_prefixes)
    if not prefixes:
        raise ValueError(
            "gene_id_prefixes must be non-empty (got an empty sequence).")

    esm_dim = protein_matrix.shape[1]
    aligned = torch.zeros(effective_vocab_size, esm_dim, dtype=torch.float32)
    gene_mask = torch.zeros(effective_vocab_size, dtype=torch.bool)

    n_gene_tokens = 0
    n_covered = 0
    missing_examples = []

    for token_str, token_id in token_dict.items():
        if not token_str.startswith(prefixes):
            # Special tokens (<pad>, cls_*, spt_*, spv_*, etc.) go through
            # the learnable special-emb branch; leave aligned row = 0.
            continue
        if not (0 <= token_id < base_vocab_size):
            raise ValueError(
                f"token_dict[{token_str!r}] = {token_id} out of range "
                f"[0, {base_vocab_size}).")
        n_gene_tokens += 1
        row_idx = ensembl_to_row.get(token_str)
        if row_idx is None:
            if len(missing_examples) < 10:
                missing_examples.append(token_str)
            continue
        aligned[token_id] = protein_matrix[row_idx]
        gene_mask[token_id] = True
        n_covered += 1

    if sep_gene_tokens_neb:
        offset = base_vocab_size
        aligned[offset:offset + base_vocab_size] = aligned[:base_vocab_size]
        gene_mask[offset:offset + base_vocab_size] = gene_mask[:base_vocab_size]

    coverage = n_covered / n_gene_tokens if n_gene_tokens else 0.0
    n_missing = n_gene_tokens - n_covered
    stats = {
        "effective_vocab_size": effective_vocab_size,
        "base_vocab_size": base_vocab_size,
        "n_gene_tokens_in_dict": n_gene_tokens,
        "n_gene_tokens_covered": n_covered,
        "n_gene_tokens_missing_in_esm": n_missing,
        "n_non_gene_tokens": base_vocab_size - n_gene_tokens,
        "coverage_fraction": coverage,
        "missing_gene_examples": missing_examples,
        "esm_dim": esm_dim,
        "gene_id_prefixes": list(prefixes),
    }
    return aligned, gene_mask, stats


class ProteinInitTokenEmbedding(nn.Module):
    """Routed token embedding: frozen ESM + learnable projection for gene
    tokens, learnable embedding for special tokens and missing genes.

    Drop-in replacement for ``nn.Embedding(vocab_size, embed_dim,
    padding_idx=0)``: takes a LongTensor of token IDs of arbitrary
    shape, returns a Tensor with an appended ``embed_dim`` axis.

    Two branches are always computed every forward pass; ``torch.where``
    selects the appropriate one per token ID, so gradients for the
    unused branch are zero by construction. This keeps the call site
    identical to ``nn.Embedding(tokens)`` in every encoder variant.
    """

    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 protein_matrix: torch.Tensor,
                 gene_mask: torch.Tensor,
                 padding_idx: int = 0,
                 init_std: float = 0.02,
                 proj_bias: bool = False,
                 use_layer_norm: bool = True,
                 freeze_esm: bool = True,
                 ):
        super().__init__()
        if protein_matrix.shape[0] != vocab_size:
            raise ValueError(
                f"protein_matrix.shape[0]={protein_matrix.shape[0]} must "
                f"equal vocab_size={vocab_size}.")
        if gene_mask.shape != (vocab_size,):
            raise ValueError(
                f"gene_mask must have shape ({vocab_size},), got {tuple(gene_mask.shape)}.")
        esm_dim = protein_matrix.shape[1]

        # ESM lookup. `freeze_esm=True` (UCE-style) keeps the matrix
        # frozen forever -- gives a stable protein prior but caps
        # per-gene expressiveness at whatever the projection can do.
        # `freeze_esm=False` uses ESM as an *initialization* and lets
        # the matrix train -- gives the model full per-gene capacity
        # (same as baseline) starting from a sensible point. The
        # `_no_weight_decay` marker is honored by init_opt so that the
        # prior is not eroded by weight decay over training.
        self.protein_emb = nn.Embedding.from_pretrained(
            protein_matrix.float().contiguous(),
            freeze=freeze_esm,
            padding_idx=padding_idx,
        )
        self.freeze_esm = freeze_esm
        if not freeze_esm:
            # Flag for init_opt to route this parameter into the
            # no-weight-decay group.
            self.protein_emb.weight._no_weight_decay = True

        # LayerNorm on the ESM input before projection. Without this,
        # the projection has to absorb per-gene magnitude variation in
        # the (raw) ESM-C output -- short / disordered / domain-rich
        # proteins have wildly different vector norms, so the projection
        # output scale ends up gene-dependent, which the rest of the
        # encoder can't easily compensate for. This matches UCE's
        # standard pattern of feeding ESM through a LayerNorm before
        # the downstream model.
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.protein_norm = nn.LayerNorm(esm_dim)
        else:
            self.protein_norm = nn.Identity()

        self.protein_proj = nn.Linear(esm_dim, embed_dim, bias=proj_bias)

        # Use nn.Embedding's *default* init -- N(0, 1) per element, with
        # row `padding_idx` zeroed. That matches the baseline encoder
        # (which is a plain nn.Embedding for the whole vocab and isn't
        # touched by the codebase's trunc_normal_(0.02) Linear init).
        # IMPORTANT: do NOT trunc_normal_(0.02) here -- it makes special
        # tokens (CLS / batch / spv_* / spt_*) start ~50x smaller than
        # the segment / value embeddings they're summed with, which
        # silently destroys early training.
        self.special_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)

        self.register_buffer("gene_mask", gene_mask.to(torch.bool), persistent=True)

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.esm_dim = esm_dim
        self.padding_idx = padding_idx

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        proj_emb = self.protein_proj(self.protein_norm(self.protein_emb(tokens)))
        spec_emb = self.special_emb(tokens)
        route = self.gene_mask[tokens].unsqueeze(-1)
        return torch.where(route, proj_emb, spec_emb)

    def extra_repr(self) -> str:
        return (f"vocab_size={self.vocab_size}, embed_dim={self.embed_dim}, "
                f"esm_dim={self.esm_dim}, padding_idx={self.padding_idx}, "
                f"use_layer_norm={self.use_layer_norm}, "
                f"freeze_esm={self.freeze_esm}, "
                f"n_gene_tokens={int(self.gene_mask.sum().item())}")


def _warm_start_init_tensor(
        aligned: torch.Tensor,
        gene_mask: torch.Tensor,
        embed_dim: int,
        padding_idx: int = 0,
        target_std: float = 1.0,
        seed: int = 0,
        ) -> Tuple[torch.Tensor, dict]:
    """Build a (vocab, embed_dim) initialization tensor from ESM.

    Used by warm-start mode: gene-token rows are PCA-reduced from
    ``aligned`` to ``embed_dim`` so they carry the ESM signal in the
    encoder's own dimension, while non-gene tokens get the same
    N(0, 1) init as a plain ``nn.Embedding``. The output is written
    directly into a ``nn.Embedding`` whose architecture is identical
    to the baseline, so subsequent training dynamics are unchanged.

    PCA is computed on the gene rows of ``aligned`` only. Non-gene
    rows are left as random N(0, 1) so special tokens behave exactly
    like the baseline. The pad row (``padding_idx``) is zeroed last.

    The output is rescaled so its per-element std across the full
    vocab matches ``target_std`` (default 1.0, matching baseline
    nn.Embedding's default init). This keeps gene-token magnitudes
    comparable to special tokens.
    """
    vocab_size, esm_dim = aligned.shape
    stats: dict = {"target_std": target_std, "esm_dim": esm_dim}

    # Random non-gene rows -- match baseline nn.Embedding default
    # init (mean=0, std=1) so special tokens behave identically.
    g = torch.Generator(device="cpu").manual_seed(seed)
    init = torch.randn(vocab_size, embed_dim, generator=g) * target_std

    gene_rows = aligned[gene_mask]
    if gene_rows.shape[0] > 0:
        # Center and PCA-reduce. SVD on (n_genes, esm_dim).
        centered = gene_rows - gene_rows.mean(dim=0, keepdim=True)
        # full_matrices=False -> Vt has shape (min(n_genes, esm_dim), esm_dim)
        _, S, Vt = torch.linalg.svd(centered, full_matrices=False)
        k = min(embed_dim, Vt.shape[0])
        reduced = centered @ Vt[:k].T  # (n_genes, k)
        if k < embed_dim:
            # esm_dim < embed_dim: pad the extra dims with N(0, target_std)
            # so the gene rows don't have a structurally-zero subspace.
            pad = torch.randn(
                reduced.shape[0], embed_dim - k, generator=g
            ) * target_std
            reduced = torch.cat([reduced, pad], dim=1)
        # Rescale so per-element std matches target. Each PCA dim has
        # a different natural variance (decreasing with k); a uniform
        # rescale to target_std keeps the relative variances intact
        # while matching baseline magnitude overall.
        current_std = float(reduced.std().item())
        if current_std > 1e-8:
            reduced = reduced * (target_std / current_std)
        init[gene_mask] = reduced.to(init.dtype)
        # Variance retained by the top-`k` components.
        total_var = float((S ** 2).sum().item())
        kept_var = float((S[:k] ** 2).sum().item())
        stats["pca_variance_retained"] = kept_var / max(total_var, 1e-12)
        stats["pca_components_used"] = int(k)
        stats["pca_target_components"] = int(embed_dim)
    else:
        stats["pca_variance_retained"] = 0.0
        stats["pca_components_used"] = 0
        stats["pca_target_components"] = int(embed_dim)

    # Pad row is always zero.
    init[padding_idx] = 0.0
    stats["init_elem_std_overall"] = float(init.std().item())
    stats["init_elem_std_genes"] = float(init[gene_mask].std().item()) if gene_mask.any() else 0.0
    stats["init_elem_std_special"] = (
        float(init[~gene_mask].std().item()) if (~gene_mask).any() else 0.0
    )
    return init, stats


def build_warm_start_token_embedding(
        token_dict: Dict[str, int],
        embedding_path: str | Path,
        mapping_path: str | Path,
        effective_vocab_size: int,
        embed_dim: int,
        sep_gene_tokens_neb: bool = False,
        padding_idx: int = 0,
        gene_id_prefixes: Sequence[str] = DEFAULT_ENSEMBL_GENE_ID_PREFIXES,
        target_std: float = 1.0,
        seed: int = 0,
        log: bool = True,
        ) -> nn.Embedding:
    """Build a plain ``nn.Embedding`` initialized from ESM via PCA.

    Returns the same type and shape as the baseline encoder's
    ``self.token_embed``, so the only thing that differs from baseline
    is the initial weight values for gene-token rows. No routing
    module, no LayerNorm in the forward path, no separate
    ``special_emb`` table.
    """
    protein_matrix, ensembl_to_row = load_protein_embeddings(
        embedding_path, mapping_path)
    aligned, gene_mask, align_stats = build_aligned_protein_matrix(
        token_dict=token_dict,
        protein_matrix=protein_matrix,
        ensembl_to_row=ensembl_to_row,
        effective_vocab_size=effective_vocab_size,
        sep_gene_tokens_neb=sep_gene_tokens_neb,
        gene_id_prefixes=gene_id_prefixes,
    )
    init_tensor, init_stats = _warm_start_init_tensor(
        aligned=aligned,
        gene_mask=gene_mask,
        embed_dim=embed_dim,
        padding_idx=padding_idx,
        target_std=target_std,
        seed=seed,
    )
    emb = nn.Embedding(
        effective_vocab_size, embed_dim, padding_idx=padding_idx)
    with torch.no_grad():
        emb.weight.copy_(init_tensor)

    if log:
        banner = (
            "\n"
            "================================================================\n"
            " PROTEIN-INIT TOKEN EMBEDDING -- WARM-START MODE\n"
            "----------------------------------------------------------------\n"
            "   (PCA-reduced ESM written into a plain nn.Embedding; "
            "architecture identical to baseline.)\n"
            f"   Prefixes recognized as genes : {align_stats['gene_id_prefixes']}\n"
            f"   ESM dim -> encoder embed_dim : "
            f"{align_stats['esm_dim']} -> {embed_dim} via PCA\n"
            f"   Gene tokens covered by ESM   : "
            f"{align_stats['n_gene_tokens_covered']}/{align_stats['n_gene_tokens_in_dict']} "
            f"({100.0 * align_stats['coverage_fraction']:.2f}%)\n"
            f"   PCA components used          : "
            f"{init_stats['pca_components_used']}/"
            f"{init_stats['pca_target_components']} "
            f"(variance retained "
            f"{100.0 * init_stats['pca_variance_retained']:.2f}%)\n"
            f"   target per-elem std          : {target_std:.3f}\n"
            f"   init elem_std (overall)      : "
            f"{init_stats['init_elem_std_overall']:.3f}\n"
            f"   init elem_std (gene rows)    : "
            f"{init_stats['init_elem_std_genes']:.3f}\n"
            f"   init elem_std (special rows) : "
            f"{init_stats['init_elem_std_special']:.3f}\n"
            f"   Effective vocab size         : "
            f"{align_stats['effective_vocab_size']} "
            f"(sep_gene_tokens_neb={sep_gene_tokens_neb})\n"
            f"   All params trainable         : "
            f"{emb.weight.numel():,} (single nn.Embedding, no projection)\n"
            "================================================================"
        )
        logger.info(banner)

    return emb


def build_protein_init_token_embedding(
        token_dict: Dict[str, int],
        embedding_path: str | Path,
        mapping_path: str | Path,
        effective_vocab_size: int,
        embed_dim: int,
        sep_gene_tokens_neb: bool = False,
        padding_idx: int = 0,
        init_std: float = 0.02,
        proj_bias: bool = False,
        use_layer_norm: bool = True,
        freeze_esm: bool = True,
        gene_id_prefixes: Sequence[str] = DEFAULT_ENSEMBL_GENE_ID_PREFIXES,
        log: bool = True,
        ) -> ProteinInitTokenEmbedding:
    """End-to-end constructor: load files, align, instantiate module.

    The encoder's ``__init__`` calls this to swap its ``self.token_embed``
    when a ``protein_init`` config section is provided.
    """
    protein_matrix, ensembl_to_row = load_protein_embeddings(
        embedding_path, mapping_path)
    aligned, gene_mask, stats = build_aligned_protein_matrix(
        token_dict=token_dict,
        protein_matrix=protein_matrix,
        ensembl_to_row=ensembl_to_row,
        effective_vocab_size=effective_vocab_size,
        sep_gene_tokens_neb=sep_gene_tokens_neb,
        gene_id_prefixes=gene_id_prefixes,
    )
    module = ProteinInitTokenEmbedding(
        vocab_size=effective_vocab_size,
        embed_dim=embed_dim,
        protein_matrix=aligned,
        gene_mask=gene_mask,
        padding_idx=padding_idx,
        init_std=init_std,
        proj_bias=proj_bias,
        use_layer_norm=use_layer_norm,
        freeze_esm=freeze_esm,
    )
    if log:
        # Multi-line banner so it's obvious in the training logs that
        # protein-init actually fired and which numbers it produced.
        # Logged at INFO so it appears alongside the rest of the
        # encoder construction info (rank 0 only).
        n_esm_params = int(module.protein_emb.weight.numel())
        n_trainable_proj = int(module.protein_proj.weight.numel())
        if module.protein_proj.bias is not None:
            n_trainable_proj += int(module.protein_proj.bias.numel())
        n_trainable_special = int(module.special_emb.weight.numel())
        n_trainable_ln = (sum(p.numel() for p in module.protein_norm.parameters())
                          if use_layer_norm else 0)
        esm_status = (
            f"FROZEN ({n_esm_params:,} params)" if freeze_esm
            else f"TRAINABLE ({n_esm_params:,} params, weight decay excluded)"
        )
        missing_preview = (
            ", ".join(stats["missing_gene_examples"])
            if stats["missing_gene_examples"] else "<none>"
        )

        # Scale diagnostic: compare init-time magnitudes of the gene-
        # token branch vs the special-token branch. If these differ by
        # more than ~5x, the additive `seg + token + value` sum is
        # dominated by one branch and training will struggle.
        with torch.no_grad():
            gene_token_ids = torch.nonzero(gene_mask, as_tuple=False).squeeze(-1)
            spec_token_ids = torch.nonzero(~gene_mask, as_tuple=False).squeeze(-1)
            # Sample up to 256 of each to keep the diagnostic cheap.
            if gene_token_ids.numel() > 256:
                gene_token_ids = gene_token_ids[
                    torch.randperm(gene_token_ids.numel())[:256]]
            if spec_token_ids.numel() > 256:
                # Skip row padding_idx since it's zero by design.
                spec_token_ids = spec_token_ids[spec_token_ids != padding_idx]
                if spec_token_ids.numel() > 256:
                    spec_token_ids = spec_token_ids[
                        torch.randperm(spec_token_ids.numel())[:256]]
            gene_out = module.protein_proj(
                module.protein_norm(module.protein_emb(gene_token_ids)))
            spec_out = module.special_emb(spec_token_ids)
            esm_raw = module.protein_emb(gene_token_ids)
            esm_elem_std = float(esm_raw.std().item())
            esm_row_norm = float(esm_raw.norm(dim=-1).mean().item())
            gene_elem_std = float(gene_out.std().item())
            gene_row_norm = float(gene_out.norm(dim=-1).mean().item())
            spec_elem_std = float(spec_out.std().item())
            spec_row_norm = float(spec_out.norm(dim=-1).mean().item())
            scale_ratio = gene_elem_std / max(spec_elem_std, 1e-8)

        banner = (
            "\n"
            "================================================================\n"
            " PROTEIN-INIT TOKEN EMBEDDING -- INITIALIZED\n"
            "----------------------------------------------------------------\n"
            f"   Prefixes recognized as genes : {stats['gene_id_prefixes']}\n"
            f"   ESM dim -> encoder embed_dim : "
            f"{stats['esm_dim']} -> {embed_dim}\n"
            f"   Gene tokens covered by ESM   : "
            f"{stats['n_gene_tokens_covered']}/{stats['n_gene_tokens_in_dict']} "
            f"({100.0 * stats['coverage_fraction']:.2f}%)\n"
            f"   Gene tokens missing from ESM : "
            f"{stats['n_gene_tokens_missing_in_esm']} "
            f"(random init, frozen via special_emb)\n"
            f"   Non-gene (special) tokens    : "
            f"{stats['n_non_gene_tokens']} (learnable via special_emb)\n"
            f"   Effective vocab size         : "
            f"{stats['effective_vocab_size']} "
            f"(sep_gene_tokens_neb={sep_gene_tokens_neb})\n"
            f"   LayerNorm before projection  : {use_layer_norm}\n"
            f"   ESM matrix                   : {esm_status}\n"
            f"   Trainable: projection        : {n_trainable_proj:,} "
            f"({stats['esm_dim']}x{embed_dim}, bias={proj_bias})\n"
            f"   Trainable: LayerNorm         : {n_trainable_ln:,}\n"
            f"   Trainable: special embedding : {n_trainable_special:,} "
            f"({stats['effective_vocab_size']}x{embed_dim})\n"
            "   Init-time scale diagnostics --\n"
            f"     raw ESM       : elem_std={esm_elem_std:.3f}, "
            f"row_norm={esm_row_norm:.2f}\n"
            f"     gene branch   : elem_std={gene_elem_std:.3f}, "
            f"row_norm={gene_row_norm:.2f}  (proj output)\n"
            f"     special branch: elem_std={spec_elem_std:.3f}, "
            f"row_norm={spec_row_norm:.2f}  (target ~1.0)\n"
            f"     gene/special elem_std ratio : {scale_ratio:.2f}x "
            f"(target ~1.0; >5x or <0.2x means scale mismatch)\n"
            f"   Missing-gene examples        : {missing_preview}\n"
            "================================================================"
        )
        logger.info(banner)
    return module
