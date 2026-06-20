"""Tests for the protein-embedding initialization of the gene-token
embedding layer (UCE-style: frozen ESM + learnable projection).
"""

import json
import pickle
from pathlib import Path

import pytest
import torch

from terra.models.protein_init import (
    ProteinInitTokenEmbedding,
    _warm_start_init_tensor,
    build_aligned_protein_matrix,
    build_protein_init_token_embedding,
    build_warm_start_token_embedding,
    load_protein_embeddings,
)
import torch.nn as nn


def _make_synthetic_token_dict():
    """Mimic the shape of TERRA's token_dictionary_homo_sapiens.pkl:
    a handful of special tokens at the low IDs, then ENSG-prefixed gene
    tokens. Token 0 is <pad>."""
    return {
        "<pad>":         0,
        "cls_0":         1,
        "cls_1":         2,
        "spt_batch":     3,
        "spv_batch_a":   4,
        "ENSG00000000001": 5,  # gene with ESM embedding
        "ENSG00000000002": 6,  # gene with ESM embedding
        "ENSG00000000003": 7,  # gene WITHOUT ESM embedding (missing)
        "ENSG00000000004": 8,  # gene with ESM embedding
    }


def _make_synthetic_protein_matrix(esm_dim: int = 16):
    """Three rows of distinctive ESM-like vectors for genes 1, 2, 4."""
    return torch.tensor([
        [1.0] * esm_dim,                          # row 0 -> ENSG...0001
        [-1.0] * esm_dim,                         # row 1 -> ENSG...0002
        [float(i) for i in range(esm_dim)],       # row 2 -> ENSG...0004
    ], dtype=torch.float32)


def _make_synthetic_mapping():
    return {
        "ENSG00000000001": 0,
        "ENSG00000000002": 1,
        "ENSG00000000004": 2,
        # ENSG...0003 deliberately absent — missing-gene case.
    }


def test_build_aligned_protein_matrix_basic():
    token_dict = _make_synthetic_token_dict()
    protein_matrix = _make_synthetic_protein_matrix(esm_dim=16)
    mapping = _make_synthetic_mapping()

    aligned, gene_mask, stats = build_aligned_protein_matrix(
        token_dict=token_dict,
        protein_matrix=protein_matrix,
        ensembl_to_row=mapping,
        effective_vocab_size=len(token_dict),
        sep_gene_tokens_neb=False,
    )

    # Shape
    assert aligned.shape == (len(token_dict), 16)
    assert gene_mask.shape == (len(token_dict),)
    assert gene_mask.dtype == torch.bool

    # Gene tokens with ESM embeddings: rows match source matrix
    assert torch.equal(aligned[5], protein_matrix[0])  # ENSG...0001
    assert torch.equal(aligned[6], protein_matrix[1])  # ENSG...0002
    assert torch.equal(aligned[8], protein_matrix[2])  # ENSG...0004

    # Missing gene token (ENSG...0003): zero row, mask False
    assert torch.all(aligned[7] == 0.0)
    assert gene_mask[7].item() is False

    # Special tokens: zero rows, mask False
    for tid in [0, 1, 2, 3, 4]:
        assert torch.all(aligned[tid] == 0.0)
        assert gene_mask[tid].item() is False

    # Gene tokens covered: mask True
    for tid in [5, 6, 8]:
        assert gene_mask[tid].item() is True

    # Stats
    assert stats["n_gene_tokens_in_dict"] == 4
    assert stats["n_gene_tokens_covered"] == 3
    assert stats["n_gene_tokens_missing_in_esm"] == 1
    assert stats["coverage_fraction"] == pytest.approx(0.75)
    assert "ENSG00000000003" in stats["missing_gene_examples"]


def _make_synthetic_mouse_token_dict():
    """Mimic the shape of TERRA's token_dictionary_mus_musculus.pkl:
    special tokens at low IDs, then ENSMUSG-prefixed gene tokens."""
    return {
        "<pad>":              0,
        "cls_0":              1,
        "spt_batch":          2,
        "ENSMUSG00000000001": 3,  # covered
        "ENSMUSG00000000002": 4,  # missing from ESM file
        "ENSMUSG00000000003": 5,  # covered
    }


def test_build_aligned_protein_matrix_mouse():
    """Default prefix tuple must accept ENSMUSG-keyed mouse token dicts.
    Without this, a mouse run would route every gene through the
    learnable-random branch with zero protein init — silent failure
    mode worth a regression test."""
    token_dict = _make_synthetic_mouse_token_dict()
    protein_matrix = torch.tensor([
        [2.0] * 16,                            # row 0 -> ENSMUSG...0001
        [float(i) * 0.5 for i in range(16)],   # row 1 -> ENSMUSG...0003
    ], dtype=torch.float32)
    mapping = {
        "ENSMUSG00000000001": 0,
        "ENSMUSG00000000003": 1,
        # ENSMUSG...0002 deliberately absent.
    }
    aligned, gene_mask, stats = build_aligned_protein_matrix(
        token_dict=token_dict,
        protein_matrix=protein_matrix,
        ensembl_to_row=mapping,
        effective_vocab_size=len(token_dict),
    )
    assert torch.equal(aligned[3], protein_matrix[0])  # ENSMUSG...0001
    assert torch.equal(aligned[5], protein_matrix[1])  # ENSMUSG...0003
    assert torch.all(aligned[4] == 0.0)                # missing
    assert gene_mask.tolist() == [False, False, False, True, False, True]
    assert stats["n_gene_tokens_in_dict"] == 3
    assert stats["n_gene_tokens_covered"] == 2
    assert "ENSMUSG" in stats["gene_id_prefixes"]


def test_build_aligned_protein_matrix_custom_prefix():
    """Caller can override gene_id_prefixes to restrict or extend
    detection (e.g. zebrafish ENSDARG)."""
    token_dict = {"<pad>": 0, "ENSDARG00000000001": 1, "ENSG00000000001": 2}
    protein_matrix = torch.zeros(1, 4)
    mapping = {"ENSDARG00000000001": 0}
    _, gene_mask, _ = build_aligned_protein_matrix(
        token_dict=token_dict,
        protein_matrix=protein_matrix,
        ensembl_to_row=mapping,
        effective_vocab_size=len(token_dict),
        gene_id_prefixes=("ENSDARG",),
    )
    # ENSDARG matches; ENSG is excluded by the restricted prefix tuple.
    assert gene_mask.tolist() == [False, True, False]


def test_build_aligned_protein_matrix_sep_gene_tokens_neb():
    """When sep_gene_tokens_neb=True the encoder doubles its vocab so a
    gene gets distinct IDs in cell vs. neighborhood context. The aligned
    matrix must mirror the first half into the second half so both
    point at the same protein row."""
    token_dict = _make_synthetic_token_dict()
    protein_matrix = _make_synthetic_protein_matrix(esm_dim=16)
    mapping = _make_synthetic_mapping()
    base = len(token_dict)

    aligned, gene_mask, stats = build_aligned_protein_matrix(
        token_dict=token_dict,
        protein_matrix=protein_matrix,
        ensembl_to_row=mapping,
        effective_vocab_size=base * 2,
        sep_gene_tokens_neb=True,
    )

    assert aligned.shape == (2 * base, 16)
    # Second-half rows must mirror first-half rows.
    assert torch.equal(aligned[:base], aligned[base:])
    assert torch.equal(gene_mask[:base], gene_mask[base:])

    # Neighborhood-context copy of ENSG...0001 (cell-context ID = 5)
    # lives at ID base + 5 and points at the same ESM row.
    assert torch.equal(aligned[base + 5], protein_matrix[0])


def test_build_aligned_protein_matrix_mismatched_vocab_raises():
    token_dict = _make_synthetic_token_dict()
    protein_matrix = _make_synthetic_protein_matrix()
    mapping = _make_synthetic_mapping()
    with pytest.raises(ValueError, match="effective_vocab_size"):
        build_aligned_protein_matrix(
            token_dict=token_dict,
            protein_matrix=protein_matrix,
            ensembl_to_row=mapping,
            effective_vocab_size=len(token_dict) + 1,  # off by one
            sep_gene_tokens_neb=False,
        )


def test_protein_init_token_embedding_forward_routes_correctly():
    token_dict = _make_synthetic_token_dict()
    protein_matrix = _make_synthetic_protein_matrix(esm_dim=16)
    mapping = _make_synthetic_mapping()
    aligned, gene_mask, _ = build_aligned_protein_matrix(
        token_dict=token_dict,
        protein_matrix=protein_matrix,
        ensembl_to_row=mapping,
        effective_vocab_size=len(token_dict),
    )
    embed_dim = 8
    module = ProteinInitTokenEmbedding(
        vocab_size=len(token_dict),
        embed_dim=embed_dim,
        protein_matrix=aligned,
        gene_mask=gene_mask,
    )

    # Shape check
    tokens = torch.tensor([[0, 1, 5, 6, 7, 8]], dtype=torch.long)
    out = module(tokens)
    assert out.shape == (1, 6, embed_dim)

    # Gene-token rows match proj(esm_row)
    for tid in [5, 6, 8]:
        expected = module.protein_proj(
            module.protein_emb(torch.tensor([tid])))
        out_at = module(torch.tensor([[tid]]))
        torch.testing.assert_close(out_at, expected.unsqueeze(0))

    # Special / missing tokens match special_emb(tid)
    for tid in [1, 3, 7]:
        expected = module.special_emb(torch.tensor([tid]))
        out_at = module(torch.tensor([[tid]]))
        torch.testing.assert_close(out_at, expected.unsqueeze(0))


def test_protein_init_token_embedding_pad_is_zero():
    token_dict = _make_synthetic_token_dict()
    protein_matrix = _make_synthetic_protein_matrix(esm_dim=16)
    mapping = _make_synthetic_mapping()
    aligned, gene_mask, _ = build_aligned_protein_matrix(
        token_dict=token_dict,
        protein_matrix=protein_matrix,
        ensembl_to_row=mapping,
        effective_vocab_size=len(token_dict),
    )
    module = ProteinInitTokenEmbedding(
        vocab_size=len(token_dict),
        embed_dim=8,
        protein_matrix=aligned,
        gene_mask=gene_mask,
        padding_idx=0,
    )
    out = module(torch.tensor([[0]], dtype=torch.long))
    # padding_idx is non-gene, routes through special_emb whose row 0
    # is zeroed by padding_idx semantics. proj_bias=False so the
    # alternative branch would also produce zeros, but the route uses
    # special_emb because gene_mask[0] is False.
    assert torch.all(out == 0.0)


def test_protein_init_token_embedding_frozen_protein_emb():
    """The ESM matrix must not receive gradient updates. The projection
    and the special-token table must."""
    token_dict = _make_synthetic_token_dict()
    protein_matrix = _make_synthetic_protein_matrix(esm_dim=16)
    mapping = _make_synthetic_mapping()
    aligned, gene_mask, _ = build_aligned_protein_matrix(
        token_dict=token_dict,
        protein_matrix=protein_matrix,
        ensembl_to_row=mapping,
        effective_vocab_size=len(token_dict),
    )
    module = ProteinInitTokenEmbedding(
        vocab_size=len(token_dict),
        embed_dim=8,
        protein_matrix=aligned,
        gene_mask=gene_mask,
    )

    assert module.protein_emb.weight.requires_grad is False
    assert module.protein_proj.weight.requires_grad is True
    assert module.special_emb.weight.requires_grad is True

    # Backward through gene + special tokens; verify gradients land
    # only on protein_proj and special_emb, not protein_emb.
    tokens = torch.tensor([[1, 5, 6, 8]], dtype=torch.long)
    out = module(tokens)
    out.sum().backward()
    assert module.protein_emb.weight.grad is None
    assert module.protein_proj.weight.grad is not None
    assert module.special_emb.weight.grad is not None


def test_protein_init_token_embedding_handles_arbitrary_shape():
    """The wrapper must accept any LongTensor shape that nn.Embedding
    would accept — encoders call it as `self.token_embed(batch['tokens'])`
    with shape (B, L)."""
    token_dict = _make_synthetic_token_dict()
    protein_matrix = _make_synthetic_protein_matrix(esm_dim=16)
    mapping = _make_synthetic_mapping()
    aligned, gene_mask, _ = build_aligned_protein_matrix(
        token_dict=token_dict,
        protein_matrix=protein_matrix,
        ensembl_to_row=mapping,
        effective_vocab_size=len(token_dict),
    )
    module = ProteinInitTokenEmbedding(
        vocab_size=len(token_dict),
        embed_dim=8,
        protein_matrix=aligned,
        gene_mask=gene_mask,
    )
    for shape in [(4,), (3, 5), (2, 3, 7)]:
        tokens = torch.randint(0, len(token_dict), shape, dtype=torch.long)
        out = module(tokens)
        assert out.shape == shape + (8,)


def test_load_protein_embeddings_round_trip(tmp_path: Path):
    matrix = _make_synthetic_protein_matrix(esm_dim=16)
    mapping = _make_synthetic_mapping()
    emb_path = tmp_path / "esm.pt"
    map_path = tmp_path / "esm.json"
    torch.save(matrix, str(emb_path))
    with open(map_path, "w") as f:
        json.dump(mapping, f)

    loaded_matrix, loaded_mapping = load_protein_embeddings(emb_path, map_path)
    assert torch.equal(loaded_matrix, matrix)
    assert loaded_mapping == mapping


def test_load_protein_embeddings_rejects_oob_row(tmp_path: Path):
    matrix = _make_synthetic_protein_matrix(esm_dim=16)  # 3 rows
    bad_mapping = {"ENSG00000000001": 0, "ENSG00000000099": 99}
    emb_path = tmp_path / "esm.pt"
    map_path = tmp_path / "esm.json"
    torch.save(matrix, str(emb_path))
    with open(map_path, "w") as f:
        json.dump(bad_mapping, f)
    with pytest.raises(ValueError, match="row index"):
        load_protein_embeddings(emb_path, map_path)


def test_build_protein_init_token_embedding_end_to_end(tmp_path: Path):
    token_dict = _make_synthetic_token_dict()
    matrix = _make_synthetic_protein_matrix(esm_dim=16)
    mapping = _make_synthetic_mapping()
    emb_path = tmp_path / "esm.pt"
    map_path = tmp_path / "esm.json"
    torch.save(matrix, str(emb_path))
    with open(map_path, "w") as f:
        json.dump(mapping, f)

    module = build_protein_init_token_embedding(
        token_dict=token_dict,
        embedding_path=emb_path,
        mapping_path=map_path,
        effective_vocab_size=len(token_dict),
        embed_dim=8,
        sep_gene_tokens_neb=False,
        log=False,
    )
    assert isinstance(module, ProteinInitTokenEmbedding)
    assert module.vocab_size == len(token_dict)
    assert module.embed_dim == 8
    assert module.esm_dim == 16
    assert module.protein_emb.weight.requires_grad is False


def test_encoder_with_protein_init_swaps_token_embed(tmp_path: Path):
    """Sanity-check the integration: building a real encoder with
    protein_init_kwargs replaces `self.token_embed` with the routed
    module without breaking the existing forward signature."""
    from terra.models.gene_transformers import GeneTransformerRankEncoder

    token_dict = _make_synthetic_token_dict()
    matrix = _make_synthetic_protein_matrix(esm_dim=16)
    mapping = _make_synthetic_mapping()
    emb_path = tmp_path / "esm.pt"
    map_path = tmp_path / "esm.json"
    torch.save(matrix, str(emb_path))
    with open(map_path, "w") as f:
        json.dump(mapping, f)

    # Realistic-ish but tiny encoder. seq_len must be (n_special + cell + neb).
    n_special = 4
    cell = 4
    neb = 8
    seq_len = n_special + cell + neb
    n_segments = 2

    encoder = GeneTransformerRankEncoder(
        vocab_size=len(token_dict),
        seq_len=seq_len,
        n_special_tokens=n_special,
        n_segments=n_segments,
        cell_pos_enc="segment",
        embed_dim=8,
        depth=1,
        num_heads=2,
        use_flash_attention=False,
        protein_init_kwargs={
            "token_dict": token_dict,
            "embedding_path": str(emb_path),
            "mapping_path": str(map_path),
        },
    )
    assert isinstance(encoder.token_embed, ProteinInitTokenEmbedding)
    assert encoder.token_embed.protein_emb.weight.requires_grad is False


# ---------------------------------------------------------------------------
# Warm-start mode tests
# ---------------------------------------------------------------------------

def test_warm_start_skips_pca_when_dims_match():
    """When ``embed_dim == esm_dim``, the warm-start path should copy
    ESM rows directly (no centering, no SVD rotation) so users can
    reason about gene-token embeddings in protein space."""
    token_dict = _make_synthetic_token_dict()
    esm_dim = 16
    protein_matrix = _make_synthetic_protein_matrix(esm_dim=esm_dim)
    mapping = _make_synthetic_mapping()
    aligned, gene_mask, _ = build_aligned_protein_matrix(
        token_dict=token_dict,
        protein_matrix=protein_matrix,
        ensembl_to_row=mapping,
        effective_vocab_size=len(token_dict),
    )
    init, stats = _warm_start_init_tensor(
        aligned=aligned,
        gene_mask=gene_mask,
        embed_dim=esm_dim,  # match
        target_std=1.0,
        seed=0,
    )
    assert stats["pca_skipped_dim_match"] is True
    assert stats["pca_variance_retained"] == 1.0
    # Direct-copy path preserves the direction of each gene row.
    # Rescaling is a single positive scalar applied uniformly, so the
    # cosine similarity between any two gene rows must match the cosine
    # similarity between the source ESM rows for those tokens.
    gene_token_ids = [tid for k, tid in token_dict.items()
                      if k.startswith("ENSG") and k in mapping]
    a, b = gene_token_ids[0], gene_token_ids[1]
    src_a = protein_matrix[mapping[
        next(k for k, v in token_dict.items() if v == a)]]
    src_b = protein_matrix[mapping[
        next(k for k, v in token_dict.items() if v == b)]]
    cos_src = float(torch.nn.functional.cosine_similarity(
        src_a.unsqueeze(0), src_b.unsqueeze(0)).item())
    cos_init = float(torch.nn.functional.cosine_similarity(
        init[a].unsqueeze(0), init[b].unsqueeze(0)).item())
    assert abs(cos_src - cos_init) < 1e-5, (
        f"Direct-copy mode must preserve angles between gene rows "
        f"(got cos {cos_init} vs source {cos_src}).")


def test_warm_start_uses_pca_when_dims_differ():
    """When dims differ, PCA-reduce and report variance retained."""
    token_dict = _make_synthetic_token_dict()
    esm_dim = 16
    embed_dim = 8
    protein_matrix = _make_synthetic_protein_matrix(esm_dim=esm_dim)
    mapping = _make_synthetic_mapping()
    aligned, gene_mask, _ = build_aligned_protein_matrix(
        token_dict=token_dict,
        protein_matrix=protein_matrix,
        ensembl_to_row=mapping,
        effective_vocab_size=len(token_dict),
    )
    init, stats = _warm_start_init_tensor(
        aligned=aligned,
        gene_mask=gene_mask,
        embed_dim=embed_dim,
        target_std=1.0,
        seed=0,
    )
    assert stats["pca_skipped_dim_match"] is False
    assert stats["pca_components_used"] == embed_dim
    assert init.shape == (len(token_dict), embed_dim)


def test_warm_start_padding_row_is_zero():
    token_dict = _make_synthetic_token_dict()
    protein_matrix = _make_synthetic_protein_matrix(esm_dim=16)
    mapping = _make_synthetic_mapping()
    aligned, gene_mask, _ = build_aligned_protein_matrix(
        token_dict=token_dict,
        protein_matrix=protein_matrix,
        ensembl_to_row=mapping,
        effective_vocab_size=len(token_dict),
    )
    init, _ = _warm_start_init_tensor(
        aligned=aligned, gene_mask=gene_mask, embed_dim=8,
        padding_idx=0, target_std=1.0, seed=0,
    )
    assert torch.all(init[0] == 0.0)


def test_warm_start_builder_returns_plain_nn_embedding(tmp_path: Path):
    """The whole point of warm-start mode is that the encoder ends up
    with the same module type as baseline. Confirm that the builder
    returns a vanilla nn.Embedding, not a ProteinInitTokenEmbedding."""
    token_dict = _make_synthetic_token_dict()
    matrix = _make_synthetic_protein_matrix(esm_dim=16)
    mapping = _make_synthetic_mapping()
    emb_path = tmp_path / "esm.pt"
    map_path = tmp_path / "esm.json"
    torch.save(matrix, str(emb_path))
    with open(map_path, "w") as f:
        json.dump(mapping, f)

    emb = build_warm_start_token_embedding(
        token_dict=token_dict,
        embedding_path=emb_path,
        mapping_path=map_path,
        effective_vocab_size=len(token_dict),
        embed_dim=8,
        log=False,
    )
    assert type(emb) is nn.Embedding
    assert emb.weight.shape == (len(token_dict), 8)
    assert emb.weight.requires_grad is True
    assert torch.all(emb.weight[0] == 0.0)  # pad row stays zero
