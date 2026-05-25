"""Tests for the polar and ALiBi spatial-encoding modes added on top of
the existing 'segment' / 'coord' positional encodings.
"""

import math

import pytest
import torch

from nichejepa.models.gene_transformers import GeneTransformerBaseEncoder


# ---------------------------------------------------------------------------
# Build a small concrete encoder for testing. The base class is ABC, so we
# instantiate the simplest subclass (RankEncoder) at minimal size. We only
# poke at the spatial-encoding methods, not full training.
# ---------------------------------------------------------------------------

def _make_rank_encoder(cell_pos_enc: str, num_heads: int = 4, embed_dim: int = 32):
    from nichejepa.models.gene_transformers import GeneTransformerRankEncoder
    return GeneTransformerRankEncoder(
        vocab_size=16,
        seq_len=12,
        n_special_tokens=2,
        n_segments=2,
        cell_pos_enc=cell_pos_enc,
        embed_dim=embed_dim,
        depth=1,
        num_heads=num_heads,
        use_flash_attention=False,
    )


# ---------------------------------------------------------------------------
# Polar mode
# ---------------------------------------------------------------------------

def test_polar_seg_emb_shape_and_special_tokens():
    """`_get_seg_emb` in polar mode must produce (B, L, embed_dim) with
    special-token positions (rel = -inf) yielding the same const value
    as the existing coord-mode -inf handling produces."""
    enc = _make_rank_encoder("polar", embed_dim=32)
    B, L = 2, 12
    rel_x = torch.randn(B, L)
    rel_y = torch.randn(B, L)
    # Mark the first two positions as special tokens.
    rel_x[:, :2] = float("-inf")
    rel_y[:, :2] = float("-inf")
    seg_emb = enc._get_seg_emb({
        "rel_x_coords": rel_x,
        "rel_y_coords": rel_y,
    })
    assert seg_emb.shape == (B, L, 32)
    # Special-token rows should be the const sincos(0) value (constant
    # across all special positions). All non-special rows differ.
    assert torch.allclose(seg_emb[0, 0], seg_emb[0, 1])
    assert not torch.allclose(seg_emb[0, 0], seg_emb[0, 2])


def test_polar_rotation_changes_only_angle_dims():
    """A pure rotation of (rel_x, rel_y) must leave the radial (first
    embed_dim // 2) sincos block unchanged and only affect the angular
    block. This is the structural promise of the polar parameterization."""
    enc = _make_rank_encoder("polar", embed_dim=32)
    rel_x = torch.tensor([[1.0, 0.5, 2.0, -1.0]])
    rel_y = torch.tensor([[0.0, 0.5, -1.0, 1.0]])
    angle = math.pi / 3  # arbitrary rotation
    c, s = math.cos(angle), math.sin(angle)
    rel_x_rot = c * rel_x - s * rel_y
    rel_y_rot = s * rel_x + c * rel_y

    a = enc._get_seg_emb({"rel_x_coords": rel_x, "rel_y_coords": rel_y})
    b = enc._get_seg_emb({"rel_x_coords": rel_x_rot, "rel_y_coords": rel_y_rot})

    half = a.size(-1) // 2  # split into radial / angular halves
    # Radial half (log(1+r) sincos) -- identical under rotation.
    torch.testing.assert_close(a[..., :half], b[..., :half],
                                rtol=1e-5, atol=1e-5)
    # Angular half (theta sincos) -- different.
    assert not torch.allclose(a[..., half:], b[..., half:],
                              rtol=1e-5, atol=1e-5)


def test_polar_zero_distance_is_well_defined():
    """log(1+r) and atan2(0,0)=0 -- r=0 must not produce NaN/inf."""
    enc = _make_rank_encoder("polar", embed_dim=32)
    rel_x = torch.zeros(1, 4)
    rel_y = torch.zeros(1, 4)
    seg_emb = enc._get_seg_emb({"rel_x_coords": rel_x, "rel_y_coords": rel_y})
    assert torch.isfinite(seg_emb).all()


# ---------------------------------------------------------------------------
# ALiBi mode
# ---------------------------------------------------------------------------

def test_alibi_slopes_are_strictly_positive_and_decreasing():
    """ALiBi slopes must form a strictly decreasing geometric sequence
    starting just below 1.0 (the standard ALiBi recipe). Sanity for
    num_heads = 4, 6, 8, 12 -- including non-power-of-2 cases."""
    for H in (2, 4, 6, 8, 12):
        slopes = GeneTransformerBaseEncoder._get_alibi_slopes(H).tolist()
        assert all(s > 0 for s in slopes), f"non-positive slope at H={H}"
        for a, b in zip(slopes, slopes[1:]):
            assert a > b, f"slopes not strictly decreasing at H={H}: {slopes}"
        assert slopes[0] < 1.0, f"first slope >= 1.0 at H={H}: {slopes[0]}"


def test_alibi_bias_shape_and_decay_with_distance():
    """The alibi bias should be (B, H, L, L), with diagonal zero (no
    self-distance penalty) and off-diagonal entries strictly negative,
    more negative the further apart the cells are."""
    enc = _make_rank_encoder("alibi", num_heads=4, embed_dim=32)
    B, L = 2, 6
    # Cell positions in a straight line: 0, 1, 2, ..., L-1 in x; 0 in y.
    rel_x = torch.arange(L, dtype=torch.float32).unsqueeze(0).expand(B, L).clone()
    rel_y = torch.zeros(B, L)
    bias = enc._compute_alibi_bias({
        "rel_x_coords": rel_x,
        "rel_y_coords": rel_y,
    })
    assert bias.shape == (B, 4, L, L)
    # Diagonal: distance 0 -> bias 0.
    for h in range(4):
        torch.testing.assert_close(
            bias[0, h].diagonal(), torch.zeros(L), atol=1e-6, rtol=0)
    # Bias for cell 0 attending to cell k should monotonically
    # decrease (become more negative) with k.
    for h in range(4):
        row = bias[0, h, 0]
        for k in range(1, L - 1):
            assert row[k] > row[k + 1], (
                f"bias not strictly decreasing along row 0 of head {h}: "
                f"{row.tolist()}"
            )


def test_alibi_bias_zero_for_special_tokens():
    """Special-token positions (rel = -inf) must have zero bias both
    ways -- they sit 'anywhere' spatially and shouldn't be penalized
    for attending to or being attended by real positions."""
    enc = _make_rank_encoder("alibi", num_heads=4, embed_dim=32)
    rel_x = torch.tensor([[float("-inf"), float("-inf"), 0.0, 1.0, 5.0]])
    rel_y = torch.tensor([[float("-inf"), float("-inf"), 0.0, 1.0, 5.0]])
    bias = enc._compute_alibi_bias({
        "rel_x_coords": rel_x,
        "rel_y_coords": rel_y,
    })
    # First two positions are special -- all their bias entries are 0.
    for special_pos in (0, 1):
        torch.testing.assert_close(
            bias[0, :, special_pos, :], torch.zeros_like(bias[0, :, special_pos, :]))
        torch.testing.assert_close(
            bias[0, :, :, special_pos], torch.zeros_like(bias[0, :, :, special_pos]))


def test_alibi_bias_per_head_scaling():
    """Per-head slopes must produce per-head-different magnitudes -- a
    head with a steeper slope penalizes far cells more aggressively
    than a head with a shallow slope."""
    enc = _make_rank_encoder("alibi", num_heads=4, embed_dim=32)
    rel_x = torch.tensor([[0.0, 5.0]])
    rel_y = torch.tensor([[0.0, 0.0]])
    bias = enc._compute_alibi_bias({
        "rel_x_coords": rel_x,
        "rel_y_coords": rel_y,
    })
    # bias[0, h, 0, 1] = -slope[h] * 5. Slopes strictly decrease,
    # so |bias[0, 0, 0, 1]| > |bias[0, 1, 0, 1]| > ... > |bias[0, 3, 0, 1]|.
    far_biases = [bias[0, h, 0, 1].item() for h in range(4)]
    for a, b in zip(far_biases, far_biases[1:]):
        assert a < b, f"far-cell bias not increasing across heads: {far_biases}"


def test_compute_attention_bias_passthrough_for_non_alibi_modes():
    """For 'segment', 'coord', 'polar', the helper must not change the
    incoming masks_attention -- existing configs see byte-identical
    behavior."""
    for mode in ("segment", "coord", "polar"):
        enc = _make_rank_encoder(mode, embed_dim=32)
        # Dummy batch -- not used for non-alibi modes.
        batch = {"rel_x_coords": torch.zeros(1, 4),
                 "rel_y_coords": torch.zeros(1, 4)}
        bool_mask = torch.tensor([[[[True, True, False, True]]]])
        out = enc._compute_attention_bias(batch, masks_attention=bool_mask)
        assert out is bool_mask
        out = enc._compute_attention_bias(batch, masks_attention=None)
        assert out is None


def test_compute_attention_bias_slices_alibi_with_keep_indices():
    """When the encoder applies a JEPA mask, ``_compute_attention_bias``
    must slice the alibi tensor down so its sequence length matches
    the post-mask sequence length."""
    enc = _make_rank_encoder("alibi", num_heads=4, embed_dim=32)
    B, L = 1, 6
    rel_x = torch.arange(L, dtype=torch.float32).unsqueeze(0)
    rel_y = torch.zeros(1, L)
    batch = {"rel_x_coords": rel_x, "rel_y_coords": rel_y}
    keep = torch.tensor([[0, 2, 4]])  # keep positions 0, 2, 4
    sliced = enc._compute_attention_bias(
        batch, masks_attention=None, keep_indices=keep)
    assert sliced.shape == (B, 4, 3, 3)
    # Diagonal still zero (a position has zero distance to itself even
    # after slicing).
    for h in range(4):
        torch.testing.assert_close(
            sliced[0, h].diagonal(), torch.zeros(3), atol=1e-6, rtol=0)


def test_alibi_initializes_slopes_buffer():
    enc = _make_rank_encoder("alibi", num_heads=4, embed_dim=32)
    assert hasattr(enc, "alibi_slopes")
    assert enc.alibi_slopes.shape == (4,)


def test_unknown_cell_pos_enc_still_raises():
    """Backward compat: invalid mode names must still fail loudly."""
    with pytest.raises(Exception):
        # Should fail at _get_seg_emb time (or earlier).
        enc = _make_rank_encoder("bogus_mode")
        enc._get_seg_emb({"segments": torch.zeros(1, 4, dtype=torch.long),
                          "rel_x_coords": torch.zeros(1, 4),
                          "rel_y_coords": torch.zeros(1, 4)})


# ---------------------------------------------------------------------------
# 'polar+alibi' combined mode
# ---------------------------------------------------------------------------

def test_polar_alibi_input_encoding_matches_polar():
    """polar+alibi must produce the same input encoding as polar
    alone -- the only addition is the attention bias side."""
    enc_p = _make_rank_encoder("polar", embed_dim=32)
    enc_pa = _make_rank_encoder("polar+alibi", embed_dim=32, num_heads=4)
    rel_x = torch.randn(2, 12)
    rel_y = torch.randn(2, 12)
    batch = {"rel_x_coords": rel_x, "rel_y_coords": rel_y}
    a = enc_p._get_seg_emb(batch)
    b = enc_pa._get_seg_emb(batch)
    torch.testing.assert_close(a, b, atol=1e-6, rtol=0)


def test_polar_alibi_attention_bias_matches_alibi():
    """polar+alibi must produce the same ALiBi attention bias as
    pure 'alibi' mode (input encoding differs; bias path is identical)."""
    enc_a = _make_rank_encoder("alibi", embed_dim=32, num_heads=4)
    enc_pa = _make_rank_encoder("polar+alibi", embed_dim=32, num_heads=4)
    # ALiBi slopes are deterministic given num_heads, so both encoders
    # produce identical biases on the same input.
    rel_x = torch.tensor([[0.0, 1.0, 2.0, 5.0]])
    rel_y = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    batch = {"rel_x_coords": rel_x, "rel_y_coords": rel_y}
    bias_a = enc_a._compute_alibi_bias(batch)
    bias_pa = enc_pa._compute_alibi_bias(batch)
    torch.testing.assert_close(bias_a, bias_pa, atol=1e-6, rtol=0)


def test_polar_alibi_compute_attention_bias_returns_alibi():
    """In polar+alibi mode the helper must produce the (B, H, L, L)
    alibi bias rather than passing masks_attention through."""
    enc = _make_rank_encoder("polar+alibi", embed_dim=32, num_heads=4)
    rel_x = torch.tensor([[0.0, 1.0, 2.0, 5.0]])
    rel_y = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    batch = {"rel_x_coords": rel_x, "rel_y_coords": rel_y}
    out = enc._compute_attention_bias(batch, masks_attention=None)
    assert out is not None
    assert out.shape == (1, 4, 4, 4)


# ---------------------------------------------------------------------------
# 'laplacian' mode
# ---------------------------------------------------------------------------

def _make_laplacian_batch(B: int, encoder, nz_spc: bool = False):
    """Build a fake batch that obeys the standard sequence layout:
    [n_special tokens][cell_1 tokens][cell_2 tokens]... in n_segments
    blocks of seq_len_cell each. ``nz_spc=True`` mimics the dataset's
    behavior of assigning special tokens non-zero segment IDs (as
    happens in the real configs that triggered the original bug)."""
    L = encoder.seq_len
    n_special = encoder.n_special_tokens
    n_cells = encoder.n_segments
    seq_len_cell = encoder.seq_len_cell

    # Random cell positions in 2D
    cell_positions = torch.randn(B, n_cells, 2)

    # Per-token rel_x / rel_y: broadcast cell position to every token
    # of that cell; special tokens get -inf.
    rel_x = torch.full((B, L), float("-inf"))
    rel_y = torch.full((B, L), float("-inf"))
    segments = torch.zeros(B, L, dtype=torch.long)
    # Mark special-token positions with non-trivial segment IDs in
    # the nz_spc=True case (the dataset uses arange(2, n_special+2)).
    if nz_spc:
        for s in range(n_special):
            segments[:, s] = 2 + s
    for c in range(n_cells):
        start = n_special + c * seq_len_cell
        end = start + seq_len_cell
        rel_x[:, start:end] = cell_positions[:, c, 0:1]
        rel_y[:, start:end] = cell_positions[:, c, 1:2]
        segments[:, start:end] = c + 1  # 1-indexed
    # All tokens non-zero (no pad) so the pad-mask path doesn't fire.
    tokens = torch.ones(B, L, dtype=torch.long)
    return {
        "rel_x_coords": rel_x,
        "rel_y_coords": rel_y,
        "segments": segments,
        "tokens": tokens,
    }, cell_positions


def test_laplacian_pe_shape_and_special_tokens_zero():
    enc = _make_rank_encoder("laplacian", embed_dim=32)
    batch, _ = _make_laplacian_batch(B=2, encoder=enc)
    seg_emb = enc._get_seg_emb(batch)
    assert seg_emb.shape == (2, enc.seq_len, 32)
    # Special-token positions (segment == 0) must be exactly zero.
    special_mask = batch["segments"] == 0
    assert torch.all(seg_emb[special_mask] == 0.0)


def test_laplacian_pe_constant_within_cell():
    """Every token of the same cell shares the same per-cell PE,
    because the per-cell eigenvector is broadcast via segments."""
    enc = _make_rank_encoder("laplacian", embed_dim=32)
    batch, _ = _make_laplacian_batch(B=1, encoder=enc)
    seg_emb = enc._get_seg_emb(batch)
    # Pick cell 1 (segment == 1) and check all its rows are identical.
    mask = batch["segments"][0] == 1
    cell_rows = seg_emb[0][mask]
    assert cell_rows.shape[0] > 1
    torch.testing.assert_close(
        cell_rows, cell_rows[0:1].expand_as(cell_rows),
        atol=1e-5, rtol=1e-5)


def test_laplacian_pe_sign_is_deterministic():
    """The sign-fix convention (first non-zero entry positive) must
    make the PE deterministic across reruns with the same input."""
    enc = _make_rank_encoder("laplacian", embed_dim=32)
    batch, _ = _make_laplacian_batch(B=2, encoder=enc)
    a = enc._get_seg_emb(batch)
    b = enc._get_seg_emb(batch)
    torch.testing.assert_close(a, b, atol=1e-6, rtol=0)


def test_laplacian_pe_changes_with_geometry():
    """Different cell geometries must produce different PE -- if not,
    the encoding carries no spatial info."""
    enc = _make_rank_encoder("laplacian", embed_dim=32)
    b1, _ = _make_laplacian_batch(B=1, encoder=enc)
    # Rebuild with a different random geometry
    torch.manual_seed(42)
    b2, _ = _make_laplacian_batch(B=1, encoder=enc)
    a = enc._get_seg_emb(b1)
    b = enc._get_seg_emb(b2)
    assert not torch.allclose(a, b, atol=1e-3)


def test_laplacian_k_caps_at_n_segments_minus_one():
    """laplacian_k > n_segments - 1 must be silently capped so the
    eigendecomposition stays well-defined."""
    from nichejepa.models.gene_transformers import GeneTransformerRankEncoder
    enc = GeneTransformerRankEncoder(
        vocab_size=16, seq_len=12, n_special_tokens=2, n_segments=2,
        cell_pos_enc="laplacian", embed_dim=32, depth=1, num_heads=4,
        use_flash_attention=False,
        laplacian_k=99,
    )
    # n_segments == 2 -> max usable k is 1
    assert enc.laplacian_k == 1


def test_laplacian_k_zero_raises():
    from nichejepa.models.gene_transformers import GeneTransformerRankEncoder
    with pytest.raises(ValueError, match="laplacian_k"):
        GeneTransformerRankEncoder(
            vocab_size=16, seq_len=12, n_special_tokens=2, n_segments=1,
            cell_pos_enc="laplacian", embed_dim=32, depth=1, num_heads=4,
            use_flash_attention=False,
            laplacian_k=1,
        )


def test_laplacian_pe_handles_nz_spc_special_segments():
    """Regression test for the index-out-of-bounds reported on the
    cluster: with ``nz_spc=True`` the dataset assigns special tokens
    segment IDs that overflow ``n_cells`` (e.g. special token gets
    segment 2 while there are only 2 cells in the test layout). The
    broadcast must be layout-based, not segments-based, so this no
    longer crashes ``torch.gather``."""
    enc = _make_rank_encoder("laplacian", embed_dim=32)
    batch, _ = _make_laplacian_batch(B=2, encoder=enc, nz_spc=True)
    # Sanity: the batch actually has special segments > n_cells
    n_cells = enc.n_segments
    assert (batch["segments"][:, :enc.n_special_tokens] > n_cells - 1).any()
    # Must not raise; must produce zero PE on the special-token prefix.
    seg_emb = enc._get_seg_emb(batch)
    special_block = seg_emb[:, :enc.n_special_tokens, :]
    assert torch.all(special_block == 0.0)


def test_laplacian_pe_pad_tokens_zeroed():
    """Pad tokens within a cell block (token == 0) get zero PE."""
    enc = _make_rank_encoder("laplacian", embed_dim=32)
    batch, _ = _make_laplacian_batch(B=1, encoder=enc)
    # Mark the last position of the first cell as pad.
    pad_pos = enc.n_special_tokens + enc.seq_len_cell - 1
    batch["tokens"][:, pad_pos] = 0
    seg_emb = enc._get_seg_emb(batch)
    assert torch.all(seg_emb[0, pad_pos, :] == 0.0)
    # The rest of cell 1 keeps its PE (non-zero somewhere).
    cell_pos = enc.n_special_tokens
    assert torch.any(seg_emb[0, cell_pos, :] != 0.0)
