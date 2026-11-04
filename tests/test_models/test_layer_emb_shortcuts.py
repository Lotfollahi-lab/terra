"""
Tests for the inference shortcuts of ``_compute_layer_emb``.

Without JEPA masks, the full context passes the attention mask to the blocks
as it is (a ``[B, 1, 1, L]`` key padding mask is not copied to
``[B, 1, L, L]``), and the cell-only context is computed on the special-token
and index-cell block alone. Both give the embeddings of the full-length
computation (``layer_emb_shortcuts = False``); the neighborhood positions of
the cell-only context are zero. With JEPA masks nothing changes.
"""

import pytest
import torch

from terra.utils.helper import init_model

B, SEQ_LEN_CELL, N_SEGMENTS, VOCAB_SIZE, DEPTH = 2, 8, 4, 300, 2
L = SEQ_LEN_CELL * N_SEGMENTS


def _batch():
    generator = torch.Generator().manual_seed(0)
    tokens = torch.randint(110, VOCAB_SIZE, (B, L), generator=generator)
    tokens[:, 6:SEQ_LEN_CELL] = 0  # padding inside segments
    tokens[:, SEQ_LEN_CELL + 5:2 * SEQ_LEN_CELL] = 0
    segments = torch.arange(1, N_SEGMENTS + 1).repeat_interleave(
        SEQ_LEN_CELL).expand(B, L).clone()
    positions = torch.arange(1, SEQ_LEN_CELL + 1).repeat(
        N_SEGMENTS).expand(B, L).clone()
    rel_x = (torch.rand(B, N_SEGMENTS, generator=generator) * 100
             ).repeat_interleave(SEQ_LEN_CELL, dim=1)
    rel_y = (torch.rand(B, N_SEGMENTS, generator=generator) * 100
             ).repeat_interleave(SEQ_LEN_CELL, dim=1)
    rel_x[:, :SEQ_LEN_CELL] = 0
    rel_y[:, :SEQ_LEN_CELL] = 0
    values = torch.rand(B, L, generator=generator) * 3
    pad = tokens == 0
    segments[pad] = 0
    positions[pad] = 0
    values[pad] = 0
    rel_x[pad] = float('-inf')
    rel_y[pad] = float('-inf')
    batch = dict(tokens=tokens, segments=segments, positions=positions,
                 values=values, rel_x_coords=rel_x, rel_y_coords=rel_y)
    return batch, (~pad)[:, None, None, :]


def _encoder(cell_pos_enc='segment', use_flash_attention=True,
             n_special_tokens=0):
    torch.manual_seed(0)
    encoder, _ = init_model(
        gt_type='combined', count_encoding='mlp', n_value_bins=20,
        cell_pos_enc=cell_pos_enc, device='cpu', vocab_size=VOCAB_SIZE,
        seq_len=L + n_special_tokens, n_special_tokens=n_special_tokens,
        n_segments=N_SEGMENTS, n_special_values=0, enc_emb_dim=32,
        enc_depth=DEPTH, pred_emb_dim=16, pred_depth=1, num_heads=4,
        use_flash_attention=use_flash_attention, mlp_bias=True)
    return encoder.eval()


def _layer_emb(backbone, shortcuts, batch, masks_attention, **kwargs):
    backbone.layer_emb_shortcuts = shortcuts
    with torch.no_grad():
        return backbone.return_layer_emb(
            layers=[1, DEPTH], batch=batch, masks_attention=masks_attention,
            **kwargs)


def _block_calls(backbone):
    """Record the sequence length and mask shape of every call of the
    first block."""
    calls = []

    def hook(module, args, kwargs):
        masks = kwargs.get('masks')
        calls.append((args[0].shape[1],
                      None if masks is None else tuple(masks.shape)))
    return calls, backbone.blocks[0].register_forward_pre_hook(
        hook, with_kwargs=True)


@pytest.mark.parametrize('use_flash_attention', [False, True])
@pytest.mark.parametrize('cell_pos_enc',
                         ['segment', 'coord', 'rope', 'alibi', 'polar+alibi'])
def test_shortcuts_give_the_full_length_embeddings(cell_pos_enc,
                                                   use_flash_attention):
    backbone = _encoder(cell_pos_enc, use_flash_attention).backbone
    batch, masks_attention = _batch()
    full, cell_only = _layer_emb(backbone, False, batch, masks_attention)
    fast_full, fast_cell_only = _layer_emb(backbone, True, batch,
                                           masks_attention)
    for layer in (1, DEPTH):
        torch.testing.assert_close(fast_full[layer], full[layer],
                                   rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(
            fast_cell_only[layer][:, :SEQ_LEN_CELL],
            cell_only[layer][:, :SEQ_LEN_CELL], rtol=1e-5, atol=1e-6)
        # Not computed: the neighborhood positions of the cell-only context
        assert torch.count_nonzero(
            fast_cell_only[layer][:, SEQ_LEN_CELL:]) == 0
        assert fast_cell_only[layer].shape == cell_only[layer].shape


def test_shortcuts_skip_the_mask_copy_and_the_neighborhood():
    backbone = _encoder().backbone
    batch, masks_attention = _batch()
    calls, handle = _block_calls(backbone)
    _layer_emb(backbone, True, batch, masks_attention)
    _layer_emb(backbone, False, batch, masks_attention)
    handle.remove()
    assert calls == [
        # Shortcuts: full context with the key padding mask, cell-only
        # context on the index cell
        (L, (B, 1, 1, L)), (SEQ_LEN_CELL, (B, 1, 1, SEQ_LEN_CELL)),
        # Full-length computation: copied [B, 1, L, L] masks
        (L, (B, 1, L, L)), (L, (B, 1, L, L))]


def test_masked_forward_keeps_the_full_length_computation():
    # With JEPA masks (here keeping every token) both contexts are computed
    # at full length, with or without shortcuts
    backbone = _encoder().backbone
    batch, masks_attention = _batch()
    masks = [torch.arange(L).expand(B, L)]
    calls, handle = _block_calls(backbone)
    with_shortcuts = _layer_emb(backbone, True, batch, masks_attention,
                                masks=masks)
    handle.remove()
    without = _layer_emb(backbone, False, batch, masks_attention, masks=masks)
    assert [length for length, _ in calls] == [L, L]
    for fast, slow in zip(with_shortcuts, without):
        for layer in (1, DEPTH):
            assert torch.equal(fast[layer], slow[layer])


@pytest.mark.parametrize('mask_shape', ['key_padding', 'full'])
def test_cell_block_includes_the_special_tokens(mask_shape):
    # As in the released models: one special token before the index cell
    backbone = _encoder(n_special_tokens=1).backbone
    generator = torch.Generator().manual_seed(1)
    x = torch.randn(B, 1 + L, backbone.embed_dim, generator=generator)
    attn_base = torch.ones(B, 1, 1, 1 + L, dtype=torch.bool)
    attn_base[:, :, :, 1 + 6:1 + SEQ_LEN_CELL] = False
    attn_base[:, :, :, 1 + SEQ_LEN_CELL + 5:1 + 2 * SEQ_LEN_CELL] = False
    if mask_shape == 'full':
        attn_base = attn_base.expand(-1, -1, 1 + L, -1).clone()
    outputs = {}
    for shortcuts in (False, True):
        backbone.layer_emb_shortcuts = shortcuts
        with torch.no_grad():
            outputs[shortcuts] = [
                backbone._compute_layer_emb(
                    x.clone(), attn_base, [DEPTH], None, cell_only=cell_only)
                for cell_only in (False, True)]
    (full, cell_only), (fast_full, fast_cell_only) = (outputs[False],
                                                      outputs[True])
    torch.testing.assert_close(fast_full[DEPTH], full[DEPTH],
                               rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(fast_cell_only[DEPTH][:, :SEQ_LEN_CELL],
                               cell_only[DEPTH][:, :SEQ_LEN_CELL],
                               rtol=1e-5, atol=1e-6)
    assert torch.count_nonzero(fast_cell_only[DEPTH][:, SEQ_LEN_CELL:]) == 0


def test_cell_only_context_without_attention_mask_stays_full_length():
    # Without an attention mask the index cell would see the zeroed
    # neighborhood tokens, so the cell-only shortcut does not apply
    backbone = _encoder().backbone
    batch, _ = _batch()
    calls, handle = _block_calls(backbone)
    with_shortcuts = _layer_emb(backbone, True, batch, None)
    handle.remove()
    without = _layer_emb(backbone, False, batch, None)
    assert [length for length, _ in calls] == [L, L]
    for fast, slow in zip(with_shortcuts, without):
        torch.testing.assert_close(fast[DEPTH], slow[DEPTH],
                                   rtol=1e-5, atol=1e-6)
