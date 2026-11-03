"""
Tests for the embedding path of encoders with ALiBi position encodings.

``return_layer_emb`` (used by ``embed_dataset``, ``gene_embed_dataset`` and
``infer``) passed the attention mask to the blocks unchanged, without the
ALiBi distance bias that the forward pass of training adds for
``cell_pos_enc`` 'alibi' and 'polar+alibi', so the embeddings of such models
differed from the model's own forward pass. It now adds the bias; in the
cell-only context the blocked positions of the additive bias are -inf.
Other encodings are unchanged.
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


def _encoder(cell_pos_enc, use_flash_attention):
    torch.manual_seed(0)
    encoder, _ = init_model(
        gt_type='combined', count_encoding='mlp', n_value_bins=20,
        cell_pos_enc=cell_pos_enc, device='cpu', vocab_size=VOCAB_SIZE,
        seq_len=L, n_special_tokens=0, n_segments=N_SEGMENTS,
        n_special_values=0, enc_emb_dim=32, enc_depth=DEPTH, pred_emb_dim=16,
        pred_depth=1, num_heads=4, use_flash_attention=use_flash_attention,
        mlp_bias=True)
    return encoder.eval()


@pytest.mark.parametrize('use_flash_attention', [False, True])
@pytest.mark.parametrize('cell_pos_enc',
                         ['segment', 'coord', 'alibi', 'polar+alibi'])
def test_embeddings_match_the_forward_pass(cell_pos_enc,
                                           use_flash_attention):
    encoder = _encoder(cell_pos_enc, use_flash_attention)
    batch, masks_attention = _batch()
    with torch.no_grad():
        forward, _ = encoder(batch=batch, masks=None,
                             masks_attention=masks_attention)
        full_ctx, _ = encoder.backbone.return_layer_emb(
            layers=[DEPTH], batch=batch, masks_attention=masks_attention,
            need_cell_only_context=False)
    genes = batch['tokens'] != 0
    torch.testing.assert_close(full_ctx[DEPTH][genes], forward[genes],
                               rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('cell_pos_enc',
                         ['segment', 'coord', 'alibi', 'polar+alibi'])
def test_cell_only_context_sees_only_the_index_cell(cell_pos_enc):
    encoder = _encoder(cell_pos_enc, use_flash_attention=False)
    batch, masks_attention = _batch()
    index_cell = {key: value[:, :SEQ_LEN_CELL] for key, value in batch.items()}
    with torch.no_grad():
        _, cell_only_ctx = encoder.backbone.return_layer_emb(
            layers=[DEPTH], batch=batch, masks_attention=masks_attention)
        alone, _ = encoder.backbone.return_layer_emb(
            layers=[DEPTH], batch=index_cell,
            masks_attention=masks_attention[..., :SEQ_LEN_CELL],
            need_cell_only_context=False)
    genes = index_cell['tokens'] != 0
    torch.testing.assert_close(
        cell_only_ctx[DEPTH][:, :SEQ_LEN_CELL][genes], alone[DEPTH][genes],
        rtol=1e-5, atol=1e-6)
