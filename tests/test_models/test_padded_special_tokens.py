"""
Tests for batches whose special tokens (metatokens) are padded.

With ``mask.special_token_pad_ratio > 0`` the mask collator pads the special
tokens of a random share of the batches, and the context and target masks of
those batches start with gene tokens. The ``new_spc`` predictor layout
``[context special tokens, target tokens after the special tokens, context
gene tokens]`` assumed that both masks start with the special tokens, so in
padded batches it put the first context gene token into the first target's
slot and dropped that target's mask token. The predictor now takes
``pad_special_tokens`` and uses the plain ``[target tokens, context tokens]``
concatenation for padded batches; batches with special tokens are unchanged.
"""

import pytest
import torch
from datasets import Dataset

from terra.datasets.cell_datasets import init_cell_dataset
from terra.masks.block_masking import BlockMaskCollator
from terra.masks.utils import padding_key_masks
from terra.utils.helper import init_model

SEQ_LEN_CELL, N_SEGMENTS, N_CELLS = 8, 4, 8
SEQ_LEN_NEIGHBORHOOD = SEQ_LEN_CELL * (N_SEGMENTS - 1)
SEQ_LEN = SEQ_LEN_CELL + SEQ_LEN_NEIGHBORHOOD + 1
# Large enough for the token ids of rank metatokens (value + 21957)
VOCAB_SIZE = 23407


def _hf_dataset(seed: int = 3) -> Dataset:
    rng = torch.Generator().manual_seed(seed)
    rows = {'gene_tokens': [], 'gene_expr': [], 'n_nonzero_tokens': [],
            'batch_value': []}
    for _ in range(N_CELLS):
        tokens, values = [], []
        for _ in range(N_SEGMENTS):
            n = int(torch.randint(3, SEQ_LEN_CELL + 1, (1,), generator=rng))
            tokens += (torch.randperm(500, generator=rng)[:n] + 107).tolist()
            tokens += [0] * (SEQ_LEN_CELL - n)
            values += sorted(torch.randint(1, 50, (n,), generator=rng)
                             .float().tolist(), reverse=True)
            values += [0.0] * (SEQ_LEN_CELL - n)
        rows['gene_tokens'].append(tokens)
        rows['gene_expr'].append(values)
        rows['n_nonzero_tokens'].append(sum(t != 0 for t in tokens))
        rows['batch_value'].append([213])
    dataset = Dataset.from_dict(rows)
    dataset.set_format(type='torch', columns=list(rows),
                       output_all_columns=True)
    return dataset


def _batch(gt_type: str, special_token_pad_ratio: float,
           restrict_special_attention: bool = False):
    cell_dataset = init_cell_dataset(
        tokenizer_type='cell_graph', dataset=_hf_dataset(),
        vocab_size=VOCAB_SIZE, seq_len_cell=SEQ_LEN_CELL,
        seq_len_neighborhood=SEQ_LEN_NEIGHBORHOOD, gt_type=gt_type,
        cell_pos_enc='segment', special_tokens=['batch'],
        sampling_strategy=None, n_nonzero_tokens_list=None,
        include_cell_id=False, sep_gene_tokens_neb=False, nz_spc=True)
    collator = BlockMaskCollator(
        n_targets=1, n_contexts=1, n_segments=N_SEGMENTS,
        seq_len_cell=SEQ_LEN_CELL,
        seq_len_neighborhood=SEQ_LEN_NEIGHBORHOOD, n_special_tokens=1,
        per_block_mask_ratio=0.6, sample_segments=False,
        sample_gene_masks=True,
        special_token_pad_ratio=special_token_pad_ratio,
        restrict_special_attention=restrict_special_attention)
    torch.manual_seed(1)
    return collator([cell_dataset[i] for i in range(N_CELLS)])


def _model(gt_type: str, new_spc: bool, use_flash_attention: bool = True):
    torch.manual_seed(0)
    encoder, predictor = init_model(
        gt_type=gt_type, count_encoding='value_bins', n_value_bins=20,
        cell_pos_enc='segment', device='cpu', vocab_size=VOCAB_SIZE,
        seq_len=SEQ_LEN, n_special_tokens=1, n_segments=N_SEGMENTS,
        n_special_values=1448, enc_emb_dim=32, enc_depth=2, pred_emb_dim=16,
        pred_depth=2, num_heads=4, use_flash_attention=use_flash_attention,
        nz_spc=True, new_spc=new_spc, mlp_bias=True)
    with torch.no_grad():
        # A non-zero mask token, so that mask tokens and context embeddings
        # cannot be confused
        predictor.backbone.mask_token.normal_()
    return encoder.eval(), predictor.eval()


def _predict(encoder, predictor, batch, **kwargs):
    udata, masks_enc, masks_pred, _, _ = batch
    # [N_MASKS, B, L] -> list of [B, L], as in the training loop
    masks_enc, masks_pred = list(masks_enc), list(masks_pred)
    with torch.no_grad():
        z, token_emb = encoder(batch=udata, masks=masks_enc)
        return predictor(z=z, token_emb=token_emb, batch=udata,
                         masks_enc=masks_enc, masks_pred=masks_pred,
                         masks_attention=None, **kwargs)[0]


@pytest.mark.parametrize('gt_type', ['rank', 'counts', 'combined'])
def test_padded_batch_uses_plain_layout(gt_type):
    batch = _batch(gt_type, special_token_pad_ratio=1.0)
    udata, masks_enc, masks_pred, _, pad_special_tokens = batch
    assert pad_special_tokens
    # Both masks start with gene tokens
    assert (masks_enc[0][:, 0] >= 1).all() and (masks_pred[0][:, 0] >= 1).all()

    encoder, predictor = _model(gt_type, new_spc=True)
    _, plain_predictor = _model(gt_type, new_spc=False)
    plain_predictor.load_state_dict(predictor.state_dict())

    out = _predict(encoder, predictor, batch, pad_special_tokens=True)
    expected = _predict(encoder, plain_predictor, batch)
    assert out.shape == (N_CELLS, masks_pred[0].shape[1], 32)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_padded_batch_first_target_gets_its_mask_token():
    batch = _batch('combined', special_token_pad_ratio=1.0)
    udata, _, masks_pred, _, _ = batch
    encoder, predictor = _model('combined', new_spc=True)
    out = _predict(encoder, predictor, batch, pad_special_tokens=True)

    # Move the first target of every cell to another position id: the first
    # prediction must follow it (it ignored it with the new_spc layout)
    moved = {k: v.clone() for k, v in udata.items()}
    cells = torch.arange(N_CELLS)
    moved['positions'][cells, masks_pred[0][:, 0]] += 3
    out_moved = _predict(encoder, predictor, (moved,) + batch[1:],
                         pad_special_tokens=True)
    assert (out_moved[:, 0] - out[:, 0]).abs().max() > 1e-4


def test_batches_with_special_tokens_keep_new_spc_layout():
    batch = _batch('combined', special_token_pad_ratio=0.0)
    assert not batch[-1]
    encoder, predictor = _model('combined', new_spc=True)
    _, plain_predictor = _model('combined', new_spc=False)
    plain_predictor.load_state_dict(predictor.state_dict())

    out = _predict(encoder, predictor, batch)
    torch.testing.assert_close(
        _predict(encoder, predictor, batch, pad_special_tokens=False), out,
        rtol=0, atol=0)
    # The new_spc layout is used: it differs from the plain concatenation
    assert not torch.allclose(out, _predict(encoder, plain_predictor, batch))


def test_padding_key_masks_follow_the_padded_layout():
    ctx_valid = torch.tensor([[True, True, True, False]])
    tgt_valid = torch.tensor([[True, True, False]])
    _, pred_mask = padding_key_masks(
        ctx_valid, tgt_valid, n_special_tokens=1, new_spc=True,
        pad_special_tokens=True)
    _, plain_mask = padding_key_masks(
        ctx_valid, tgt_valid, n_special_tokens=1, new_spc=False)
    torch.testing.assert_close(pred_mask, plain_mask)
    _, new_spc_mask = padding_key_masks(
        ctx_valid, tgt_valid, n_special_tokens=1, new_spc=True)
    assert not torch.equal(new_spc_mask, plain_mask)
