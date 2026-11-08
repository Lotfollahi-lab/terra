"""meta.zero_fully_masked_attention_rows: an attention query row whose keys
are all masked (e.g. a cell whose JEPA context is only padding, without a
metatoken) outputs zeros with finite gradients instead of NaN, on every
attention path; all other rows are bit-identical to the unguarded attention.
Off by default."""
import copy

import pytest
import torch

import terra.models.modules as modules
from terra.utils.helper import init_model_from_config, parse_arch_kwargs

N_TOKENS = 6


def _masks(kind):
    # Sample 1 has no valid key in any row ('key_padding', 'additive') or in
    # one query row ('full_bool')
    if kind == 'key_padding':
        masks = torch.ones(3, 1, 1, N_TOKENS, dtype=torch.bool)
        masks[0, ..., 4:] = False
        masks[1] = False
    elif kind == 'full_bool':
        masks = torch.rand(3, 1, N_TOKENS, N_TOKENS,
                           generator=torch.Generator().manual_seed(1)) > 0.3
        masks[..., torch.arange(N_TOKENS), torch.arange(N_TOKENS)] = True
        masks[1, :, 2] = False
    else:
        masks = -torch.rand(3, 2, N_TOKENS, N_TOKENS,
                            generator=torch.Generator().manual_seed(1))
        masks[0, ..., 4:] = float('-inf')
        masks[1] = float('-inf')
    return masks


def _run(attention, masks, guard):
    attention.zero_fully_masked_attention_rows = guard
    x = torch.randn(3, N_TOKENS, 16,
                    generator=torch.Generator().manual_seed(2))
    x.requires_grad_(True)
    out, _ = attention(x, masks=masks)
    return out, x


def _attention(use_flash_attention, preset='auto'):
    torch.manual_seed(0)
    return modules.Attention(16, num_heads=2, qkv_bias=True,
                             use_flash_attention=use_flash_attention,
                             attention_backend=preset).eval()


def test_off_by_default():
    assert _attention(True).zero_fully_masked_attention_rows is False
    assert parse_arch_kwargs({'meta': {}})[
        'zero_fully_masked_attention_rows'] is False


def test_eager_attention_without_guard_gives_nan():
    # Documents the issue: softmax over a row of -inf is NaN. (PyTorch >= 2.5
    # SDPA kernels on CPU return zeros for such rows; whether every GPU kernel
    # does, in the forward and backward pass, depends on the PyTorch build.)
    out, _ = _run(_attention(False), _masks('key_padding'), guard=False)
    assert out[1].isnan().all()
    assert torch.isfinite(out[[0, 2]]).all()


@pytest.mark.parametrize('use_flash_attention,preset', [
    (False, 'auto'), (True, 'auto'), (True, 'math'), (True, 'torch')])
@pytest.mark.parametrize('kind', ['key_padding', 'full_bool', 'additive'])
def test_guard_is_finite_and_leaves_other_rows_unchanged(
        use_flash_attention, preset, kind):
    attention = _attention(use_flash_attention, preset)
    masks = _masks(kind)
    reference, _ = _run(attention, masks, guard=False)
    out, x = _run(attention, masks, guard=True)
    (out ** 2).sum().backward()

    assert torch.isfinite(out).all()
    assert torch.isfinite(x.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in attention.parameters())
    # The input mask is not modified
    assert torch.equal(masks, _masks(kind))
    if kind == 'full_bool':
        # Only query row 2 of sample 1 has no valid key
        keep = torch.ones(3, N_TOKENS, dtype=torch.bool)
        keep[1, 2] = False
        assert torch.equal(out[keep], reference[keep])
        # Its attention output is zero, so only the projection bias remains
        torch.testing.assert_close(out[1, 2], attention.proj.bias,
                                   rtol=0, atol=0)
    else:
        assert torch.equal(out[[0, 2]], reference[[0, 2]])
        torch.testing.assert_close(
            out[1], attention.proj.bias.expand(N_TOKENS, -1), rtol=0, atol=0)


def test_guard_does_not_change_attention_without_fully_masked_rows():
    attention = _attention(True)
    masks = _masks('key_padding')[[0, 2]]
    x = torch.randn(2, N_TOKENS, 16)
    with torch.no_grad():
        reference, _ = attention(x, masks=masks)
        attention.zero_fully_masked_attention_rows = True
        out, _ = attention(x, masks=masks)
        unmasked, _ = attention(x)
        attention.zero_fully_masked_attention_rows = False
        unmasked_reference, _ = attention(x)
    assert torch.equal(out, reference)
    assert torch.equal(unmasked, unmasked_reference)


@pytest.mark.parametrize('use_flash_attention', [False, True])
def test_guard_saves_no_extra_tensors_for_the_backward_pass(
        use_flash_attention):
    def saved_bytes(guard):
        attention = _attention(use_flash_attention).train()
        attention.zero_fully_masked_attention_rows = guard
        x = torch.randn(3, N_TOKENS, 16, requires_grad=True)
        saved = {}

        def pack(t):
            saved[t.data_ptr()] = t.numel() * t.element_size()
            return t
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            _, attn = attention(x, masks=_masks('key_padding'))
        return sum(saved.values()), attn

    reference, _ = saved_bytes(False)
    guarded, attn = saved_bytes(True)
    # Only the (B, 1, N, 1) row mask is added
    assert guarded - reference <= 3 * N_TOKENS
    if not use_flash_attention:
        assert (attn[1] == 0).all()


CONFIG = {
    'data': {'seq_len_cell': 4, 'seq_len_neighborhood': 12, 'n_segments': 4,
             'tokenizer_type': 'cell_graph', 'n_special_values': 8},
    'meta': {'gt_type': 'combined', 'count_encoding': 'value_bins',
             'n_value_bins': 10, 'cell_pos_enc': 'segment', 'enc_emb_dim': 32,
             'enc_depth': 2, 'pred_emb_dim': 16, 'pred_depth': 2,
             'num_heads': 2, 'special_tokens': [], 'use_flash_attention': True,
             'attention_backend': 'torch', 'api_version': 'v3'},
}


@pytest.mark.parametrize('flag', [None, False, True])
def test_option_reaches_every_attention_module(flag):
    config = copy.deepcopy(CONFIG)
    if flag is not None:
        config['meta']['zero_fully_masked_attention_rows'] = flag
    encoder, predictor = init_model_from_config(
        config, {f't{i}': i for i in range(200)}, 'cpu')
    attentions = [m for model in (encoder, predictor)
                  for m in model.modules()
                  if isinstance(m, modules.Attention)]
    assert len(attentions) == 4
    assert all(m.zero_fully_masked_attention_rows is bool(flag)
               for m in attentions)
