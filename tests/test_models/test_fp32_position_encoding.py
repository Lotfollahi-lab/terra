"""meta.fp32_position_encoding: the phases of the coordinate position
encodings are computed in float32 under bf16 autocast (opt-in; the default
keeps the encoding the released models were trained with)."""
import copy

import pytest
import torch

import terra.models.utils as model_utils
from terra.models.rope2d import RoPE2D
from terra.models.utils import (get_1d_sincos_pos_embed_from_coord,
                                sincos_pos_embed_from_coord)
from terra.utils.helper import init_model_from_config, parse_arch_kwargs

CONFIG = {
    'data': {'seq_len_cell': 4, 'seq_len_neighborhood': 12, 'n_segments': 4,
             'tokenizer_type': 'cell_graph', 'n_special_values': 8},
    'meta': {'gt_type': 'combined', 'count_encoding': 'value_bins',
             'n_value_bins': 10, 'cell_pos_enc': 'coord', 'enc_emb_dim': 32,
             'enc_depth': 1, 'pred_emb_dim': 16, 'pred_depth': 1,
             'num_heads': 2, 'special_tokens': [], 'use_flash_attention': True,
             'api_version': 'v3'},
}


def _gpu_device():
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return None


def _autocast_enabled(device_type):
    try:
        return torch.is_autocast_enabled(device_type)
    except TypeError:  # PyTorch < 2.4
        return torch.is_autocast_cpu_enabled()


def _models(cell_pos_enc, fp32):
    config = copy.deepcopy(CONFIG)
    config['meta']['cell_pos_enc'] = cell_pos_enc
    if fp32 is not None:
        config['meta']['fp32_position_encoding'] = fp32
    token_dict = {f't{i}': i for i in range(200)}
    return init_model_from_config(config, token_dict, 'cpu')


@pytest.mark.parametrize('cell_pos_enc', ['coord', 'rope'])
@pytest.mark.parametrize('fp32', [None, False, True])
def test_option_reaches_encoder_and_predictor(cell_pos_enc, fp32):
    encoder, predictor = _models(cell_pos_enc, fp32)
    expected = bool(fp32)
    assert parse_arch_kwargs(
        {'meta': {} if fp32 is None else {'fp32_position_encoding': fp32}}
    )['fp32_position_encoding'] is expected
    for model in (encoder.backbone, predictor.backbone):
        assert model.fp32_position_encoding is expected
        if cell_pos_enc == 'rope':
            assert model.rope.fp32 is expected


def test_phases_leave_autocast_only_with_the_option(monkeypatch):
    seen = []

    def spy(embed_dim, omega, coord):
        seen.append((_autocast_enabled('cpu'), coord.dtype))
        return get_1d_sincos_pos_embed_from_coord(embed_dim, omega, coord)
    monkeypatch.setattr(model_utils, 'get_1d_sincos_pos_embed_from_coord',
                        spy)
    coord = torch.randn(2, 8) * 300
    omega = torch.rand(8)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        default = sincos_pos_embed_from_coord(16, omega, coord)
        fp32 = sincos_pos_embed_from_coord(16, omega, coord, fp32=True)
    assert seen == [(True, torch.float32), (False, torch.float32)]
    # Without autocast casting the einsum (CPU), both agree
    torch.testing.assert_close(default, fp32)


def test_rope_phases_in_float32():
    rope = RoPE2D(head_dim=8, fp32=True)
    coords = (torch.randn(2, 5, 2) * 300).to(torch.bfloat16)
    cos, sin = rope._cos_sin(coords)
    assert cos.dtype == sin.dtype == torch.float32
    q = torch.randn(2, 1, 5, 8, dtype=torch.bfloat16)
    q_rot, k_rot = rope(q, q.clone(), coords.float())
    # q and k keep their dtype
    assert q_rot.dtype == k_rot.dtype == torch.bfloat16


@pytest.mark.skipif(_gpu_device() is None, reason='needs a CUDA or MPS device')
def test_option_restores_float32_phases_on_gpu():
    # On GPUs, bf16 autocast runs the einsum of the encoding in bf16
    device = _gpu_device()
    coord = (torch.rand(4, 33) * 2 - 1).to(device) * 500
    omega = 1.0 / (10000 ** (torch.arange(96.) / 96)).to(device)
    reference = sincos_pos_embed_from_coord(192, omega, coord)
    with torch.autocast(device, dtype=torch.bfloat16):
        default = sincos_pos_embed_from_coord(192, omega, coord)
        fp32 = sincos_pos_embed_from_coord(192, omega, coord, fp32=True)
    assert (default.float() - reference).abs().max() > 0.1
    torch.testing.assert_close(fp32, reference)
