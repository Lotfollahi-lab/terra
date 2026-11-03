"""
Tests for loading trained checkpoints outside of training.

Checkpoints are saved from DistributedDataParallel models, whose state dict
keys start with ``module.``. The inference loader (``load_checkpoint`` with
``is_training=False``) and the self-supervised finetuning loader removed
``module.`` anywhere in a key, which also renamed submodules whose names end
in ``module``: models with read-depth-aware conditioning (``rda_module``)
could not be loaded. Only the leading prefix is now removed.
"""

import pytest
import torch

from terra.training.finetune_self_supervised import _load_pretrained_checkpoint
from terra.utils.helper import init_model, load_checkpoint, strip_ddp_prefix

KWARGS = dict(
    gt_type='combined', count_encoding='value_bins', n_value_bins=20,
    cell_pos_enc='segment', device='cpu', vocab_size=200, seq_len=17,
    n_special_tokens=1, n_segments=4, n_special_values=8, enc_emb_dim=16,
    enc_depth=1, pred_emb_dim=8, pred_depth=1, num_heads=2, nz_spc=True,
    mlp_bias=True, rda_kwargs={'enabled': True})


def _model(seed):
    torch.manual_seed(seed)
    encoder, predictor = init_model(**KWARGS)
    with torch.no_grad():
        for param in list(encoder.parameters()) + list(predictor.parameters()):
            param.add_(torch.randn(param.shape))
    return encoder, predictor


@pytest.fixture
def checkpoint(tmp_path):
    encoder, predictor = _model(seed=0)
    assert any('rda_module' in key for key in encoder.state_dict())
    path = tmp_path / 'model_checkpoint.pt'
    # DDP-style keys, as saved in training
    torch.save({'encoder': {f'module.{k}': v for k, v
                            in encoder.state_dict().items()},
                'predictor': {f'module.{k}': v for k, v
                              in predictor.state_dict().items()},
                'target_encoder': {f'module.{k}': v for k, v
                                   in encoder.state_dict().items()},
                'epoch': 0}, path)
    return path, encoder, predictor


def _assert_equal(module, expected):
    expected = expected.state_dict()
    for key, value in module.state_dict().items():
        assert torch.equal(value, expected[key]), key


def test_strip_ddp_prefix_removes_only_the_leading_prefix():
    assert strip_ddp_prefix({'module.backbone.rda_module.mlp.0.weight': 1,
                             'backbone.norm.weight': 2}) == {
        'backbone.rda_module.mlp.0.weight': 1, 'backbone.norm.weight': 2}


def test_inference_loads_a_model_with_read_depth_conditioning(checkpoint):
    path, encoder, _ = checkpoint
    target_encoder, _ = _model(seed=1)
    _, _, target_encoder, _, _, _, _ = load_checkpoint(
        device='cpu', r_path=path, encoder=None, predictor=None,
        target_encoder=target_encoder, opt=None, scaler=None,
        is_training=False)
    _assert_equal(target_encoder, encoder)


def test_finetuning_loads_a_model_with_read_depth_conditioning(checkpoint):
    path, encoder, predictor = checkpoint
    new_encoder, new_predictor = _model(seed=1)
    new_target_encoder, _ = _model(seed=2)
    loaded = _load_pretrained_checkpoint(
        'cpu', path, new_encoder, new_predictor, new_target_encoder)
    for module, expected in zip(loaded, (encoder, predictor, encoder)):
        _assert_equal(module, expected)
