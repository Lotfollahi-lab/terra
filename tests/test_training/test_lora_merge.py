"""
Tests for merging LoRA adapters into a finetuned model.

``prepare_finetuned_model(use_peft=True)`` folded the adapters as
``W + lora_B @ lora_A`` without PEFT's scaling ``lora_alpha / r``, so the
encoder and predictor of the prepared model folder were wrong whenever
``peft_alpha != peft_rank`` (16x too small adapter updates with the tutorial
settings r=16, alpha=256). The target encoder, which the embeddings use, was
not affected. The merge now uses the scaling of the run's ``params.yaml``.
"""

import copy
import pickle

import pytest
import torch
import yaml

pytest.importorskip('peft')

from terra.training.finetune_self_supervised import (  # noqa: E402
    _apply_lora, merge_lora_weights, prepare_finetuned_model)
from terra.utils.helper import init_model  # noqa: E402

RANK, ALPHA = 2, 16


def _lora_models():
    torch.manual_seed(0)
    encoder, predictor = init_model(
        gt_type='combined', count_encoding='value_bins', n_value_bins=20,
        cell_pos_enc='segment', device='cpu', vocab_size=200, seq_len=17,
        n_special_tokens=1, n_segments=4, n_special_values=8,
        enc_emb_dim=16, enc_depth=1, pred_emb_dim=8, pred_depth=1,
        num_heads=2, nz_spc=True, mlp_bias=True)
    models = []
    for model in (encoder, predictor):
        model = _apply_lora(copy.deepcopy(model), peft_rank=RANK,
                            peft_alpha=ALPHA, peft_dropout=0.0)
        with torch.no_grad():
            # As after some training steps (lora_B starts at zero)
            for name, param in model.named_parameters():
                if 'lora_B' in name:
                    param.normal_(0, 0.1)
        models.append(model)
    return encoder, models


def _peft_merge(model):
    return copy.deepcopy(model).merge_and_unload().state_dict()


def _assert_close(state_dict, expected):
    assert sorted(state_dict) == sorted(expected)
    for key, value in expected.items():
        torch.testing.assert_close(state_dict[key], value)


def test_merge_matches_peft():
    _, (encoder, predictor) = _lora_models()
    for model in (encoder, predictor):
        _assert_close(merge_lora_weights(model.state_dict(), ALPHA / RANK),
                      _peft_merge(model))


@pytest.fixture
def finetuning_run(tmp_path):
    plain_encoder, (encoder, predictor) = _lora_models()
    pretrained = tmp_path / 'pretrained'
    pretrained.mkdir()
    with open(pretrained / 'model_config.yaml', 'w') as f:
        yaml.safe_dump({'meta': {}, 'data': {}}, f)
    with open(pretrained / 'token_dictionary.pkl', 'wb') as f:
        pickle.dump({'<pad>': 0}, f)
    run = tmp_path / 'run'
    run.mkdir()
    torch.save({'encoder': encoder.state_dict(),
                'predictor': predictor.state_dict(),
                'target_encoder': plain_encoder.state_dict(), 'epoch': 1},
               run / 'checkpoint_epoch_1.pt')
    return run, pretrained, encoder, predictor, plain_encoder


def test_prepared_model_holds_the_peft_merge(tmp_path, finetuning_run):
    run, pretrained, encoder, predictor, plain_encoder = finetuning_run
    with open(run / 'params.yaml', 'w') as f:
        yaml.safe_dump({'finetune': {'peft_rank': RANK, 'peft_alpha': ALPHA}},
                       f)
    output = prepare_finetuned_model(run, pretrained, tmp_path / 'prepared',
                                     use_peft=True)
    checkpoint = torch.load(f'{output}/model_checkpoint.pt')
    _assert_close(checkpoint['encoder'], _peft_merge(encoder))
    _assert_close(checkpoint['predictor'], _peft_merge(predictor))
    _assert_close(checkpoint['target_encoder'], plain_encoder.state_dict())


@pytest.mark.parametrize('params', [None, {'finetune': {'peft_rank': RANK}}])
def test_merge_needs_the_lora_settings_of_the_run(
        tmp_path, finetuning_run, params):
    run, pretrained, *_ = finetuning_run
    if params is not None:
        with open(run / 'params.yaml', 'w') as f:
            yaml.safe_dump(params, f)
    with pytest.raises(ValueError, match='peft_alpha'):
        prepare_finetuned_model(run, pretrained, tmp_path / 'prepared',
                                use_peft=True)
