"""
End-to-end tests of the pretraining loop (``terra.training.train.train``),
run on CPU with a tiny model and synthetic data in a single-process gloo
group.

* Mid-epoch checkpoints: ``itr % checkpoint_freq_iter == 0`` also saved a
  checkpoint after the first iteration of every epoch (a near copy of the
  previous end-of-epoch checkpoint), whatever ``checkpoint_freq_iter``.
  Checkpoint k is now saved after k * checkpoint_freq_iter iterations.
* Gradient norms: the gradient statistics were read after
  ``optimizer.zero_grad``, and without clipping (e.g. during warmup) the
  logged global norms were 0. They are now read before the gradients are
  zeroed.
* Fractional warmup: clipping was skipped for the first warmup * ipe
  iterations of every epoch, not only of the run.
* ``ipe_scale < 1``: training stopped with StopIteration once the EMA
  momentum schedule ran out.
* The target encoder (a deep copy of the context encoder) stayed in training
  mode, so with RoPE rotation augmentation its targets were randomly
  rotated.
* ``add_cls``: the cell-neighborhood branch compared the tokenizer type with
  'cell_neighborhood' instead of 'cell_neigh', so the option was silently
  ignored for cell-neighborhood models; with the cell-graph tokenizer it
  failed in the data loader with a KeyError. Both now ask for the CLS token
  columns, which the TERRA tokenizers do not write.
* Cycle consistency: in batches with padded special tokens the swapped
  labels cannot reach the encoder (they are carried by the special tokens),
  but every cell with a new label counted as changed, so a second encoder
  pass ran for a zero loss. Rank models had no check that the swapped
  labels, written into the special token ids, stay inside the token
  dictionary.
* Non-finite values: the loss was checked for NaN only after the optimizer
  step, on each rank separately, and a non-finite state could be saved over
  the last checkpoint. Training now stops before the step at the first
  non-finite loss or gradient norm on any rank, and never saves a non-finite
  model or optimizer state.
* Resume: the checkpoint held no random number generator states, so a
  resumed run drew the masks and DataLoader worker seeds of epoch 0 again.
  Checkpoints now hold the states of every rank, and a run resumed from the
  end of an epoch continues exactly like the uninterrupted run.
"""

import copy
import importlib
import os
import pickle
import sys
import types

import pytest
import torch
import torch.distributed as dist
from datasets import Dataset
from torch.nn.parallel import DistributedDataParallel

from terra.models.multimask import EncoderMultiMaskWrapper

# The module (terra.training re-exports the `train` function under its name)
T = importlib.import_module('terra.training.train')

SEQ_LEN_CELL, N_SEGMENTS, N_CELLS, BATCH_SIZE = 8, 4, 32, 8
SEQ_LEN_NEIGHBORHOOD = SEQ_LEN_CELL * (N_SEGMENTS - 1)
ITERATIONS_PER_EPOCH = N_CELLS // BATCH_SIZE


def _dataset(one_gene_cells=()) -> Dataset:
    # Cells in `one_gene_cells` have a single gene, in their own segment
    rng = torch.Generator().manual_seed(4)
    rows = {'gene_tokens': [], 'gene_expr': [], 'n_nonzero_tokens': [],
            'batch_value': [], 'cell_id': []}
    for cell in range(N_CELLS):
        tokens, values = [], []
        for segment in range(N_SEGMENTS):
            n = int(torch.randint(3, SEQ_LEN_CELL + 1, (1,), generator=rng))
            if cell in one_gene_cells:
                n = int(segment == 0)
            tokens += (torch.randperm(500, generator=rng)[:n] + 107).tolist()
            tokens += [0] * (SEQ_LEN_CELL - n)
            values += sorted(torch.randint(1, 50, (n,), generator=rng)
                             .float().tolist(), reverse=True)
            values += [0.0] * (SEQ_LEN_CELL - n)
        rows['gene_tokens'].append(tokens)
        rows['gene_expr'].append(values)
        rows['n_nonzero_tokens'].append(sum(t != 0 for t in tokens))
        rows['batch_value'].append([5])
        rows['cell_id'].append(f'cell_{cell}')
    dataset = Dataset.from_dict(rows)
    dataset.set_format(type='torch', columns=[c for c in rows if c != 'cell_id'],
                       output_all_columns=True)
    return dataset


def _args(folder, token_dict_path, overrides):
    args = {
        'data': dict(
            dataset_name='synthetic', token_dict_folder_path=token_dict_path,
            tokenizer_type='cell_graph', seq_len_cell=SEQ_LEN_CELL,
            seq_len_neighborhood=SEQ_LEN_NEIGHBORHOOD, n_segments=N_SEGMENTS,
            sampling_strategy=None, batch_size=BATCH_SIZE, num_workers=0,
            pin_memory=False, use_sampler=True, n_special_values=16,
            nz_spc=True, precomputed_n_nonzero_tokens=None),
        'meta': dict(
            add_cls=False, gt_type='combined', count_encoding='value_bins',
            n_value_bins=20, cell_pos_enc='segment', enc_depth=1,
            enc_emb_dim=16, pred_depth=1, pred_emb_dim=8, num_heads=2,
            mlp_ratio=4, special_tokens=['batch'], use_bfloat16=False,
            use_flash_attention=True, use_layer_norm=True, new_spc=True,
            predict_gene=True, pos_learnable=False),
        'mask': dict(
            n_contexts=1, n_targets=1, block_masking=True, cell_masking=False,
            context_mask_size=200, target_mask_size=50,
            per_block_mask_ratio=0.6, sample_segments=True,
            targets_list=None),
        'optimization': dict(
            warmup=1, epochs=2, ema=[0.9995, 1.0], start_lr=1e-6, lr=6e-5,
            final_lr=1e-6, weight_decay=0.04, final_weight_decay=0.4,
            ipe_scale=1.25, clip_grad=2.0),
        'state': dict(
            log_freq=1, checkpoint_freq=1, write_tag='terra',
            folder_path=str(folder), load_checkpoint=False,
            read_checkpoint=None, checkpoint_freq_iter=10**9),
    }
    for section, values in overrides.items():
        args.setdefault(section, {}).update(values)
    return args


@pytest.fixture
def run_train(tmp_path, monkeypatch):
    """
    Return a function that runs ``train`` with config overrides and reports
    the logged payloads, the steps with gradient clipping and the saved
    checkpoints.
    """
    # train() sets process-wide options; restore them afterwards
    monkeypatch.setattr(T.mp, 'set_start_method', lambda *a, **k: None)
    precision = torch.get_float32_matmul_precision()
    backend_flags = (torch.backends.cudnn.deterministic,
                     torch.backends.cudnn.benchmark,
                     torch.backends.cuda.matmul.allow_tf32,
                     torch.backends.cudnn.allow_tf32)
    # DDP on CPU: no device ids
    monkeypatch.setattr(T, 'DistributedDataParallel', lambda module, **kw: (
        DistributedDataParallel(module, **{
            k: v for k, v in kw.items()
            if k not in ('device_ids', 'output_device')})))
    dist.init_process_group('gloo', init_method=f"file://{tmp_path / 'pg'}",
                            rank=0, world_size=1)

    def run(vocab_size=700, dataset=None, **overrides):
        folder = tmp_path / 'run'
        folder.mkdir(exist_ok=True)
        token_dict_path = str(tmp_path / 'token_dictionary.pkl')
        with open(token_dict_path, 'wb') as f:
            pickle.dump({f't{i}': i for i in range(vocab_size)}, f)

        result = {'logged': [], 'clip_steps': [], 'saves': [], 'copies': []}
        monkeypatch.setitem(sys.modules, 'wandb', types.SimpleNamespace(
            log=lambda payload, **kw: result['logged'].append(payload)))

        clip_grad_norm_ = torch.nn.utils.clip_grad_norm_

        def clip(*args, **kwargs):
            # Clipping runs before the step is logged (log_freq 1)
            result['clip_steps'].append(len(result['logged']))
            return clip_grad_norm_(*args, **kwargs)
        monkeypatch.setattr(torch.nn.utils, 'clip_grad_norm_', clip)

        save = torch.save

        def record_save(obj, path, *args, **kwargs):
            name = os.path.basename(str(path))
            if name.endswith('.pth.tar') and 'latest' not in name:
                state = obj['opt']['state']
                steps = int(next(iter(state.values()))['step']) if state else 0
                result['saves'].append((name, steps))
            return save(obj, path, *args, **kwargs)
        monkeypatch.setattr(torch, 'save', record_save)

        deepcopy = copy.deepcopy

        def record_deepcopy(obj, *args, **kwargs):
            out = deepcopy(obj, *args, **kwargs)
            if isinstance(out, EncoderMultiMaskWrapper):
                result['copies'].append(out)
            return out
        monkeypatch.setattr(T.copy, 'deepcopy', record_deepcopy)

        T.train(_args(folder, token_dict_path, overrides),
                train_dataset=_dataset() if dataset is None else dataset,
                save_folder_path=str(folder),
                LOCAL_RANK=0, WORLD_RANK=0)
        return result

    yield run
    dist.destroy_process_group()
    torch.set_float32_matmul_precision(precision)
    (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark,
     torch.backends.cuda.matmul.allow_tf32,
     torch.backends.cudnn.allow_tf32) = backend_flags


def test_checkpoints_only_at_epoch_ends_by_default(run_train):
    result = run_train()
    assert len(result['logged']) == 2 * ITERATIONS_PER_EPOCH
    # No checkpoint after the first iteration of each epoch
    assert result['saves'] == [('terra-ep0.pth.tar', 4),
                               ('terra-ep1.pth.tar', 8)]


def test_mid_epoch_checkpoints_every_checkpoint_freq_iter(run_train):
    result = run_train(state={'checkpoint_freq_iter': 2})
    # Checkpoint k after k * 2 iterations of the epoch; the one after the
    # last iteration is the end-of-epoch checkpoint
    assert result['saves'] == [('terra-ep0_1.pth.tar', 2),
                               ('terra-ep0.pth.tar', 4),
                               ('terra-ep1_1.pth.tar', 6),
                               ('terra-ep1.pth.tar', 8)]


def test_gradient_norms_are_logged_during_warmup(run_train):
    result = run_train()
    # Warmup 1: no clipping in epoch 0, clipping in epoch 1
    assert result['clip_steps'] == [step for step in range(4, 8)
                                    for _ in ('encoder', 'predictor')]
    for payload in result['logged']:
        assert payload['global_norm_enc'] > 0
        assert payload['global_norm_pred'] > 0


@pytest.mark.parametrize('warmup, first_clipped_step', [(0.5, 3), (0, 1)])
def test_fractional_warmup_clips_every_step_after_warmup(
        run_train, warmup, first_clipped_step):
    result = run_train(optimization={'warmup': warmup})
    # Clipping (encoder and predictor) from the first step after
    # warmup * ITERATIONS_PER_EPOCH iterations on, in both epochs
    assert result['clip_steps'] == [
        step for step in range(first_clipped_step, 2 * ITERATIONS_PER_EPOCH)
        for _ in ('encoder', 'predictor')]


def test_ipe_scale_below_one_trains_every_step(run_train):
    result = run_train(optimization={'ipe_scale': 0.5})
    assert len(result['logged']) == 2 * ITERATIONS_PER_EPOCH
    # The learning rate stays at final_lr after the end of the schedule
    for payload in result['logged'][ITERATIONS_PER_EPOCH:]:
        assert payload['lr'] == pytest.approx(1e-6)


def test_target_encoder_is_in_eval_mode(run_train):
    result = run_train(optimization={'epochs': 1})
    (target_encoder,) = result['copies']
    assert not target_encoder.training
    assert not any(m.training for m in target_encoder.modules())


@pytest.mark.parametrize('tokenizer_type, cls_tokens', [
    ('cell_neigh', ['cls_0', 'cls_1']),
    ('cell_graph', [f'cls_{i}' for i in range(N_SEGMENTS)]),
])
def test_add_cls_adds_cls_tokens_for_both_tokenizers(
        run_train, tokenizer_type, cls_tokens):
    with pytest.raises(ValueError, match='add_cls') as error:
        run_train(data={'tokenizer_type': tokenizer_type},
                  meta={'add_cls': True})
    assert str(error.value).count('_token') == len(cls_tokens)
    for token in cls_tokens:
        assert f"'{token}_token', '{token}_value'" in str(error.value)


# Cycle consistency with the batch label of the `batch` special token
CYCLE = {'cycle_consistency': {
    'enabled': True, 'lambda_cycle': 1.0, 'swap_fraction': 1.0,
    'keys': ['batch_value'], 'n_classes': [8], 'offsets': [0]}}
# Token id of special value 0 in rank models (homo sapiens layout)
RANK_SPECIAL_VALUE_OFFSET = 21957


@pytest.mark.parametrize('special_token_pad_ratio', [0.0, 1.0])
def test_cycle_consistency_skips_batches_with_padded_special_tokens(
        run_train, special_token_pad_ratio):
    result = run_train(
        optimization={'epochs': 1},
        mask={'special_token_pad_ratio': special_token_pad_ratio},
        special_token_correction=CYCLE)
    cycle = [payload for payload in result['logged']
             if 'cycle/loss' in payload]
    if special_token_pad_ratio:
        # No special token to carry the swapped labels: no second pass
        assert cycle == []
    else:
        assert len(cycle) == ITERATIONS_PER_EPOCH
        for payload in cycle:
            assert payload['cycle/n_changed'] > 0
            assert payload['cycle/loss'] > 0


@pytest.mark.parametrize('n_classes', [8, 32])
def test_cycle_consistency_checks_rank_labels_against_token_dictionary(
        run_train, n_classes):
    # Special values 0-17 have token ids (batch_value 5: id offset + 5)
    vocab_size = RANK_SPECIAL_VALUE_OFFSET + 18
    cycle = copy.deepcopy(CYCLE)
    cycle['cycle_consistency']['n_classes'] = [n_classes]
    if n_classes > 18:
        with pytest.raises(ValueError, match='special value token ids'):
            run_train(vocab_size=vocab_size, meta={'gt_type': 'rank'},
                      special_token_correction=cycle)
    else:
        result = run_train(vocab_size=vocab_size, optimization={'epochs': 1},
                           meta={'gt_type': 'rank'},
                           special_token_correction=cycle)
        assert any('cycle/loss' in payload for payload in result['logged'])


# Batches in dataset order with two one-gene cells each, no metatoken and
# masks padded to the longest cell: with mask ratio 0.6 the one gene of these
# cells is a target, so their context is only padding (fully masked
# attention rows in the context encoder)
EMPTY_CONTEXT = dict(
    dataset=_dataset(one_gene_cells=range(0, N_CELLS, 4)),
    data={'use_sampler': False},
    meta={'special_tokens': [], 'use_flash_attention': False},
    mask={'sample_segments': False, 'pad_masks_to_batch_max': True})


def _saved_checkpoints(tmp_path):
    return sorted(p.name for p in (tmp_path / 'run').glob('*.pth.tar*'))


def test_nonfinite_loss_stops_training_before_the_optimizer_step(
        run_train, tmp_path):
    # The eager attention gives NaN for the fully masked rows
    with pytest.raises(FloatingPointError,
                       match='Non-finite loss or gradient norm at epoch 0, '
                             'iteration 0'):
        run_train(**EMPTY_CONTEXT)
    assert _saved_checkpoints(tmp_path) == []


def test_zero_fully_masked_attention_rows_keeps_training_finite(run_train):
    config = copy.deepcopy(EMPTY_CONTEXT)
    config['meta']['zero_fully_masked_attention_rows'] = True
    result = run_train(**config)
    assert len(result['logged']) == 2 * ITERATIONS_PER_EPOCH
    assert all(torch.isfinite(torch.tensor(payload['loss']))
               for payload in result['logged'])


def test_nonfinite_state_is_not_saved(run_train, tmp_path, monkeypatch):
    # Finite loss and gradients, but the last step of epoch 0 leaves a NaN
    # weight: the end-of-epoch checkpoint is not written
    step = torch.optim.AdamW.step
    n_steps = []

    def corrupting_step(self, *args, **kwargs):
        out = step(self, *args, **kwargs)
        n_steps.append(1)
        if len(n_steps) == ITERATIONS_PER_EPOCH:
            with torch.no_grad():
                self.param_groups[0]['params'][0].fill_(float('nan'))
        return out
    monkeypatch.setattr(torch.optim.AdamW, 'step', corrupting_step)

    with pytest.raises(FloatingPointError,
                       match='Non-finite model or optimizer state at epoch 0'):
        run_train()
    assert _saved_checkpoints(tmp_path) == []


def _final_state(tmp_path):
    return torch.load(tmp_path / 'run' / 'terra-latest.pth.tar',
                      weights_only=False)


def _same_training_state(a, b):
    """Bit-identical models and optimizer state."""
    tensors = []
    for key in ('encoder', 'predictor', 'target_encoder'):
        assert a[key].keys() == b[key].keys()
        tensors += [(a[key][name], b[key][name]) for name in a[key]]
    for state_a, state_b in zip(a['opt']['state'].values(),
                                b['opt']['state'].values()):
        tensors += [(state_a[name], state_b[name]) for name in state_a]
    return all(torch.equal(x, y) for x, y in tensors)


RESUME = {'state': {'load_checkpoint': True,
                    'read_checkpoint': 'terra-ep0.pth.tar'}}


def test_resumed_run_continues_the_uninterrupted_run(run_train, tmp_path):
    run_train()
    uninterrupted = _final_state(tmp_path)
    rng_states = uninterrupted['rng_state']
    assert len(rng_states) == 1
    assert set(rng_states[0]) == {'python', 'numpy', 'torch', 'cuda'}

    # Resume from the end of epoch 0: epoch 1 draws the same masks
    result = run_train(**RESUME)
    assert len(result['logged']) == ITERATIONS_PER_EPOCH
    assert _same_training_state(_final_state(tmp_path), uninterrupted)


def test_resume_from_checkpoint_without_rng_state_warns(run_train, tmp_path):
    run_train()
    uninterrupted = _final_state(tmp_path)
    checkpoint = torch.load(tmp_path / 'run' / 'terra-ep0.pth.tar',
                            weights_only=False)
    del checkpoint['rng_state']
    torch.save(checkpoint, tmp_path / 'run' / 'terra-old.pth.tar')

    with pytest.warns(UserWarning, match='no random number generator'):
        run_train(state={'load_checkpoint': True,
                         'read_checkpoint': 'terra-old.pth.tar'})
    # Same state after epoch 0, but different masks in epoch 1
    assert not _same_training_state(_final_state(tmp_path), uninterrupted)


def test_ranks_resuming_from_different_epochs_raise(
        run_train, tmp_path, monkeypatch):
    run_train()
    # Another rank resumed from epoch 2
    monkeypatch.setattr(T, '_all_gather_objects', lambda obj: (
        [obj, (2, None)] if isinstance(obj, tuple) else [obj]))
    with pytest.raises(RuntimeError, match='different epochs'):
        run_train(**RESUME)

