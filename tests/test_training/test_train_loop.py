"""
End-to-end tests of the pretraining loop (``terra.training.train.train``),
run on CPU with a tiny model and synthetic data in a single-process gloo
group.

* Mid-epoch checkpoints: ``itr % checkpoint_freq_iter == 0`` also saved a
  checkpoint after the first iteration of every epoch (a near copy of the
  previous end-of-epoch checkpoint), whatever ``checkpoint_freq_iter``.
  Checkpoint k is now saved after k * checkpoint_freq_iter iterations.
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

# The module (terra.training re-exports the `train` function under its name)
T = importlib.import_module('terra.training.train')

SEQ_LEN_CELL, N_SEGMENTS, N_CELLS, BATCH_SIZE = 8, 4, 32, 8
SEQ_LEN_NEIGHBORHOOD = SEQ_LEN_CELL * (N_SEGMENTS - 1)
ITERATIONS_PER_EPOCH = N_CELLS // BATCH_SIZE


def _dataset() -> Dataset:
    rng = torch.Generator().manual_seed(4)
    rows = {'gene_tokens': [], 'gene_expr': [], 'n_nonzero_tokens': [],
            'batch_value': [], 'cell_id': []}
    for cell in range(N_CELLS):
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
        args[section].update(values)
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

    def run(**overrides):
        folder = tmp_path / 'run'
        folder.mkdir(exist_ok=True)
        token_dict_path = str(tmp_path / 'token_dictionary.pkl')
        with open(token_dict_path, 'wb') as f:
            pickle.dump({f't{i}': i for i in range(700)}, f)

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
            if isinstance(out, torch.nn.Module):
                result['copies'].append(out)
            return out
        monkeypatch.setattr(T.copy, 'deepcopy', record_deepcopy)

        T.train(_args(folder, token_dict_path, overrides),
                train_dataset=_dataset(), save_folder_path=str(folder),
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
