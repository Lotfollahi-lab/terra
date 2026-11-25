"""
Time what epoch index lists (`data.epoch_index_lists`) change in the data
path of training:

1. the sampler's index computation at the start of every epoch, for the
   default order and for index lists (all rows, a random quarter, a fixed
   eighth of the rows presented eight times);
2. reading rows through `CellGraphDataset` in a contiguous block, in a full
   shuffle, and in the order the sampler gives for a fixed eighth of the
   rows (the rows a selection run reads).

Usage, on a tokenized dataset with the model config that trains on it (reads
the training split as `train` does)::

    python scripts/benchmark_epoch_index_lists.py --config <model>.yaml \
        [--n_reads 20000] [--num_workers 0]

or on a synthetic compact dataset (local check)::

    python scripts/benchmark_epoch_index_lists.py --synthetic 20000

The patterns of step 2 run one after another in one process, so later ones
can profit from the operating system's page cache: compare them with care,
or run the script once per pattern (`--patterns`).
"""

import argparse
import os
import pickle
import tempfile
import time

import numpy as np
import torch

from terra.datasets.cell_datasets import init_cell_dataset
from terra.datasets.dataloaders import (CustomDistributedLengthGroupedSampler,
                                        EpochIndexLists)


class _Lengths(torch.utils.data.Dataset):
    def __init__(self, lengths):
        self.n_nz_tokens = lengths

    def __len__(self):
        return len(self.n_nz_tokens)


def _time_sampler(lengths, batch_size, world_size, mega, tmp_dir):
    n = len(lengths)
    rng = np.random.default_rng(0)
    eighth = rng.choice(n, n // 8, replace=False)
    designs = {
        'default': None,
        'all rows': np.arange(n)[None],
        'random quarter': rng.choice(n, n // 4, replace=False)[None],
        'fixed eighth x 8': np.tile(eighth, 8)[None],
    }
    dataset = _Lengths(lengths)
    print(f'Sampler, {n} rows, batch {batch_size} x {world_size} ranks:')
    for name, array in designs.items():
        lists = None
        if array is not None:
            path = os.path.join(tmp_dir, f'{name.replace(" ", "_")}.npy')
            np.save(path, array)
            lists = EpochIndexLists(path, n_rows=n, n_epochs=1)
        sampler = CustomDistributedLengthGroupedSampler(
            dataset, batch_size=batch_size, num_replicas=world_size, rank=0,
            mega_batch_mult_max=mega, epoch_index_lists=lists)
        t0 = time.perf_counter()
        n_items = len(list(sampler))
        print(f'  {name:18s} {time.perf_counter() - t0:7.2f} s per epoch '
              f'({n_items} items on this rank)')


def _time_reads(cell_dataset, n_reads, num_workers, patterns, tmp_dir,
                batch_size):
    n = len(cell_dataset)
    n_reads = min(n_reads, n)
    rng = np.random.default_rng(1)
    orders = {}
    if 'contiguous' in patterns:
        start = int(rng.integers(0, n - n_reads + 1))
        orders['contiguous'] = np.arange(start, start + n_reads)
    if 'shuffle' in patterns:
        orders['shuffle'] = rng.permutation(n)[:n_reads]
    if 'eighth' in patterns:
        eighth = rng.choice(n, max(1, n // 8), replace=False)
        path = os.path.join(tmp_dir, 'eighth.npy')
        np.save(path, np.tile(eighth, 8)[None])
        sampler = CustomDistributedLengthGroupedSampler(
            cell_dataset, batch_size=batch_size, num_replicas=1, rank=0,
            epoch_index_lists=EpochIndexLists(path, n_rows=n, n_epochs=1))
        orders['eighth'] = np.asarray(list(sampler))[:n_reads]
    print(f'Reads through CellGraphDataset ({n_reads} items, '
          f'{num_workers} workers):')
    for name, order in orders.items():
        loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(cell_dataset, order.tolist()),
            batch_size=batch_size, num_workers=num_workers)
        t0 = time.perf_counter()
        for _ in loader:
            pass
        dt = time.perf_counter() - t0
        print(f'  {name:12s} {n_reads / dt:9.0f} items/s')


def _synthetic(n_rows, tmp_dir, n_segments=11, seq_len_cell=256):
    from datasets import Dataset
    from terra.datasets.compact import compact_features
    rng = np.random.default_rng(2)
    mis = n_segments * seq_len_cell
    features = compact_features(np.int16, mis, include_special_tokens=False,
                                add_neigh_cell_ids=False)

    def rows():
        for i in range(n_rows):
            seg = rng.integers(20, seq_len_cell + 1, n_segments)
            k = int(seg.sum())
            yield dict(rel_x_coord=rng.normal(size=n_segments).astype('f4'),
                       rel_y_coord=rng.normal(size=n_segments).astype('f4'),
                       cell_id=f'0_b0_{i}',
                       gene_tokens=rng.integers(2, 20000, k).astype('i2'),
                       gene_expr=rng.random(k).astype('f4'),
                       n_nonzero_tokens=np.int32(k),
                       segment_lengths=seg.astype('i2'),
                       model_input_size=np.int32(mis))
    path = os.path.join(tmp_dir, 'synthetic.dataset')
    Dataset.from_generator(rows, features=features,
                           cache_dir=os.path.join(tmp_dir, 'cache')
                           ).save_to_disk(path)
    from datasets import load_from_disk
    ds = load_from_disk(path)
    ds.set_format(type='torch', columns=[c for c in ds.column_names
                                         if c != 'cell_id'])
    return init_cell_dataset(
        tokenizer_type='cell_graph', dataset=ds, gt_type='combined',
        cell_pos_enc='coord', vocab_size=20000, seq_len_cell=seq_len_cell,
        seq_len_neighborhood=seq_len_cell * (n_segments - 1))


def _from_config(config_path):
    from terra.datasets.utils import prepare_dataset
    from terra.utils.config import create_params_from_YAML_wandb_config
    import logging
    args = create_params_from_YAML_wandb_config(
        config_path, logging.getLogger(__name__))
    train_dataset, _, _ = prepare_dataset(args, train_mode=True)
    with open(args['data']['token_dict_folder_path'], 'rb') as f:
        vocab_size = len(pickle.load(f))
    data, meta = args['data'], args['meta']
    return init_cell_dataset(
        tokenizer_type=data['tokenizer_type'], dataset=train_dataset,
        gt_type=meta['gt_type'], cell_pos_enc=meta.get('cell_pos_enc',
                                                       'segment'),
        vocab_size=vocab_size, seq_len_cell=data['seq_len_cell'],
        seq_len_neighborhood=data['seq_len_neighborhood'],
        special_tokens=meta.get('special_tokens', []),
        sampling_strategy=data.get('sampling_strategy'),
        nz_spc=data.get('nz_spc', False),
        truncate_neighbors=data.get('truncate_neighbors', False),
        tokenized_seq_len_cell=data.get('tokenized_seq_len_cell')), data


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--config', help='model config of a training run')
    source.add_argument('--synthetic', type=int,
                        help='rows of a synthetic compact dataset')
    parser.add_argument('--n_reads', type=int, default=20000)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--world_size', type=int, default=4)
    parser.add_argument('--sampler_rows', type=int, default=None,
                        help='rows for the sampler timing (default: the '
                             'dataset; e.g. 4600000 for the selection proxy)')
    parser.add_argument('--patterns', nargs='+',
                        default=['shuffle', 'contiguous', 'eighth'])
    opts = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp_dir:
        if opts.synthetic:
            cell_dataset = _synthetic(opts.synthetic, tmp_dir)
            mega = 1000
        else:
            cell_dataset, data = _from_config(opts.config)
            mega = data.get('mega_batch_mult_max', 1000)
        lengths = list(cell_dataset.n_nz_tokens)
        if opts.sampler_rows:
            lengths = list(np.resize(np.asarray(lengths), opts.sampler_rows))
        _time_sampler(lengths, opts.batch_size, opts.world_size, mega,
                      tmp_dir)
        _time_reads(cell_dataset, opts.n_reads, opts.num_workers,
                    opts.patterns, tmp_dir, opts.batch_size)


if __name__ == '__main__':
    main()
