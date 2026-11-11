import json
import logging
import pickle
import random
import re
import requests
from typing import Literal

import datasets
import numpy as np
import pyarrow.compute as pc
from datasets import load_from_disk


logger = logging.getLogger(__name__)


def get_ensembl_ids(gene_names: list[str],
                    species: Literal['homo_sapiens',
                                     'mus_musculus'],
                    ) -> dict:
    """
    Get gene Ensembl IDs based on gene names via Ensembl REST API.

    Parameters
    ----------
    gene_names:
        List of gene names.
    species:
        Species for which to retrieve Ensembl IDs.

    Returns
    ----------
    ensembl_ids:
        Dictionary where keys are gene names and values are Ensembl IDs.
    """
    server = 'https://rest.ensembl.org'
    endpoint = f'/lookup/symbol/{species}'
    headers = {
        'Content-Type': 'application/json', 'Accept': 'application/json'}

    data = {'symbols': gene_names}
    response = requests.post(f'{server}{endpoint}',
                             headers=headers,
                             data=json.dumps(data))
    
    if response.ok:
        ensembl_ids = {}
        for key, value in response.json().items():
            ensembl_ids[key] = value['id']
        if len(ensembl_ids.keys()) != len(gene_names):
            missing_genes = [
                gene for gene in gene_names if gene not in ensembl_ids.keys()]
            logger.warning(
                f'Could not find Ensembl IDs for genes: {missing_genes}.')
        return ensembl_ids
    else:
        response.raise_for_status()


# Format tag of the precomputed split files written by `save_precomputed_split`
PRECOMPUTED_SPLIT_FORMAT = 'terra_precomputed_split_v1'
# Number of evenly spaced rows whose cell ids identify the split's dataset
N_SPLIT_IDENTITY_ROWS = 1000


def _batch_ids_from_cell_ids(cell_ids: np.ndarray) -> np.ndarray:
    """Map cell ids '<part0>_<part1>_...' to batch ids '<part0>_<part1>'."""
    p1 = np.char.partition(cell_ids, '_')
    p2 = np.char.partition(p1[:, 2], '_')
    return np.char.add(np.char.add(p1[:, 0], '_'), p2[:, 0])


def _identity_positions(n_rows: int) -> list[int]:
    if n_rows == 0:
        return []
    n = min(N_SPLIT_IDENTITY_ROWS, n_rows)
    return sorted(set(
        np.linspace(0, n_rows - 1, num=n, dtype=np.int64).tolist()))


def _cell_ids_at(ds: datasets.Dataset, positions: list[int]) -> list[str]:
    return list(ds.select_columns(['cell_id']).select(positions)['cell_id'])


def save_precomputed_split(dataset: datasets.Dataset,
                           indices: list,
                           file_path: str,
                           excluded_batch_ids: list[str] | None = None):
    """
    Save row indices of a tokenized dataset as a precomputed split.

    Besides the indices, the file records the dataset's number of rows and the
    cell ids of evenly spaced rows, so that `prepare_dataset` refuses the split
    when it is applied to another tokenization (e.g. after the corpus was
    re-tokenized and its row order changed).

    Parameters
    ----------
    dataset:
        The tokenized dataset (with its `cell_id` column) the indices refer to.
    indices:
        Row indices of the split (`data.precomputed_split`), or one list of
        row indices per epoch (`data.precomputed_epoch_splits`).
    file_path:
        Path of the pickle file to write.
    excluded_batch_ids:
        Batch ids whose cells the split was built to exclude (e.g. the
        held-out batches of a training split). Training then trusts the
        record instead of scanning every cell id at start-up when the
        configured test and validation batches are among them.
    """
    positions = _identity_positions(len(dataset))
    record = {
        'format': PRECOMPUTED_SPLIT_FORMAT,
        'n_rows': len(dataset),
        'identity_positions': positions,
        'identity_cell_ids': _cell_ids_at(dataset, positions),
        'excluded_batch_ids': (None if excluded_batch_ids is None
                               else sorted(excluded_batch_ids)),
        'indices': indices,
    }
    with open(file_path, 'wb') as f:
        pickle.dump(record, f)


def load_precomputed_split(file_path: str, dataset: datasets.Dataset):
    """
    Load the row indices of a precomputed split for `dataset`.

    Files written by `save_precomputed_split` are checked against the dataset
    (number of rows and the recorded cell ids) and a mismatch raises. A bare
    pickled list of row numbers (the legacy format) cannot be checked and is
    used as it is, with a warning.

    Parameters
    ----------
    file_path:
        Path of the pickle file.
    dataset:
        The tokenized dataset the split is applied to, with its `cell_id`
        column.

    Returns
    ----------
    indices:
        The row indices (or one list of row indices per epoch).
    """
    return _read_precomputed_split(file_path, dataset)[0]


def _read_precomputed_split(file_path: str, dataset: datasets.Dataset):
    """The indices of a precomputed split and its record (None if legacy)."""
    with open(file_path, 'rb') as f:
        split = pickle.load(f)
    if (isinstance(split, dict)
            and split.get('format') == PRECOMPUTED_SPLIT_FORMAT):
        if split['n_rows'] != len(dataset):
            raise ValueError(
                f"The precomputed split '{file_path}' was made for a dataset "
                f"with {split['n_rows']} rows, but the dataset has "
                f"{len(dataset)} rows. Rebuild the split from this dataset.")
        if 'cell_id' in dataset.column_names:
            cell_ids = _cell_ids_at(dataset, split['identity_positions'])
            n_diff = sum(a != b for a, b in
                         zip(cell_ids, split['identity_cell_ids']))
            if n_diff:
                raise ValueError(
                    f"The precomputed split '{file_path}' was made for "
                    f"another tokenization of the data: {n_diff} of "
                    f"{len(cell_ids)} recorded rows hold other cells (e.g. "
                    "the dataset was re-tokenized and its row order "
                    "changed). Rebuild the split from this dataset.")
        return split['indices'], split
    logger.warning(
        f"The precomputed split '{file_path}' is a bare list of row numbers, "
        "which cannot be checked against the dataset: if the dataset was "
        "re-tokenized since, its rows may hold other cells. Write splits with "
        "terra.datasets.utils.save_precomputed_split to have them checked.")
    return split, None


def _heldout_row_mask(ds: datasets.Dataset,
                      heldout: set,
                      chunk_size: int = 1_000_000) -> np.ndarray:
    """Rows of `ds` whose batch id (see `_batch_ids_from_cell_ids`) is held out."""
    if ds._indices is None:
        # Vectorized on the Arrow column: batch ids have one '_', and a cell
        # id belongs to batch b if it is b or starts with b + '_'
        ids = sorted(b for b in heldout if b.count('_') == 1)
        if not ids:
            return np.zeros(len(ds), dtype=bool)
        pattern = '^(?:' + '|'.join(re.escape(b) for b in ids) + ')(?:_|$)'
        matches = pc.match_substring_regex(ds.data.column('cell_id'), pattern)
        return np.asarray(pc.fill_null(matches, False).to_numpy(
            zero_copy_only=False), dtype=bool)
    heldout_arr = np.array(sorted(heldout), dtype='U')
    cell_id_ds = ds.select_columns(['cell_id'])
    mask = np.zeros(len(ds), dtype=bool)
    for start in range(0, len(ds), chunk_size):
        stop = min(start + chunk_size, len(ds))
        cell_ids = np.asarray(cell_id_ds[start:stop]['cell_id'], dtype='U')
        mask[start:stop] = np.isin(_batch_ids_from_cell_ids(cell_ids),
                                   heldout_arr)
    return mask


def _check_split_excludes_heldout(args: dict,
                                  ds: datasets.Dataset,
                                  index_lists: list,
                                  file_path: str,
                                  record: dict | None = None):
    """
    Raise when a precomputed training split selects cells of the test or
    validation batches (`data.test_batch_ids` / `data.val_batch_ids`).
    A split whose record lists these batches as excluded is not scanned.
    """
    heldout = (set(args['data'].get('test_batch_ids') or [])
               | set(args['data'].get('val_batch_ids') or []))
    if not heldout or args['data'].get('allow_heldout_in_split', False):
        return
    if 'cell_id' not in ds.column_names:
        logger.warning(
            f"Cannot check that the precomputed split '{file_path}' excludes "
            "the test and validation batches: the dataset has no cell_id "
            "column.")
        return
    excluded = (record or {}).get('excluded_batch_ids')
    if excluded is not None and heldout <= set(excluded):
        logger.info(
            f"The precomputed split '{file_path}' records that it excludes "
            "the test and validation batches.")
        return
    heldout_mask = _heldout_row_mask(ds, heldout)
    for idx in index_lists:
        n_heldout = int(heldout_mask[np.asarray(idx, dtype=np.int64)].sum())
        if n_heldout:
            raise ValueError(
                f"{n_heldout} of the {len(idx)} rows that the precomputed "
                f"split '{file_path}' selects for training belong to test or "
                "validation batches (data.test_batch_ids / "
                "data.val_batch_ids). The split does not fit this dataset or "
                "these batch lists: rebuild it from this dataset. If training "
                "is meant to include these batches, remove them from the "
                "batch lists or set data.allow_heldout_in_split: true.")


def prepare_dataset(args: dict, train_mode: bool = False):
    data_path = args['data']['tokenized_data_folder_path']
    ds = load_from_disk(data_path)

    # 1) Remove obviously unneeded columns but KEEP cell_id for now
    fields_to_remove = [
        'cell_degrees','batch_value_token','gene_panel_value_token',
        'assay_value_token','species_value_token','tissue_value_token',
        'cls_tokens','cell_total_counts','cell_n_probed_genes'
    ]
    # Modes that don't need rel_x_coord / rel_y_coord at the per-cell
    # level: 'segment' (uses sincos by rank only), 'none' (no spatial
    # encoding at all), and the legacy case where the key is missing.
    if (
        'cell_pos_enc' not in args['meta']
        or args['meta']['cell_pos_enc'] in ('segment', 'none')
    ):
        fields_to_remove += ['rel_x_coord','rel_y_coord']

    existing = [c for c in fields_to_remove if c in ds.column_names]
    if existing:
        ds = ds.remove_columns(existing)

    # Early outs with precomputed splits
    if args['data'].get('precomputed_epoch_splits'):
        file_path = args['data']['precomputed_epoch_splits']
        epoch_indices, record = _read_precomputed_split(file_path, ds)
        if train_mode:
            _check_split_excludes_heldout(args, ds, epoch_indices, file_path,
                                          record)
        splits = [ds.select(idx) for idx in epoch_indices]
        # Drop cell_id if training
        if train_mode and 'cell_id' in splits[0].column_names:
            splits = [s.remove_columns(['cell_id']) for s in splits]
        # Set minimal torch format on each split
        for i in range(len(splits)):
            cols = [c for c in splits[i].column_names if c != 'cell_id' or not train_mode]
            splits[i].set_format(type="torch", columns=cols, output_all_columns=False)
        return splits, None, None
    if args['data'].get('precomputed_split'):
        file_path = args['data']['precomputed_split']
        indices, record = _read_precomputed_split(file_path, ds)
        if train_mode:
            _check_split_excludes_heldout(args, ds, [indices], file_path,
                                          record)
        ds = ds.select(indices)
        # Drop cell_id if training
        if train_mode and 'cell_id' in ds.column_names:
            ds = ds.remove_columns(['cell_id'])
        # Set minimal torch format
        cols = [c for c in ds.column_names if c != 'cell_id' or not train_mode]
        ds.set_format(type="torch", columns=cols, output_all_columns=False)
        return ds, None, None

    # Early outs without test and validation sets
    test_set = set(args['data'].get('test_batch_ids', []) or [])
    val_set  = set(args['data'].get('val_batch_ids', [])  or [])

    if len(test_set) == 0 and len(val_set) == 0:
        # Drop cell_id if training
        if train_mode and 'cell_id' in ds.column_names:
            ds = ds.remove_columns(['cell_id'])
        # Set minimal torch format
        cols = [c for c in ds.column_names if c != 'cell_id' or not train_mode]
        ds.set_format(type="torch", columns=cols, output_all_columns=False)
        return ds, None, None

    # Vectorized batch_id extraction
    cell_ids = np.asarray(ds['cell_id'], dtype='U')  # zero-copy from Arrow buffers
    batch_ids = _batch_ids_from_cell_ids(cell_ids)  # "<part0>_<part1>"

    # Masks -> indices (single pass)
    test_mask = np.isin(batch_ids, np.fromiter(test_set, dtype=batch_ids.dtype)) if test_set else np.zeros(len(batch_ids), dtype=bool)
    val_mask  = np.isin(batch_ids, np.fromiter(val_set,  dtype=batch_ids.dtype)) if val_set  else np.zeros(len(batch_ids), dtype=bool)
    train_mask = ~(test_mask | val_mask)

    train_idx = np.nonzero(train_mask)[0].tolist()
    val_idx   = np.nonzero(val_mask)[0].tolist()
    test_idx  = np.nonzero(test_mask)[0].tolist()

    train_ds = ds.select(train_idx)
    val_ds   = ds.select(val_idx)
    test_ds  = ds.select(test_idx)

    # Now drop cell_id for training if requested
    if train_mode and 'cell_id' in train_ds.column_names:
        train_ds = train_ds.remove_columns(['cell_id'])
        if 'cell_id' in val_ds.column_names:  val_ds  = val_ds.remove_columns(['cell_id'])
        if 'cell_id' in test_ds.column_names: test_ds = test_ds.remove_columns(['cell_id'])

    # Minimal torch formatting per split
    def set_minimal_format(d):
        cols = [c for c in d.column_names if c != 'cell_id' or not train_mode]
        d.set_format(type="torch", columns=cols, output_all_columns=False)
        return d

    train_ds = set_minimal_format(train_ds)
    val_ds   = set_minimal_format(val_ds)
    test_ds  = set_minimal_format(test_ds)

    return train_ds, val_ds, test_ds