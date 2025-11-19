import json
import random
import requests
from typing import Literal

import pickle
import datasets
from datasets import load_from_disk


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
            print(f'Could not find Ensembl IDs for genes: {missing_genes}.')
        return ensembl_ids
    else:
        response.raise_for_status()


def prepare_dataset(args: dict, train_mode: bool = False):
    data_path = args['data']['tokenized_data_folder_path']
    ds = load_from_disk(data_path)

    # 1) Remove obviously unneeded columns but KEEP cell_id for now
    fields_to_remove = [
        'cell_degrees','batch_value_token','gene_panel_value_token',
        'assay_value_token','species_value_token','tissue_value_token',
        'cls_tokens','cell_total_counts','cell_n_probed_genes'
    ]
    if 'cell_pos_enc' not in args['meta'] or args['meta']['cell_pos_enc'] == 'segment':
        fields_to_remove += ['rel_x_coord','rel_y_coord']

    existing = [c for c in fields_to_remove if c in ds.column_names]
    if existing:
        ds = ds.remove_columns(existing)

    # Early outs with precomputed splits
    if args['data'].get('precomputed_epoch_splits'):
        with open(args['data']['precomputed_epoch_splits'], 'rb') as f:
            epoch_indices = pickle.load(f)
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
        with open(args['data']['precomputed_split'], 'rb') as f:
            indices = pickle.load(f)
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
    p1 = np.char.partition(cell_ids, '_')
    first = p1[:, 0]
    rest = p1[:, 2]
    p2 = np.char.partition(rest, '_')
    second = p2[:, 0]
    batch_ids = np.char.add(np.char.add(first, '_'), second)  # "<part0>_<part1>"

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