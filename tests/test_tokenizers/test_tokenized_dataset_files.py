"""
Tokenizing a directory of files:

* the row order of the tokenized dataset follows the sorted file paths in
  both processing modes (it followed the file system's listing order, and the
  completion order in parallel mode, while precomputed splits refer to row
  numbers);
* the settings that determine the tokens are recorded with the dataset, and
  train() checks the model config against them and records them in it (the
  corpus could be tokenized without settings that the model config claimed,
  which inference then applied to new data);
* special tokens: only the model's special tokens are read, so a model with
  an assay token can keep it on a batch it was not trained on.
"""

import json
import logging
import pickle

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from datasets import load_from_disk

from terra.inference.embed import _check_special_token_metadata
from terra.inference.tokenize import special_tokens_to_read
from terra.tokenizers import CellGraphTokenizer
from terra.tokenizers.params import (TOKENIZER_PARAMS_FILE,
                                     reconcile_tokenizer_params)

N_GENES = 30
N_NEIGHS = 2
MODEL_INPUT_SIZE = 8 * (N_NEIGHS + 1)
UNS = dict(dataset_id='1', batch='batch0', assay='xenium',
           species='homo_sapiens', tissue='skin')
NORM_KWARGS = dict(rank_cell_norm_method=None, rank_gene_norm_method=None,
                   rank_count_norm_method=None, count_cell_norm_method=None,
                   count_gene_norm_method=None,
                   count_count_norm_method='shifted_log')


@pytest.fixture
def token_dictionary_file_path(tmp_path):
    token_dict = {'<pad>': 0, '<mask>': 1}
    token_dict.update({f'ENSG{i:011d}': i + 2 for i in range(N_GENES)})
    for key in ('spv_homo_sapiens', 'spv_skin', 'spv_xenium',
                f'spv_gene_panel{N_GENES}', 'spv_1_batch0'):
        token_dict[key] = len(token_dict)
    path = tmp_path / 'token_dictionary.pkl'
    with open(path, 'wb') as f:
        pickle.dump(token_dict, f)
    return path


def _adata(prefix, n_cells=10, seed=0, uns=UNS):
    rng = np.random.default_rng(seed)
    counts = rng.poisson(1.0, (n_cells, N_GENES)).astype(np.float32)
    counts[:, 0] += 1  # every cell expresses a gene
    adata = ad.AnnData(X=sp.csr_matrix(counts))
    adata.var['ensembl_id'] = [f'ENSG{i:011d}' for i in range(N_GENES)]
    adata.obs['cell_id'] = [f'{prefix}_{i}' for i in range(n_cells)]
    adata.obsm['spatial'] = rng.uniform(0, 100, (n_cells, 2))
    adata.uns.update(uns)
    return adata


def _tokenizer(token_dictionary_file_path, **kwargs):
    return CellGraphTokenizer(
        model_input_size=MODEL_INPUT_SIZE, n_neighs=N_NEIGHS,
        token_dictionary_file_path=token_dictionary_file_path,
        **{'include_special_tokens': False, **NORM_KWARGS, **kwargs})


def _tokenize_directory(tmp_path, token_dictionary_file_path, **kwargs):
    input_directory = tmp_path / 'silver'
    input_directory.mkdir(exist_ok=True)
    # Written in an order that is not the sorted one
    for i, name in enumerate(['b', 'c', 'a']):
        _adata(name, seed=i).write_h5ad(input_directory / f'{name}.h5ad')
    tk = _tokenizer(token_dictionary_file_path, **kwargs)
    tk.tokenize_data(input_directory=input_directory,
                     output_directory=tmp_path, output_file_prefix='corpus')
    return tmp_path / 'corpus.dataset'


@pytest.mark.parametrize('processing_mode', ['sequential', 'parallel'])
def test_rows_follow_the_sorted_file_paths(
        tmp_path, token_dictionary_file_path, processing_mode):
    dataset_path = _tokenize_directory(
        tmp_path, token_dictionary_file_path,
        processing_mode=processing_mode, nproc=2)
    prefixes = [cell_id.split('_')[0]
                for cell_id in load_from_disk(str(dataset_path))['cell_id']]
    assert prefixes == ['a'] * 10 + ['b'] * 10 + ['c'] * 10


def test_tokenize_data_records_its_settings(tmp_path,
                                            token_dictionary_file_path):
    dataset_path = _tokenize_directory(
        tmp_path, token_dictionary_file_path, stable_rank_ties=True,
        pad_empty_neighbors=True)
    with open(dataset_path / TOKENIZER_PARAMS_FILE) as f:
        recorded = json.load(f)
    assert recorded['tokenizer'] == 'CellGraphTokenizer'
    assert recorded['stable_rank_ties'] is True
    assert recorded['pad_empty_neighbors'] is True
    assert recorded['eliminate_stored_zeros'] is False
    assert recorded['count_count_norm_method'] == 'shifted_log'
    # The extra file does not disturb loading
    assert len(load_from_disk(str(dataset_path))) == 30


def _data_config(dataset_path, **keys):
    return {'tokenized_data_folder_path': str(dataset_path),
            'tokenizer_type': 'cell_graph',
            'norm_factor_file_path': None, 'pf_targets_file_path': None,
            **NORM_KWARGS, **keys}


def test_reconcile_records_settings_the_config_leaves_out(
        tmp_path, token_dictionary_file_path):
    dataset_path = _tokenize_directory(
        tmp_path, token_dictionary_file_path, stable_rank_ties=True,
        pad_empty_neighbors=True)
    data_config = _data_config(dataset_path)
    reconcile_tokenizer_params(data_config)
    assert data_config['stable_rank_ties'] is True
    assert data_config['pad_empty_neighbors'] is True
    assert data_config['eliminate_stored_zeros'] is False


@pytest.mark.parametrize('keys, mismatch', [
    (dict(stable_rank_ties=True), 'stable_rank_ties'),
    (dict(pad_empty_neighbors=True), 'pad_empty_neighbors'),
    (dict(count_count_norm_method='pflog1ppf'), 'count_count_norm_method'),
    (dict(tokenizer_type='cell_neigh'), 'tokenizer_type'),
])
def test_reconcile_rejects_settings_the_data_were_not_tokenized_with(
        tmp_path, token_dictionary_file_path, keys, mismatch):
    dataset_path = _tokenize_directory(tmp_path, token_dictionary_file_path)
    with pytest.raises(ValueError, match=mismatch):
        reconcile_tokenizer_params(_data_config(dataset_path, **keys))


def test_reconcile_checks_the_pf_targets_file(tmp_path,
                                              token_dictionary_file_path):
    pf_targets = tmp_path / 'pf_targets.csv'
    pd.DataFrame({'pf_depth_target': [100.0],
                  'pf_logsum_target': [5.0]}).to_csv(pf_targets, index=False)
    dataset_path = _tokenize_directory(
        tmp_path, token_dictionary_file_path,
        count_count_norm_method='pflog1ppf', pf_targets_file_path=pf_targets)
    keys = dict(count_count_norm_method='pflog1ppf')
    reconcile_tokenizer_params(
        _data_config(dataset_path, **keys, pf_targets_file_path=str(pf_targets)))
    other = tmp_path / 'other_pf_targets.csv'
    pd.DataFrame({'pf_depth_target': [90.0],
                  'pf_logsum_target': [5.0]}).to_csv(other, index=False)
    for path in (str(other), None):
        with pytest.raises(ValueError, match='pf_targets_file_path'):
            reconcile_tokenizer_params(_data_config(
                dataset_path, **keys, pf_targets_file_path=path))


def test_reconcile_accepts_datasets_without_a_record(tmp_path, caplog):
    data_config = _data_config(tmp_path, stable_rank_ties=True)
    with caplog.at_level(logging.WARNING):
        reconcile_tokenizer_params(data_config)
    assert 'has no record of its tokenizer settings' in caplog.text
    assert data_config['stable_rank_ties'] is True


def test_assay_only_tokenization_needs_no_batch(token_dictionary_file_path):
    full = _tokenizer(token_dictionary_file_path, include_special_tokens=True)
    expected = full._tokenize_adata(adata=_adata('a'))
    # A batch the model was not trained on, and no species/tissue entry
    adata = _adata('a', uns=dict(assay='xenium', dataset_id='9', batch='b7'))
    with pytest.raises(KeyError):
        full._tokenize_adata(adata=adata.copy())
    assay_only = _tokenizer(token_dictionary_file_path,
                            include_special_tokens=True,
                            special_tokens=['assay'])
    result = assay_only._tokenize_adata(adata=adata)
    assert result['assay_value'] == expected['assay_value']
    assert result['assay_value'][0] > 0
    for name in ('batch', 'gene_panel', 'species', 'tissue'):
        assert set(result[f'{name}_value']) == {0}


def test_unknown_special_token_name_raises(token_dictionary_file_path):
    with pytest.raises(ValueError, match='Unknown special tokens'):
        _tokenizer(token_dictionary_file_path, special_tokens=['donor'])


def test_metadata_check_follows_the_model_special_tokens(
        token_dictionary_file_path):
    with open(token_dictionary_file_path, 'rb') as f:
        token_dict = pickle.load(f)
    adata = _adata('a', uns=dict(assay='xenium'))
    _check_special_token_metadata(adata, token_dict, special_tokens=['assay'])
    with pytest.raises(ValueError, match="'dataset_id', 'batch'"):
        _check_special_token_metadata(adata, token_dict,
                                      special_tokens=['assay', 'batch'])
    with pytest.raises(ValueError, match="'species', 'tissue'"):
        _check_special_token_metadata(adata, token_dict)


def test_special_tokens_to_read():
    config = {'meta': {'special_tokens': ['assay']}}
    assert special_tokens_to_read(config) == ['assay']
    # A correction mechanism may read other special token values
    config['special_token_correction'] = {'adaln': {'enabled': True}}
    assert special_tokens_to_read(config) is None
