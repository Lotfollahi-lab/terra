"""
Tests for zeros stored explicitly in a sparse count matrix.

On the sparse ranking path every stored entry of a cell is a gene token, so
stored zeros became gene tokens with value 0 and the same counts stored with
and without them tokenized differently. ``eliminate_stored_zeros=True``
removes them; by default they are kept and a warning is logged.
"""

import logging
import pickle

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

from terra.tokenizers import CellGraphTokenizer, CellNeighborhoodTokenizer

N_CELLS = 40
N_GENES = 60
N_NEIGHS = 4
MODEL_INPUT_SIZE = 8 * (N_NEIGHS + 1)


@pytest.fixture
def token_dictionary_file_path(tmp_path):
    token_dict = {'<pad>': 0, '<mask>': 1}
    token_dict.update({f'ENSG{i:011d}': i + 2 for i in range(N_GENES)})
    path = tmp_path / 'token_dictionary.pkl'
    with open(path, 'wb') as f:
        pickle.dump(token_dict, f)
    return path


def _counts(stored_zeros):
    rng = np.random.default_rng(0)
    # ~6 expressed genes per cell, fewer than the 8 per segment
    counts = rng.poisson(0.1, (N_CELLS, N_GENES)).astype(np.float32)
    counts[counts.sum(axis=1) == 0, 0] = 1
    if not stored_zeros:
        return sp.csr_matrix(counts)
    # The same counts with explicit zeros stored at some zero-count genes
    stored = (counts == 0) & (rng.random(counts.shape) < 0.2)
    x = sp.csr_matrix(np.where(stored, -1, counts))
    x.data[x.data == -1] = 0
    return x


def _adata(stored_zeros):
    adata = ad.AnnData(X=_counts(stored_zeros))
    adata.var['ensembl_id'] = [f'ENSG{i:011d}' for i in range(N_GENES)]
    adata.obs['cell_id'] = [f'cell_{i}' for i in range(N_CELLS)]
    adata.obsm['spatial'] = np.random.default_rng(1).uniform(
        0, 100, (N_CELLS, 2))
    return adata


def _tokenize(token_dictionary_file_path, tokenizer, adata, **kwargs):
    tk = tokenizer(
        model_input_size=MODEL_INPUT_SIZE, n_neighs=N_NEIGHS,
        token_dictionary_file_path=token_dictionary_file_path,
        include_special_tokens=False, rank_cell_norm_method=None,
        rank_gene_norm_method=None, rank_count_norm_method=None,
        count_cell_norm_method=None, count_gene_norm_method=None,
        count_count_norm_method='shifted_log', **kwargs)
    adata_dict = tk._tokenize_adata(adata=adata)
    keys = ('gene_tokens_cell', 'gene_expr_cell', 'gene_tokens_neighborhood',
            'gene_expr_neighborhood', 'seg_tokens_neighborhood')
    examples = [tk._format_examples({key: adata_dict[key][i] for key in keys
                                     if key in adata_dict})
                for i in range(N_CELLS)]
    return (np.stack([example['gene_tokens'] for example in examples]),
            np.stack([example['gene_expr'] for example in examples]))


TOKENIZERS = [CellGraphTokenizer, CellNeighborhoodTokenizer]


@pytest.mark.parametrize('tokenizer', TOKENIZERS)
def test_stored_zeros_are_tokenized_and_reported_by_default(
        token_dictionary_file_path, tokenizer, caplog):
    with caplog.at_level(logging.WARNING):
        tokens, values = _tokenize(token_dictionary_file_path, tokenizer,
                                   _adata(stored_zeros=True))
    canonical, _ = _tokenize(token_dictionary_file_path, tokenizer,
                             _adata(stored_zeros=False))
    # Previous behaviour, kept as default: gene tokens with value 0
    assert np.any((tokens != 0) & (values == 0))
    assert not np.array_equal(tokens, canonical)
    n_stored_zeros = np.count_nonzero(_counts(True).data == 0)
    assert any(record.getMessage().startswith(
        f'adata.X stores {n_stored_zeros} explicit zeros')
        for record in caplog.records)


@pytest.mark.parametrize('tokenizer', TOKENIZERS)
def test_eliminate_stored_zeros(token_dictionary_file_path, tokenizer):
    adata = _adata(stored_zeros=True)
    x_before = adata.X.copy()
    tokens, values = _tokenize(token_dictionary_file_path, tokenizer, adata,
                               eliminate_stored_zeros=True)
    canonical, canonical_values = _tokenize(
        token_dictionary_file_path, tokenizer, _adata(stored_zeros=False))
    np.testing.assert_array_equal(tokens, canonical)
    np.testing.assert_array_equal(values, canonical_values)
    # The input AnnData keeps its stored zeros
    np.testing.assert_array_equal(adata.X.data, x_before.data)
    np.testing.assert_array_equal(adata.X.indices, x_before.indices)


@pytest.mark.parametrize('tokenizer', TOKENIZERS)
def test_data_without_stored_zeros_is_unchanged(token_dictionary_file_path,
                                                tokenizer, caplog):
    with caplog.at_level(logging.WARNING):
        default = _tokenize(token_dictionary_file_path, tokenizer,
                            _adata(stored_zeros=False))
    eliminated = _tokenize(token_dictionary_file_path, tokenizer,
                           _adata(stored_zeros=False),
                           eliminate_stored_zeros=True)
    for d, e in zip(default, eliminated):
        np.testing.assert_array_equal(d, e)
    assert not any('explicit zeros' in record.getMessage()
                   for record in caplog.records)
