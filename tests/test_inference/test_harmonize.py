"""
Tests for ``harmonize_adata``: count validation, gene mapping and filtering.
"""

import pickle
import threading

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from terra import harmonize_adata

N_CELLS = 30
N_GENES = 25


@pytest.fixture
def gene_mapping_dict_file_path(tmp_path):
    path = tmp_path / 'ensembl_dictionary.pkl'
    with open(path, 'wb') as f:
        pickle.dump({f'GENE{i}': f'ENSG{i + 1:011d}' for i in range(N_GENES)},
                    f)
    return str(path)


def _adata(seed=0):
    rng = np.random.default_rng(seed)
    adata = ad.AnnData(X=sp.csr_matrix(
        rng.poisson(2.0, (N_CELLS, N_GENES)).astype(np.float32)))
    adata.var_names = pd.Index([f'GENE{i}' for i in range(N_GENES)],
                               name='symbol')
    adata.obs_names = [f'c{i}' for i in range(N_CELLS)]
    return adata


def _harmonize(adata, gene_mapping_dict_file_path, **kwargs):
    return harmonize_adata(
        adata, gene_mapping_dict_file_path=gene_mapping_dict_file_path,
        min_genes_per_cell=0, min_cells_per_gene=0, **kwargs)


def test_counts_layer_is_used_without_modifying_the_input(
        gene_mapping_dict_file_path):
    adata = _adata()
    adata.layers['counts'] = adata.X.copy()
    adata.X = adata.X.log1p()
    x_before, var_before = adata.X.copy(), adata.var_names.copy()

    harmonized = _harmonize(adata, gene_mapping_dict_file_path)

    # The counts are used ...
    np.testing.assert_array_equal(harmonized.X.toarray(),
                                  adata.layers['counts'].toarray())
    # ... and the caller's AnnData is unchanged
    np.testing.assert_array_equal(adata.X.toarray(), x_before.toarray())
    assert adata.var_names.equals(var_before)
    assert adata.var_names.name == 'symbol'


def _harmonize_in_thread(adata, gene_mapping_dict_file_path):
    result = {}

    def target():
        try:
            _harmonize(adata, gene_mapping_dict_file_path)
            result['returned'] = True
        except Exception as e:  # noqa: BLE001 - the test inspects it
            result['error'] = e

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout=30)
    assert not thread.is_alive(), 'harmonize_adata did not return'
    return result


def test_non_integer_counts_raise_instead_of_hanging(
        gene_mapping_dict_file_path):
    # Previously an endless loop when X and layers['counts'] were both
    # non-integer
    adata = _adata()
    adata.X = adata.X.log1p()
    adata.layers['counts'] = adata.X.copy()
    result = _harmonize_in_thread(adata, gene_mapping_dict_file_path)
    assert isinstance(result.get('error'), ValueError)
    assert "layers['counts']" in str(result['error'])

    del adata.layers['counts']
    with pytest.raises(ValueError, match='does not contain raw counts'):
        _harmonize(adata, gene_mapping_dict_file_path)
