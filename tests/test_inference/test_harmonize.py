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


def test_ensembl_id_var_names_are_mapped_directly(
        gene_mapping_dict_file_path):
    # var_names that are already Ensembl IDs (e.g. 10x gene_ids, some with a
    # version suffix) previously matched no gene name and every gene was
    # dropped
    by_name = _adata()
    by_id = _adata()
    by_id.var_names = [f'ENSG{i + 1:011d}' + ('.3' if i % 2 else '')
                       for i in range(N_GENES)]

    harmonized_by_name = _harmonize(by_name, gene_mapping_dict_file_path)
    harmonized_by_id = _harmonize(by_id, gene_mapping_dict_file_path)

    assert list(harmonized_by_id.var_names) == list(by_id.var_names)
    assert list(harmonized_by_id.var['ensembl_id']) == list(
        harmonized_by_name.var['ensembl_id'])
    np.testing.assert_array_equal(harmonized_by_id.X.toarray(),
                                  harmonized_by_name.X.toarray())


def test_ensembl_ids_unknown_to_the_model_are_dropped(
        gene_mapping_dict_file_path):
    adata = _adata()
    names = [f'ENSG{i + 1:011d}' for i in range(N_GENES)]
    names[:2] = ['ENSG99999999999', 'ENSMUSG00000000001']
    adata.var_names = names
    harmonized = _harmonize(adata, gene_mapping_dict_file_path)
    assert list(harmonized.var_names) == names[2:]


def test_gene_names_take_precedence_over_ensembl_ids(
        gene_mapping_dict_file_path):
    # Mostly gene names: an Ensembl ID among them is not mapped, as before
    adata = _adata()
    names = list(adata.var_names)
    names[0] = 'ENSG00000000001'
    adata.var_names = names
    harmonized = _harmonize(adata, gene_mapping_dict_file_path)
    assert list(harmonized.var_names) == names[1:]


def test_no_mapped_gene_raises(gene_mapping_dict_file_path):
    adata = _adata()
    adata.var_names = [f'gene{i}' for i in range(N_GENES)]
    with pytest.raises(ValueError, match='None of the 25 genes'):
        _harmonize(adata, gene_mapping_dict_file_path)


def test_almost_no_mapped_gene_warns(gene_mapping_dict_file_path, caplog):
    adata = _adata()
    adata.var_names = ['GENE0'] + [f'gene{i}' for i in range(1, N_GENES)]
    harmonized = _harmonize(adata, gene_mapping_dict_file_path)
    assert harmonized.n_vars == 1
    assert any(record.getMessage().startswith('Only 1 of 25 genes')
               for record in caplog.records)
