"""
Tests for the handling of the caller's data by the inference API.

* ``harmonize_tokenize_embed_pipeline`` made the cell names unique with
  ``adata.obs_names_make_unique()`` on the caller's AnnData, which renamed
  the caller's cells. The harmonized copy now gets the unique names (the same
  ones as before) and the caller's AnnData keeps its names.
* A sample with no more cells than the model's ``n_neighs`` failed in
  scikit-learn ("Expected n_neighbors < n_samples_fit") after the previous
  samples were tokenized; the pipeline now names the sample.
* ``tokenize_adata`` log-transformed a dense float32 ``adata.X`` in place,
  so tokenizing the same AnnData again gave log1p(log1p(counts)) values.
* The pipeline documented ``gene_perturb_df`` but ignored it, so the returned
  embeddings were unperturbed; it now raises a ``ValueError``.
"""

import numpy as np
import pandas as pd
import pytest

from terra import (harmonize_adata, harmonize_tokenize_embed_pipeline,
                   tokenize_adata)

KW = dict(batch_key='batch', nproc=1, processing_mode='sequential',
          num_workers=0, batch_size=8)


@pytest.mark.parametrize('sample_key', ['sample', None])
@pytest.mark.parametrize('view', [False, True])
def test_pipeline_keeps_the_cell_names_of_the_caller(
        tiny, tmp_path, sample_key, view):
    bundle = tiny.bundle('96M')
    adata = tiny.adata(n_cells=44)
    adata.obs_names = [f'c{i % 22}' for i in range(44)]
    adata.obs['sample'] = ['S1'] * 22 + ['S2'] * 22
    if sample_key is None:
        adata.obs['sample'] = 'S1'
    if view:
        adata = adata[:40]
    names = adata.obs_names.copy()
    expected = adata.copy()
    expected.obs_names_make_unique()

    result = harmonize_tokenize_embed_pipeline(
        adata=adata, sample_key=sample_key, model_folder_path=str(bundle),
        cache_directory_path=str(tmp_path / 'cache'), **KW)

    assert adata.obs_names.equals(names)
    assert adata.is_view == view
    assert list(result.obs_names) == list(expected.obs_names)


def test_pipeline_names_a_sample_with_too_few_cells(tiny, tmp_path):
    bundle = tiny.bundle('96M')  # n_neighs 3
    adata = tiny.adata(n_cells=43)
    adata.obs['sample'] = ['S2'] * 3 + ['S1'] * 40
    with pytest.raises(ValueError,
                       match="Sample 'S2' has 3 cells .* at least 4 cells"):
        harmonize_tokenize_embed_pipeline(
            adata=adata, sample_key='sample', model_folder_path=str(bundle),
            cache_directory_path=str(tmp_path / 'cache'), **KW)


def test_tokenizing_an_anndata_twice_gives_the_same_data(tiny, tmp_path):
    bundle = tiny.bundle('96M')
    harmonized = harmonize_adata(
        tiny.adata(),
        gene_mapping_dict_file_path=str(bundle / 'ensembl_dictionary.pkl'),
        min_genes_per_cell=0, min_cells_per_gene=0)
    assert isinstance(harmonized.X, np.ndarray)
    assert harmonized.X.dtype == np.float32
    counts = harmonized.X.copy()
    datasets = [
        tokenize_adata(harmonized, model_folder_path=str(bundle),
                       cache_directory_path=str(tmp_path / f'cache{i}'),
                       nproc=1, processing_mode='sequential')
        for i in range(2)]

    np.testing.assert_array_equal(harmonized.X, counts)
    assert list(harmonized.obsp) == []
    for column in ('gene_tokens', 'gene_expr'):
        np.testing.assert_array_equal(
            np.stack([np.asarray(x) for x in datasets[0][column]]),
            np.stack([np.asarray(x) for x in datasets[1][column]]))


@pytest.mark.parametrize('sample_key', ['sample', None])
def test_pipeline_rejects_gene_perturbations(tiny, tmp_path, sample_key):
    bundle = tiny.bundle('96M')
    gene_perturb_df = pd.DataFrame({
        'perturbed_cell_id': ['all'], 'perturbed_ensembl_id': ['all'],
        'perturbation_target': ['cell'], 'perturbation_type': ['knockout'],
        'foldchange': [np.nan]})
    with pytest.raises(ValueError, match='perturb_dataset'):
        harmonize_tokenize_embed_pipeline(
            adata=tiny.adata(), sample_key=sample_key,
            model_folder_path=str(bundle),
            cache_directory_path=str(tmp_path / 'cache'),
            gene_perturb_df=gene_perturb_df, **KW)
