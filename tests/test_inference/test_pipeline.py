"""
Tests for ``harmonize_tokenize_embed_pipeline`` on tiny model folders.
"""

import numpy as np
import pytest

from terra import embed_dataset, harmonize_tokenize_embed_pipeline

KW = dict(batch_size=8, num_workers=0, nproc=1, processing_mode='sequential')


def _pipeline(bundle, adata, cache, **kwargs):
    return harmonize_tokenize_embed_pipeline(
        adata=adata, batch_key='batch', model_folder_path=str(bundle),
        cache_directory_path=str(cache), **KW, **kwargs)


@pytest.mark.parametrize('uns_batch, obs_batch, expected', [
    (None, None, 'sec1'), (None, 'mine', 'mine'), ('b0', 'mine', 'b0')])
def test_single_sample_batch_label(tiny, tmp_path, uns_batch, obs_batch,
                                   expected):
    # The label was taken from the save path before anything else (''
    # for './sec1.h5ad', 'v1' below), and a pathlib.Path raised
    bundle = tiny.bundle('96M')
    adata = tiny.adata(uns={'batch': uns_batch} if uns_batch else None)
    if obs_batch:
        adata.obs['batch'] = obs_batch
    save_path = tmp_path / 'v1.2' / 'sec1.h5ad'
    save_path.parent.mkdir()
    output = _pipeline(bundle, adata, tmp_path / 'cache', sample_key=None,
                       harmonized_adata_save_path=save_path)
    assert set(output.obs['batch']) == {expected}
    assert save_path.is_file()


def test_single_sample_batch_label_without_save_path(tiny, tmp_path):
    output = _pipeline(tiny.bundle('96M'), tiny.adata(), tmp_path,
                       sample_key=None)
    assert set(output.obs['batch']) == {'sample'}


def test_agg_excluded_genes_are_ensembl_ids(tiny, tmp_path):
    # Documented as token ids for the pipeline and infer(); they are Ensembl
    # IDs, left out of the aggregation only
    bundle = tiny.bundle('96M')
    genes = [f'ENSG{i + 1:011d}' for i in range(3)]
    dataset = tiny.dataset(bundle)
    kwargs = dict(batch_size=8, num_workers=0, return_token_embeddings=True)
    default = embed_dataset(dataset, str(bundle), **kwargs)
    excluded = embed_dataset(dataset, str(bundle), agg_excluded_genes=genes,
                             **kwargs)
    np.testing.assert_array_equal(excluded['token_emb'], default['token_emb'])
    assert not np.allclose(excluded['cell_emb'], default['cell_emb'])

    output = _pipeline(bundle, tiny.adata(), tmp_path, sample_key='sample',
                       agg_excluded_genes=genes)
    np.testing.assert_allclose(output.obsm['cell_emb'], excluded['cell_emb'],
                               rtol=1e-5, atol=1e-6)
