"""
Tests for ``harmonize_tokenize_embed_pipeline`` on tiny model folders.
"""

import numpy as np
import pytest

from terra import harmonize_tokenize_embed_pipeline

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
