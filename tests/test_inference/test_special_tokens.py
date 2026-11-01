"""
Tests for keeping or padding the special tokens (metatokens) at inference.

TERRA-112M saw its batch metatoken in every training sequence, but inference
pads it by default. The warning about this was only logged (invisible with
TERRA's NullHandler), and ``keep_special_tokens=True`` failed on external
data with ``KeyError: 'dataset_id'`` (pipeline) or ``KeyError:
'batch_value'`` (``embed_dataset``) instead of saying what is needed.
"""

import warnings

import numpy as np
import pytest

from terra import embed_dataset, harmonize_tokenize_embed_pipeline

KW = dict(batch_size=8, num_workers=0)


def _pipeline(bundle, adata, cache, **kwargs):
    return harmonize_tokenize_embed_pipeline(
        adata=adata, sample_key='sample', batch_key='batch',
        model_folder_path=str(bundle), cache_directory_path=str(cache),
        nproc=1, processing_mode='sequential', **KW, **kwargs)


def test_padding_warning_is_visible(tiny):
    bundle = tiny.bundle('112M')
    dataset = tiny.dataset(bundle)
    with pytest.warns(UserWarning, match='out of the training distribution'):
        embed_dataset(dataset, str(bundle), **KW)

    kept = tiny.dataset(bundle, include_special_tokens=True)
    with warnings.catch_warnings():
        warnings.filterwarnings('error', message='.*training distribution')
        embed_dataset(kept, str(bundle), keep_special_tokens=True, **KW)


def test_keep_without_special_token_columns(tiny):
    bundle = tiny.bundle('112M')
    with pytest.raises(ValueError, match='batch_value'):
        embed_dataset(tiny.dataset(bundle), str(bundle),
                      keep_special_tokens=True, **KW)


def test_pipeline_keep_names_missing_metadata(tiny, tmp_path):
    bundle = tiny.bundle('112M')
    with pytest.raises(ValueError, match="'dataset_id', 'batch', 'assay'"):
        _pipeline(bundle, tiny.adata(), tmp_path,
                  keep_special_tokens=True)


def test_pipeline_keep_unseen_batch(tiny, tmp_path):
    bundle = tiny.bundle('112M')
    adata = tiny.adata(uns=dict(tiny.uns, dataset_id='ds9', batch='b7'))
    with pytest.raises(ValueError, match='spv_ds9_b7'):
        _pipeline(bundle, adata, tmp_path, keep_special_tokens=True)


def test_pipeline_keep_with_training_batch(tiny, tmp_path):
    bundle = tiny.bundle('112M')
    adata = tiny.adata(uns=tiny.uns)
    kept = _pipeline(bundle, adata.copy(), tmp_path / 'kept',
                     keep_special_tokens=True)
    padded = _pipeline(bundle, adata.copy(), tmp_path / 'padded')
    expected = embed_dataset(tiny.dataset(bundle, include_special_tokens=True),
                             str(bundle), keep_special_tokens=True, **KW)
    np.testing.assert_allclose(kept.obsm['cell_emb'], expected['cell_emb'],
                               rtol=1e-5, atol=1e-6)
    assert not np.allclose(kept.obsm['cell_emb'], padded.obsm['cell_emb'])


def test_pipeline_keep_without_special_tokens(tiny, tmp_path):
    # TERRA-96M has no special tokens: nothing to keep, no metadata needed
    bundle = tiny.bundle('96M')
    kept = _pipeline(bundle, tiny.adata(), tmp_path / 'kept',
                     keep_special_tokens=True)
    default = _pipeline(bundle, tiny.adata(), tmp_path / 'default')
    np.testing.assert_array_equal(kept.obsm['cell_emb'],
                                  default.obsm['cell_emb'])
