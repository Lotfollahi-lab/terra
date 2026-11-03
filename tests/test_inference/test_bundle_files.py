"""Gene-reference files of a model bundle in harmonize_tokenize_embed_pipeline:
the earlier name of the gene mapping is read, and a missing file warns."""
import os
import shutil
import warnings

import numpy as np
import pytest

import terra.inference.harmonize as harmonize_module
from terra import harmonize_tokenize_embed_pipeline


class _EnsemblLookup(Exception):
    pass


@pytest.fixture
def no_ensembl_lookup(monkeypatch):
    """Fail if harmonization falls back to an Ensembl release (download)."""
    def lookup(release, species):
        raise _EnsemblLookup(f'Ensembl release {release} ({species})')
    monkeypatch.setattr(harmonize_module, 'EnsemblRelease', lookup)


def _run(bundle, adata, cache):
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter('always')
        result = harmonize_tokenize_embed_pipeline(
            adata=adata, sample_key=None, batch_key='batch',
            model_folder_path=str(bundle), cache_directory_path=str(cache),
            nproc=1, processing_mode='sequential', num_workers=0,
            batch_size=16)
    return result, [str(r.message) for r in records
                    if r.category is UserWarning]


def _copy_bundle(bundle, folder, rename=None, remove=()):
    shutil.copytree(bundle, folder)
    for old, new in (rename or {}).items():
        os.rename(folder / old, folder / new)
    for name in remove:
        os.remove(folder / name)
    return folder


def test_earlier_mapping_file_name_is_read(tiny, tmp_path, no_ensembl_lookup):
    bundle = tiny.bundle('96M')
    # Layout of the bundles published before June 2026 (Hugging Face 'v1.0')
    legacy = _copy_bundle(
        bundle, tmp_path / 'legacy',
        rename={'ensembl_dictionary.pkl': 'ensembl_dic.pkl'},
        remove=['gene_count_dictionary.pkl'])
    adata = tiny.adata()
    current, current_warnings = _run(bundle, adata.copy(), tmp_path / 'c1')
    old, old_warnings = _run(legacy, adata.copy(), tmp_path / 'c2')
    # Same gene mapping, no Ensembl lookup; every gene of the synthetic data
    # occurs often enough in pretraining, so the skipped occurrence filter
    # changes nothing here
    for key in ('cell_emb', 'neighborhood_emb'):
        np.testing.assert_array_equal(current.obsm[key], old.obsm[key])
    assert not any('gene_count_dictionary' in m for m in current_warnings)
    assert any("no 'gene_count_dictionary.pkl'" in m for m in old_warnings)


def test_missing_mapping_warns_before_ensembl_lookup(tiny, tmp_path,
                                                     no_ensembl_lookup):
    no_mapping = _copy_bundle(tiny.bundle('96M'), tmp_path / 'no_mapping',
                              remove=['ensembl_dictionary.pkl'])
    with pytest.warns(UserWarning, match='Ensembl release 111'):
        with pytest.raises(_EnsemblLookup):
            harmonize_tokenize_embed_pipeline(
                adata=tiny.adata(), sample_key=None, batch_key='batch',
                model_folder_path=str(no_mapping),
                cache_directory_path=str(tmp_path / 'c'), nproc=1,
                processing_mode='sequential', num_workers=0, batch_size=16)
