"""
Tests for the global numpy and torch random number generators at inference.

Importing the embed, infer and score modules seeds both generators with 0
(``np.random.seed(0)``, ``torch.manual_seed(0)``). This side effect of
``import terra`` is kept on purpose: analyses that draw from the global
generators after the import without seeding them (e.g. the linear probes of
``terra.evaluation`` in the published benchmarks) reproduce only with it. The
embeddings themselves do not depend on the global generators, and the
subsampling of the point-cloud summaries can be seeded with ``random_state``.
"""

import importlib.util
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch

import terra.inference
from terra import embed_dataset
from terra.inference import summarize_distance, summarize_energy_by_label


def _global_states():
    return np.random.get_state()[1].copy(), torch.get_rng_state().clone()


def _assert_global_states(states):
    np.testing.assert_array_equal(np.random.get_state()[1], states[0])
    assert torch.equal(torch.get_rng_state(), states[1])


def _exec_fresh_copy(name):
    # Execute a fresh copy of the module (its imports are cached)
    path = Path(terra.inference.__file__).parent / f'{name}.py'
    spec = importlib.util.spec_from_file_location(
        f'terra.inference._{name}_copy', path)
    spec.loader.exec_module(importlib.util.module_from_spec(spec))


@pytest.mark.parametrize('name', ['embed', 'infer', 'score'])
def test_module_import_seeds_global_rng_with_zero(name):
    np.random.seed(0)
    torch.manual_seed(0)
    seeded = _global_states()

    np.random.seed(123)
    torch.manual_seed(123)
    _exec_fresh_copy(name)

    _assert_global_states(seeded)


@pytest.mark.parametrize('name', ['perturb', 'summarize_distance',
                                  'token_distance', 'harmonize', 'tokenize'])
def test_other_module_imports_keep_global_rng_state(name):
    np.random.seed(123)
    torch.manual_seed(123)
    states = _global_states()

    _exec_fresh_copy(name)

    _assert_global_states(states)


def test_embeddings_do_not_depend_on_global_rng_state(tiny):
    # The model is initialized with the global generators before its weights
    # are loaded, so the global random state does not enter the embeddings
    bundle = tiny.bundle('112M')
    dataset = tiny.dataset(bundle)
    outputs = []
    for seed in (1, 2):
        np.random.seed(seed)
        torch.manual_seed(seed)
        outputs.append(embed_dataset(
            dataset, str(bundle), batch_size=8, num_workers=0))
    for key in outputs[0]:
        np.testing.assert_array_equal(outputs[0][key], outputs[1][key])


def test_summary_subsampling_random_state(monkeypatch):
    rng = np.random.default_rng(0)
    adata = ad.AnnData(obs=pd.DataFrame(
        {'label': np.repeat(['a', 'b'], [12, 9])},
        index=[f'c{i}' for i in range(21)]))
    adata.obsm['cell_emb'] = rng.normal(size=(21, 4))
    adata.obsm['cell_emb_perturb_case1'] = rng.normal(size=(21, 4))
    pairs = [('cell_emb', 'cell_emb_perturb_case1', 'cell_emb_energy')]

    # Record the subsampled clouds instead of computing distances
    clouds = []

    def record(X, Y, **kwargs):
        clouds.append(X)
        return 0.0

    monkeypatch.setattr(
        summarize_distance, '_geomloss_distance_pointcloud', record)

    def subsamples(**kwargs):
        clouds.clear()
        summarize_energy_by_label(
            adata, 'label', pairs=pairs, threshold=5, **kwargs)
        return list(clouds)

    def expected(generator):
        # Same draws in the same order (label by label) from `generator`
        clouds_expected = []
        for label in ('a', 'b'):
            rows = (adata.obs['label'] == label).to_numpy()
            X = summarize_distance._l2_normalize_rows(
                adata.obsm['cell_emb'][rows])
            clouds_expected.append(
                X[generator.choice(X.shape[0], size=5, replace=False)])
        return clouds_expected

    # Default: numpy's global generator, as before
    np.random.seed(3)
    for got, want in zip(subsamples(), expected(np.random.RandomState(3))):
        np.testing.assert_array_equal(got, want)

    # Explicit seed: independent of, and not drawing from, the global
    # generator
    np.random.seed(1)
    states = _global_states()
    first = subsamples(random_state=7)
    _assert_global_states(states)
    np.random.seed(2)
    second = subsamples(random_state=7)
    for got, want in zip(first, expected(np.random.RandomState(7))):
        np.testing.assert_array_equal(got, want)
    for got, want in zip(second, first):
        np.testing.assert_array_equal(got, want)
