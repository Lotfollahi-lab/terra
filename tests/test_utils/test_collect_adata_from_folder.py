"""collect_adata_from_folder reads dataset folders that are links."""
import os

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from terra.utils.embedding import collect_adata_from_folder


def _write_dataset(folder, dataset_id, n_cells):
    os.makedirs(folder)
    obs = pd.DataFrame({
        'cell_id': [f'{dataset_id}_batch0_{i}' for i in range(n_cells)],
        'cell_type': ['T'] * n_cells})
    obs.index = obs['cell_id'].values
    adata = ad.AnnData(X=np.ones((n_cells, 3), dtype=np.float32), obs=obs)
    adata.obsm['spatial'] = np.zeros((n_cells, 2))
    adata.write_h5ad(os.path.join(folder, 'part.h5ad'))


@pytest.fixture
def silver(tmp_path):
    root = tmp_path / 'silver'
    _write_dataset(root / 'xhs49-1b_1p', '49', 4)
    _write_dataset(root / 'chl59-8b_1p', '59', 3)
    cohort = root / 'hst_corpus_110m'
    cohort.mkdir()
    for name in ('xhs49-1b_1p', 'chl59-8b_1p'):
        os.symlink(os.path.join('..', name), cohort / name)
    return root, cohort


def test_reads_linked_dataset_folders(silver):
    root, cohort = silver
    linked = collect_adata_from_folder(str(cohort), obs_cols=['cell_type'])
    assert sorted(linked.obs['cell_id']) == sorted(
        [f'49_batch0_{i}' for i in range(4)]
        + [f'59_batch0_{i}' for i in range(3)])
    assert 'cell_type' in linked.obs


def test_dataset_filter_applies_to_links(silver):
    _, cohort = silver
    only49 = collect_adata_from_folder(str(cohort), dataset_ids=['49'])
    assert set(only49.obs['cell_id']) == {f'49_batch0_{i}' for i in range(4)}


def test_linked_and_copied_folders_read_the_same(silver, tmp_path):
    root, cohort = silver
    copied = tmp_path / 'copied'
    _write_dataset(copied / 'xhs49-1b_1p', '49', 4)
    _write_dataset(copied / 'chl59-8b_1p', '59', 3)
    a = collect_adata_from_folder(str(cohort), dataset_ids=['49', '59'])
    b = collect_adata_from_folder(str(copied), dataset_ids=['49', '59'])
    assert sorted(a.obs['cell_id']) == sorted(b.obs['cell_id'])
