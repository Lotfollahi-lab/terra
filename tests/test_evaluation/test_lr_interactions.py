"""Ligand-receptor utilities of ``terra.evaluation``: optional dependencies
are imported lazily, and CellPhoneDB pairs of hyphenated genes are kept."""
import importlib
import logging
import sys
import types

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from terra.evaluation.lr_interactions import _split_ligand_receptor


GENES = ['LGALS9', 'HAVCR2', 'HLA-E', 'KLRC1', 'HLA-C', 'KIR2DL1', 'CD274',
         'PDCD1']


def _adata(genes=GENES):
    return ad.AnnData(
        np.zeros((2, len(genes)), dtype=np.float32),
        var=pd.DataFrame(
            {'ensembl_id': [f'ENSG{i:011d}' for i in range(len(genes))]},
            index=genes))


def test_evaluation_imports_without_eval_extra(monkeypatch):
    # Neither the probes nor the package need cellphonedb / omnipath / IPython
    for name in ('cellphonedb', 'cellphonedb.utils', 'omnipath', 'IPython',
                 'IPython.display'):
        monkeypatch.setitem(sys.modules, name, None)
    for name in [m for m in sys.modules if m == 'terra.evaluation'
                 or m.startswith('terra.evaluation.')]:
        monkeypatch.delitem(sys.modules, name)
    evaluation = importlib.import_module('terra.evaluation')
    assert callable(evaluation.linear_classifier)
    assert callable(evaluation.knn_classifier)
    # The ligand-receptor utilities name the missing extra
    with pytest.raises(ModuleNotFoundError, match=r'terra-st\[eval\]'):
        evaluation.get_adata_omnipath_lr_pairs(_adata())
    with pytest.raises(ModuleNotFoundError, match=r'terra-st\[eval\]'):
        evaluation.get_adata_cellphonedb_lr_pairs(_adata(), 'unused')


def test_split_ligand_receptor():
    genes = set(GENES)
    assert _split_ligand_receptor('LGALS9-HAVCR2', genes) == ('LGALS9',
                                                             'HAVCR2')
    assert _split_ligand_receptor('HLA-E-KLRC1', genes) == ('HLA-E', 'KLRC1')
    assert _split_ligand_receptor('CD274-UNKNOWN', genes) == (None, None)


def test_split_ligand_receptor_ambiguous(caplog):
    caplog.set_level(logging.WARNING)
    assert _split_ligand_receptor('A-B-C', {'A', 'B-C', 'A-B', 'C'}) == (
        None, None)
    assert 'several gene pairs' in caplog.text


@pytest.fixture
def cellphonedb(monkeypatch, tmp_path):
    """A stub cellphonedb whose 'download' writes a small interaction table."""
    def download_database(target_dir, version):
        import os
        os.makedirs(target_dir, exist_ok=True)
        pd.DataFrame({
            'interactors': ['LGALS9-HAVCR2', 'HLA-E-KLRC1', 'HLA-C-KIR2DL1',
                            'CD274-PDCD1', 'TGFB1-TGFBR1+TGFBR2',
                            'CXCL12-CXCR4'],
            'directionality': ['Ligand-Receptor'] * 6,
        }).to_csv(f'{target_dir}/interaction_input.csv', index=False)

    utils = types.ModuleType('cellphonedb.utils')
    utils.db_utils = types.SimpleNamespace(download_database=download_database)
    utils.db_releases_utils = types.SimpleNamespace(
        get_remote_database_versions_html=lambda: {
            'db_releases_html_table': '<table></table>'})
    package = types.ModuleType('cellphonedb')
    package.utils = utils
    monkeypatch.setitem(sys.modules, 'cellphonedb', package)
    monkeypatch.setitem(sys.modules, 'cellphonedb.utils', utils)
    return tmp_path


def test_cellphonedb_pairs_keep_hyphenated_genes(cellphonedb):
    from terra.evaluation import get_adata_cellphonedb_lr_pairs
    adata = _adata()
    ensembl = adata.var['ensembl_id']
    pairs = get_adata_cellphonedb_lr_pairs(adata, str(cellphonedb))
    assert list(pairs.columns) == ['ligand', 'receptor']
    assert set(zip(pairs['ligand'], pairs['receptor'])) == {
        (ensembl[a], ensembl[b]) for a, b in [
            ('LGALS9', 'HAVCR2'), ('HLA-E', 'KLRC1'), ('HLA-C', 'KIR2DL1'),
            ('CD274', 'PDCD1')]}
    # Earlier behaviour, split at the first hyphen: the HLA pairs are lost
    legacy = get_adata_cellphonedb_lr_pairs(adata, str(cellphonedb),
                                            legacy_split=True)
    assert set(zip(legacy['ligand'], legacy['receptor'])) == {
        (ensembl['LGALS9'], ensembl['HAVCR2']),
        (ensembl['CD274'], ensembl['PDCD1'])}
    # Pairs of genes without hyphens are unchanged (rows and order)
    pd.testing.assert_frame_equal(
        legacy, pairs.loc[legacy.index])
