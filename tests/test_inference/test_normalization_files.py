"""
Tests for the normalization artifacts that ``tokenize_adata`` reads.

Training copies the gene-level norm factors and the PFlog1pPF targets into the
model folder (``norm_factors.csv``, ``pf_targets.csv``), and ``terra.hub``
downloads them with a model, but ``tokenize_adata`` read only the paths
recorded in the config at training time (e.g. on the training cluster), so
tokenizing for a downloaded gene-normalized model failed with a
``FileNotFoundError``. The copies in the model folder are now preferred; the
config paths remain the fallback.
"""

import logging

import pytest
import yaml

from terra.inference import tokenize as inference_tokenize

DATA = dict(tokenizer_type='cell_graph', model_input_size=40, n_neighs=4,
            rank_cell_norm_method=None, rank_gene_norm_method=None,
            rank_count_norm_method=None, count_cell_norm_method=None,
            count_gene_norm_method='mean', count_count_norm_method='pflog1ppf')


class _Stop(Exception):
    pass


def _tokenizer_paths(tmp_path, monkeypatch, config_paths, bundled):
    captured = {}

    def capture(**kwargs):
        captured.update(kwargs)
        raise _Stop

    monkeypatch.setattr(inference_tokenize.cell_tokenizers,
                        'CellGraphTokenizer', capture)
    model_folder = tmp_path / 'model'
    model_folder.mkdir()
    with open(model_folder / 'model_config.yaml', 'w') as f:
        yaml.safe_dump({'data': {**DATA, **config_paths}}, f)
    for name in bundled:
        (model_folder / name).write_text('gene_id,mean\n')
    with pytest.raises(_Stop):
        inference_tokenize.tokenize_adata(
            adata=None, model_folder_path=str(model_folder),
            cache_directory_path=str(tmp_path))
    return (captured['norm_factor_file_path'],
            captured['pf_targets_file_path'], model_folder)


CONFIG_PATHS = {'norm_factor_file_path': '/cluster/run/norm_factors.csv',
                'pf_targets_file_path': '/cluster/run/pf_targets.csv'}


def test_model_folder_copies_are_preferred(tmp_path, monkeypatch):
    norm_factors, pf_targets, model_folder = _tokenizer_paths(
        tmp_path, monkeypatch, CONFIG_PATHS,
        bundled=['norm_factors.csv', 'pf_targets.csv'])
    assert norm_factors == str(model_folder / 'norm_factors.csv')
    assert pf_targets == str(model_folder / 'pf_targets.csv')


def test_config_paths_are_the_fallback(tmp_path, monkeypatch):
    norm_factors, pf_targets, _ = _tokenizer_paths(
        tmp_path, monkeypatch, CONFIG_PATHS, bundled=[])
    assert norm_factors == CONFIG_PATHS['norm_factor_file_path']
    assert pf_targets == CONFIG_PATHS['pf_targets_file_path']


def test_missing_artifacts_are_reported(tmp_path, monkeypatch, caplog):
    with caplog.at_level(logging.WARNING, logger='terra.inference.tokenize'):
        norm_factors, pf_targets, _ = _tokenizer_paths(
            tmp_path, monkeypatch, {}, bundled=[])
    assert norm_factors is None and pf_targets is None
    assert 'gene-level norm factors will NOT be loaded' in caplog.text
    assert 'PER-FILE targets' in caplog.text
