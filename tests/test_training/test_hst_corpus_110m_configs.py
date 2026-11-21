"""
The hst_corpus_110m_knn* model configs read the 16 x 512 corpus (rows of
17 x 512 = 8704 entries) with fewer neighbours or genes per cell. They give
the corpus layout as data.n_neighs / data.model_input_size, with which
inference tokenizes new data, so that inference reads new data like the
corpus rows: tokenizing for a 10-neighbour model with n_neighs 10 builds a
fresh 10-NN graph, whose neighbours differ from the first 10 of the corpus'
16-NN graph when neighbour distances tie (duplicate coordinates, grids).
"""

from pathlib import Path

import numpy as np
import pytest
import yaml
from datasets import Dataset

from terra.inference.embed import _truncation_settings
from terra.tokenizers.params import (_inference_layout_warning,
                                     _layout_mismatches)

CONFIG_FOLDER = (Path(__file__).resolve().parents[2] / 'configs' / 'model'
                 / 'hst_corpus_110m')
CONFIGS = sorted(CONFIG_FOLDER.glob('hst_corpus_110m_knn*.yaml'))
CORPUS_RECORD = {'tokenizer': 'CellGraphTokenizer', 'n_neighs': 16,
                 'model_input_size': 17 * 512}


def test_the_knn_configs_exist():
    assert [path.stem for path in CONFIGS] == [
        f'hst_corpus_110m_knn{layout}' for layout in
        ('10_seq256', '10_seq512', '16_seq256', '16_seq512', '6_seq256')]


@pytest.mark.parametrize('path', CONFIGS, ids=lambda path: path.stem)
def test_knn_config_gives_the_corpus_layout(path):
    with open(path) as f:
        data = yaml.safe_load(f)['data']
    n_neighs_read = int(path.stem.split('knn')[1].split('_')[0])
    seq_len_cell = int(path.stem.split('seq')[1])
    assert (data['n_neighs'], data['model_input_size']) == (16, 8704)
    assert data['tokenized_seq_len_cell'] == 512
    assert data['n_segments'] == n_neighs_read + 1
    assert data['seq_len_cell'] == seq_len_cell
    assert data['seq_len_neighborhood'] == n_neighs_read * seq_len_cell
    assert data['truncate_neighbors'] == (n_neighs_read < 16)
    # train() accepts the layout without a warning
    assert _layout_mismatches(data, CORPUS_RECORD) == []
    assert _inference_layout_warning(data, CORPUS_RECORD) is None
    # Inference reads corpus rows and new data tokenized with the config
    # (both 8704 entries) with the corpus stride
    rows = Dataset.from_dict({'gene_tokens': [np.zeros(8704, np.int64)]})
    assert _truncation_settings(rows, data) == (
        data['truncate_neighbors'], 512)
