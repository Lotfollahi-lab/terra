"""
Tests for ``infer``, the inference entry point for training configs.
"""

import copy

import numpy as np
import pytest

from terra import embed_dataset, infer

KW = dict(batch_size=8, num_workers=0)


def _infer_args(tiny, bundle, config, tmp_path):
    """Training-style args for `bundle`, with the metadata AnnData folder."""
    raw = tmp_path / 'raw'
    raw.mkdir()
    tiny.harmonized(bundle).write_h5ad(raw / 'S1.h5ad')
    args = copy.deepcopy(config)
    args['data'].update({
        'token_dict_folder_path': str(bundle / 'token_dictionary.pkl'),
        'raw_data_folder_path': str(raw), 'batch_size': 8,
        'pin_memory': False, 'num_workers': 0})
    args['state'] = {'read_checkpoint': 'model_checkpoint.pt',
                     'write_tag': 'terra'}
    return args


def _assert_matches_embed_dataset(adata, dataset, bundle, layer):
    expected = embed_dataset(dataset, str(bundle), **KW)
    assert list(adata.obs_names) == list(dataset['cell_id'])
    for name in ('cell_emb', 'neighborhood_emb'):
        np.testing.assert_allclose(adata.obsm[f'{name}_layer_{layer}'],
                                   expected[name], rtol=1e-5, atol=1e-6)


def test_infer_cell_neigh_model(tiny, tmp_path):
    # infer() compared the tokenizer type to 'cell_neighborhood' (the value
    # is 'cell_neigh'), so cell-neighborhood models had no neighborhood mask
    config = copy.deepcopy(tiny.configs['112M'])
    config['data'].update({
        'tokenizer_type': 'cell_neigh', 'model_input_size': 32,
        'seq_len_cell': 16, 'seq_len_neighborhood': 16, 'n_segments': 2})
    bundle = tiny.bundle('cell_neigh', config)
    dataset = tiny.dataset(bundle)
    adata = infer(args=_infer_args(tiny, bundle, config, tmp_path),
                  dataset=dataset, load_folder_path=str(bundle),
                  include_spatial_cell_emb=True)
    _assert_matches_embed_dataset(adata, dataset, bundle, layer=2)


@pytest.mark.parametrize('layout', ['112M', '96M'])
def test_infer_default_arguments(tiny, tmp_path, layout):
    # include_spatial_cell_emb=False (the default) raised UnboundLocalError
    config = tiny.configs[layout]
    bundle = tiny.bundle(layout)
    dataset = tiny.dataset(bundle)
    adata = infer(args=_infer_args(tiny, bundle, config, tmp_path),
                  dataset=dataset, load_folder_path=str(bundle))
    assert 'spatial_cell_emb_layer_2' not in adata.obsm
    _assert_matches_embed_dataset(adata, dataset, bundle, layer=2)
