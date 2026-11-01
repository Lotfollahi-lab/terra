"""
Tests for how the inference entry points set up the model input from the
model config, as in training.
"""

import copy

import numpy as np
import pytest

from terra import embed_dataset, gene_embed_dataset

KW = dict(batch_size=8, num_workers=0)


def _config(layout, tiny, **updates):
    config = copy.deepcopy(tiny.configs[layout])
    for section, values in updates.items():
        config[section].update(values)
    return config


def test_config_without_nz_spc(tiny):
    # Training defaults data.nz_spc to False; the inference dataset used
    # nz_spc=True, which shifted the gene segments past the model's segment
    # table (IndexError)
    config = _config('112M', tiny)
    config['data'].pop('nz_spc')
    bundle = tiny.bundle('112M_no_nz_spc', config)
    dataset = tiny.dataset(bundle)

    output = embed_dataset(dataset, str(bundle), **KW)
    assert all(np.isfinite(v).all() for v in output.values())
    gene_emb = gene_embed_dataset(
        dataset, str(bundle), cell_gene_ids=[110], return_gene=True,
        compute_cosine_with_list=['neighborhood'], neighborhood_gene_ids=[110],
        **KW)
    assert np.isfinite(gene_emb[0][110][0].numpy()).all()


def _eager(tiny, name, restrict_meta=False, restrict_mask=False):
    # Without flash attention, a query row without keys gives NaN
    config = _config(
        '112M', tiny,
        meta={'use_flash_attention': False,
              'restrict_special_attention': restrict_meta},
        mask={'restrict_special_attention': restrict_mask})
    return tiny.bundle(name, config)


def test_restrict_special_attention_with_padded_special_tokens(tiny):
    reference = _eager(tiny, 'eager')
    restricted = _eager(tiny, 'eager_restricted', restrict_meta=True,
                        restrict_mask=True)
    dataset = tiny.dataset(reference)

    expected = embed_dataset(dataset, str(reference), **KW)
    output = embed_dataset(dataset, str(restricted), **KW)
    # The padded metatoken is not restricted (it has nothing to attend to,
    # and no gene token attends to it): finite and equal embeddings
    for key, value in expected.items():
        assert np.isfinite(output[key]).all(), key
        np.testing.assert_array_equal(output[key], value)


def test_restrict_special_attention_is_read_from_mask(tiny):
    reference = _eager(tiny, 'eager')
    # As in training, the `meta` value is ignored
    meta_only = _eager(tiny, 'eager_meta_restricted', restrict_meta=True)
    restricted = _eager(tiny, 'eager_mask_restricted', restrict_mask=True)
    dataset = tiny.dataset(reference, include_special_tokens=True)

    def embed(bundle):
        return embed_dataset(dataset, str(bundle), keep_special_tokens=True,
                             **KW)['cell_emb']

    expected = embed(reference)
    np.testing.assert_array_equal(embed(meta_only), expected)
    # With the metatoken kept, it attends only to itself as in training
    output = embed(restricted)
    assert np.isfinite(output).all()
    assert not np.allclose(output, expected)


def _cell_neigh_config(tiny):
    # Equal split (16/16); n_segments kept from a cell-graph config, so the
    # encoder's layout-derived index-cell length is 32 // 4 = 8
    return _config('112M', tiny, data={
        'tokenizer_type': 'cell_neigh', 'model_input_size': 32,
        'seq_len_cell': 16, 'seq_len_neighborhood': 16, 'n_segments': 4})


def _drop_neighborhood(dataset, seq_len_cell):
    """Pad the neighborhood tokens, keep the index cell."""
    def edit(example):
        for key in ('gene_tokens', 'gene_expr'):
            values = list(example[key])
            example[key] = (values[:seq_len_cell]
                            + [0] * (len(values) - seq_len_cell))
        return example

    edited = dataset.with_format(None).map(edit)
    edited.set_format(
        type='torch', output_all_columns=True,
        columns=[c for c in edited.column_names if c != 'cell_id'])
    return edited


def test_cell_only_context_uses_configured_seq_len_cell(tiny):
    bundle = tiny.bundle('cell_neigh', _cell_neigh_config(tiny))
    dataset = tiny.dataset(bundle)

    output = embed_dataset(dataset, str(bundle), **KW)
    without_neighborhood = embed_dataset(
        _drop_neighborhood(dataset, 16), str(bundle), **KW)
    # The cell embedding is computed from the index cell alone
    np.testing.assert_allclose(without_neighborhood['cell_emb'],
                               output['cell_emb'], rtol=1e-5, atol=1e-6)
    assert not np.allclose(without_neighborhood['spatial_cell_emb'],
                           output['spatial_cell_emb'])


@pytest.mark.parametrize('api_version', ['v4', None])
def test_api_version_v4_or_missing(tiny, api_version):
    # Training builds the model with api_version v4 unless set; v3 and v4
    # are the same wrapped model
    config = _config('96M', tiny)
    if api_version is None:
        config['meta'].pop('api_version')
    else:
        config['meta']['api_version'] = api_version
    bundle = tiny.bundle(f'96M_{api_version}', config)
    reference = tiny.bundle('96M')
    dataset = tiny.dataset(reference)

    expected = embed_dataset(dataset, str(reference), **KW)
    output = embed_dataset(dataset, str(bundle), **KW)
    for key, value in expected.items():
        np.testing.assert_array_equal(output[key], value)


def _truncating_config(tiny, tokenized_seq_len_cell):
    # The model reads the top seq_len_cell (16) genes of each cell of a
    # pretraining dataset tokenized with tokenized_seq_len_cell genes per cell
    return _config('112M', tiny, data={
        'truncate_neighbors': True,
        'tokenized_seq_len_cell': tokenized_seq_len_cell})


def test_truncation_config_on_data_tokenized_for_the_model(tiny):
    # tokenize_adata tokenizes model_input_size // (n_neighs + 1) = 16 genes
    # per cell; reading them with the pretraining stride 32 raised
    bundle = tiny.bundle('112M_truncating', _truncating_config(tiny, 32))
    reference = tiny.bundle('112M')
    dataset = tiny.dataset(reference)

    expected = embed_dataset(dataset, str(reference), **KW)
    output = embed_dataset(dataset, str(bundle), **KW)
    for key, value in expected.items():
        np.testing.assert_array_equal(output[key], value)


def test_truncation_config_on_pretraining_layout(tiny):
    # Data tokenized with 32 genes per cell, as the pretraining dataset
    wide = tiny.bundle('112M_wide_tokenizer', _config(
        '112M', tiny, data={'model_input_size': 128}))
    wide_dataset = tiny.dataset(wide)
    assert len(wide_dataset[0]['gene_tokens']) == 128

    bundle = tiny.bundle('112M_truncating', _truncating_config(tiny, 32))
    reference = tiny.bundle('112M')
    expected = embed_dataset(tiny.dataset(reference), str(reference), **KW)
    output = embed_dataset(wide_dataset, str(bundle), **KW)
    # The top 16 of 32 genes per cell are the top 16 genes
    for key, value in expected.items():
        np.testing.assert_allclose(output[key], value, rtol=1e-5, atol=1e-6)
