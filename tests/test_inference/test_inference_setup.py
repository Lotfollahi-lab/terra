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
