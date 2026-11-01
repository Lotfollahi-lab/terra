"""
Tests for ``infer_token_distance`` with the released model layouts.

Token distances built their dataset differently from ``embed_dataset``: the
special tokens were not padded at the dataset level, so a TERRA-112M-like
model (batch metatoken) failed with ``KeyError: 'batch_value'`` on data
tokenized as in the tutorials (without special-token columns). Every
inference entry point now shares one dataset/collator setup.
"""

import numpy as np
import pytest

pytest.importorskip('geomloss')

from terra.inference import infer_token_distance  # noqa: E402

KW = dict(batch_size=8, num_workers=0, loss='energy')


@pytest.mark.parametrize('layout', ['112M', '96M'])
def test_token_distance_runs_on_tutorial_data(tiny, layout):
    bundle = tiny.bundle(layout)
    original = tiny.dataset(bundle)
    perturbed = tiny.dataset(bundle, seed=1)

    same = infer_token_distance(original, original, str(bundle), **KW)
    for key in ('cell_score', 'spatial_cell_score', 'neighborhood_score'):
        assert same[key].shape == (len(original),)
        np.testing.assert_allclose(same[key], 0.0, atol=1e-5)

    different = infer_token_distance(original, perturbed, str(bundle), **KW)
    assert np.isfinite(different['cell_score']).all()
    assert (different['neighborhood_score'] > 0).all()


def test_token_distance_ignores_special_token_columns(tiny):
    # The metatoken is padded, so the special-token columns of the data must
    # not change the distances
    bundle = tiny.bundle('112M')
    plain = infer_token_distance(
        tiny.dataset(bundle), tiny.dataset(bundle, seed=1), str(bundle), **KW)
    with_columns = infer_token_distance(
        tiny.dataset(bundle, include_special_tokens=True),
        tiny.dataset(bundle, include_special_tokens=True, seed=1),
        str(bundle), **KW)
    for key in ('cell_score', 'spatial_cell_score', 'neighborhood_score'):
        np.testing.assert_array_equal(plain[key], with_columns[key])


def test_token_distance_keep_special_tokens(tiny):
    bundle = tiny.bundle('112M')
    original = tiny.dataset(bundle, include_special_tokens=True)
    perturbed = tiny.dataset(bundle, include_special_tokens=True, seed=1)
    padded = infer_token_distance(original, perturbed, str(bundle), **KW)
    kept = infer_token_distance(original, perturbed, str(bundle),
                                keep_special_tokens=True, **KW)
    assert np.isfinite(kept['neighborhood_score']).all()
    assert not np.allclose(kept['neighborhood_score'],
                           padded['neighborhood_score'])
    with pytest.raises(ValueError, match='batch_value'):
        infer_token_distance(tiny.dataset(bundle), tiny.dataset(bundle),
                             str(bundle), keep_special_tokens=True, **KW)
