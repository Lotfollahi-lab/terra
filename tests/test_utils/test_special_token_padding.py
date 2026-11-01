"""
Tests for ``resolve_special_token_padding``.

Inference pads the special tokens (metatokens) by default. A model trained
with ``mask.special_token_pad_ratio`` 0 -- the default, and e.g. TERRA-112M --
never saw a sequence without them, so padding them is out of distribution:
this must be warned about, and ``keep_special_tokens=True`` must keep them.
"""

import warnings

import pytest
import torch

from terra.masks.block_masking import BlockMaskCollator
from terra.utils.helper import resolve_special_token_padding


def _config(special_tokens, pad_ratio=None):
    mask = {} if pad_ratio is None else {'special_token_pad_ratio': pad_ratio}
    return {'meta': {'special_tokens': special_tokens}, 'mask': mask}


def test_no_special_tokens_is_unchanged_and_silent():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert resolve_special_token_padding(_config([])) is True


def test_always_present_special_token_is_padded_with_warning():
    # TERRA-112M: special_tokens [batch], no special_token_pad_ratio key. The
    # warning is a UserWarning, visible without any logging configuration
    with pytest.warns(UserWarning, match='out of the training distribution'):
        assert resolve_special_token_padding(_config(['batch'])) is True


def test_model_trained_with_padding_is_silent():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert resolve_special_token_padding(
            _config(['batch'], pad_ratio=0.5)) is True


def test_keep_special_tokens():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert resolve_special_token_padding(
            _config(['batch']), keep_special_tokens=True) is False


def test_collator_keeps_special_token_when_not_padded():
    # The pad flag is applied through the collator's special_token_pad_ratio
    # (1.0 when padded, 0.0 when kept).
    batch = [{'tokens': torch.tensor([5, 11, 12, 0]),
              'values': torch.tensor([3., 1., 2., 0.])}]
    for pad, expected_first_token in ((True, 0), (False, 5)):
        collator = BlockMaskCollator(
            n_targets=1, n_contexts=1, n_segments=1, seq_len_cell=3,
            seq_len_neighborhood=0, n_special_tokens=1,
            per_block_mask_ratio=0.5, sample_segments=False,
            sample_gene_masks=False,
            special_token_pad_ratio=1.0 if pad else 0.0)
        collated, _, _, masks_attention, padded = collator(
            [{k: v.clone() for k, v in b.items()} for b in batch])
        assert padded is pad
        assert collated['tokens'][0, 0].item() == expected_first_token
        assert masks_attention[0, 0, 0, 0].item() is (not pad)
