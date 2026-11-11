"""
``BlockMaskCollator(min_context_genes=...)``: at mask ratio 0.6 a cell with 1-2
gene tokens has all of them as targets (ceil(0.6 K) = K), so with padded
metatokens it has no context. With ``min_context_genes`` a cell with more gene
tokens than that keeps that many in the context.
"""

import pytest
import torch

from terra.masks.block_masking import BlockMaskCollator

N_SPECIAL_TOKENS = 1


def _collator(**kwargs):
    return BlockMaskCollator(
        n_targets=1, n_contexts=1, n_segments=1, seq_len_cell=8,
        seq_len_neighborhood=0, n_special_tokens=N_SPECIAL_TOKENS,
        per_block_mask_ratio=0.6, **kwargs)


def _tokens(n_genes):
    tokens = torch.zeros(N_SPECIAL_TOKENS + 8, dtype=torch.long)
    tokens[0] = 5  # metatoken
    tokens[N_SPECIAL_TOKENS:N_SPECIAL_TOKENS + n_genes] = torch.arange(
        10, 10 + n_genes)
    return tokens


def _n_genes(masks):
    """Gene tokens (not the metatoken) of a cell's mask."""
    return int((masks[0] >= N_SPECIAL_TOKENS).sum())


@pytest.mark.parametrize('n_genes, default, with_min', [
    # (targets, context) per cell
    (1, (1, 0), (1, 0)),   # cannot keep a context gene and have a target
    (2, (2, 0), (1, 1)),
    (3, (2, 1), (2, 1)),
    (8, (5, 3), (5, 3)),
])
def test_min_context_genes_keeps_a_context_gene(n_genes, default, with_min):
    for collator, expected in ((_collator(), default),
                               (_collator(min_context_genes=1), with_min)):
        torch.manual_seed(0)
        targets, context = collator._sample_gene_mask(
            _tokens(n_genes), pad_special_tokens=True)
        assert (_n_genes(targets), _n_genes(context)) == expected


def test_default_masks_are_unchanged():
    torch.manual_seed(0)
    default = _collator()._sample_gene_mask(_tokens(8), False)
    torch.manual_seed(0)
    explicit = _collator(min_context_genes=0)._sample_gene_mask(
        _tokens(8), False)
    for a, b in zip(default, explicit):
        assert all(torch.equal(x, y) for x, y in zip(a, b))
