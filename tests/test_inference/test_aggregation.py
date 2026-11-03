"""
Tests for the aggregation of gene embeddings into cell and neighborhood
embeddings.
"""

import pickle

import numpy as np
import pytest
import torch

from terra import embed_dataset, get_gene_embed
from terra.datasets.cell_datasets import init_cell_dataset
from terra.utils.embedding import compute_softmax_weighted_mean_emb

KW = dict(batch_size=8, num_workers=0)


def test_softmax_weighted_mean_over_mask():
    generator = torch.Generator().manual_seed(0)
    emb = torch.randn(3, 5, 4, generator=generator)
    logits = torch.randn(3, 5, generator=generator)
    mask = torch.tensor([[True, True, False, True, False],
                         [False, True, True, True, True],
                         [False] * 5])

    output = compute_softmax_weighted_mean_emb(emb, logits, mask=mask)
    for i in range(2):
        weights = torch.softmax(logits[i][mask[i]], dim=0)
        expected = (emb[i][mask[i]] * weights[:, None]).sum(0)
        torch.testing.assert_close(output[i], expected)
    # No selected position: zero embedding (as for agg_type 'avg')
    assert torch.equal(output[2], torch.zeros(4))
    # Without a mask, the softmax runs over all positions (unchanged)
    expected = (emb * torch.softmax(logits, dim=1)[..., None]).sum(1)
    torch.testing.assert_close(
        compute_softmax_weighted_mean_emb(emb, logits), expected)


def test_embed_softmax_aggregation_applies_softmax_once(tiny):
    # agg_type='softmax' is a count-softmax-weighted mean over the index-cell
    # genes; it applied the softmax twice, which put most of the weight on
    # neighborhood and padding positions
    bundle = tiny.bundle('112M')
    dataset = tiny.dataset(bundle)
    config = tiny.configs['112M']
    output = embed_dataset(dataset, str(bundle), agg_type='softmax',
                           return_token_embeddings=True, **KW)

    with open(bundle / 'token_dictionary.pkl', 'rb') as f:
        vocab_size = len(pickle.load(f))
    cell_dataset = init_cell_dataset(
        tokenizer_type='cell_graph', dataset=dataset, vocab_size=vocab_size,
        seq_len_cell=config['data']['seq_len_cell'],
        seq_len_neighborhood=config['data']['seq_len_neighborhood'],
        gt_type='combined', cell_pos_enc='segment', special_tokens=['batch'],
        sampling_strategy=None, n_nonzero_tokens_list=[], include_cell_id=True,
        nz_spc=True, pad_special_tokens=True)
    seq_len_cell = config['data']['seq_len_cell']
    for i in range(5):
        item = cell_dataset[i]
        tokens, values = item['tokens'][1:], item['values'][1:].float()
        selected = torch.zeros_like(tokens, dtype=torch.bool)
        selected[:seq_len_cell] = tokens[:seq_len_cell] != 0
        weights = torch.softmax(values[selected], dim=0).numpy()
        expected = (output['token_emb'][i][selected.numpy()]
                    * weights[:, None]).sum(0)
        np.testing.assert_allclose(output['spatial_cell_emb'][i], expected,
                                   rtol=1e-4, atol=1e-5)


def test_get_gene_embed_with_duplicate_genes(tiny):
    # A gene listed twice raised KeyError; it is returned once
    bundle = tiny.bundle('112M')
    dataset = tiny.dataset(bundle)
    genes = [f'ENSG{i + 1:011d}' for i in range(3)]
    kwargs = dict(include_spatial_cell_emb=True, **KW)
    expected = get_gene_embed(dataset, str(bundle), cell_gene_ensembl_id=genes,
                              neighborhood_gene_ensembl_id=genes[1:], **kwargs)
    output = get_gene_embed(
        dataset, str(bundle), cell_gene_ensembl_id=genes + genes[:1],
        neighborhood_gene_ensembl_id=genes[1:] + genes[1:], **kwargs)
    assert list(output) == list(expected)
    for key, value in expected.items():
        np.testing.assert_array_equal(output[key], value)



def test_unknown_excluded_genes_raise_a_clear_error(tiny):
    bundle = tiny.bundle('112M')
    dataset = tiny.dataset(bundle)
    with open(bundle / 'token_dictionary.pkl', 'rb') as f:
        token_dict = pickle.load(f)
    gene = next(key for key in token_dict if key.startswith('ENSG'))
    with pytest.raises(ValueError, match='ENSG_UNKNOWN') as error:
        embed_dataset(dataset, str(bundle),
                      agg_excluded_genes=[gene, 'ENSG_UNKNOWN', 107], **KW)
    assert '107' in str(error.value)
    assert gene not in str(error.value)
