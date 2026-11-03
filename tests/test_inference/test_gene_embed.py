"""
Tests for the gene embedding entry points.

* ``get_average_gene_embed`` (``gene_embed_dataset`` with
  ``return_gene_per_data=True``): on CPU the cell and spatial cell count
  accumulators of a gene were views of one tensor, so after the first batch
  every batch added its counts twice and the averages were too small. GPU
  results were correct (``.cpu()`` copies there).
"""

import pickle

import numpy as np
import pytest

from terra import get_average_gene_embed

SEQ_LEN_CELL = 16


def _present_genes(tiny, bundle, n=6):
    """Genes in the index-cell segment of many cells, with their counts."""
    dataset = tiny.dataset(bundle)
    with open(bundle / 'token_dictionary.pkl', 'rb') as f:
        token_dict = pickle.load(f)
    cell_tokens = [np.asarray(dataset[i]['gene_tokens'])[:SEQ_LEN_CELL]
                   for i in range(len(dataset))]
    genes = [gene for gene in token_dict if gene.startswith('ENSG')]
    counts = {gene: sum(int(token_dict[gene] in tokens)
                        for tokens in cell_tokens) for gene in genes}
    genes = sorted(genes, key=lambda gene: -counts[gene])[:n]
    return genes, [counts[gene] for gene in genes]


@pytest.mark.parametrize('layout', ['112M', '96M'])
def test_average_gene_embeddings_do_not_depend_on_the_batch_size(
        tiny, layout):
    bundle = tiny.bundle(layout)
    dataset = tiny.dataset(bundle)
    genes, true_counts = _present_genes(tiny, bundle)

    outputs = {
        batch_size: get_average_gene_embed(
            dataset, str(bundle), cell_gene_ensembl_id=genes,
            neighborhood_gene_ensembl_id=genes, batch_size=batch_size,
            num_workers=0)
        for batch_size in (8, len(dataset))}

    for batch_size, output in outputs.items():
        for key in ('cell_gene_emb_counts_per_data',
                    'spatial_cell_gene_emb_counts_per_data'):
            np.testing.assert_array_equal(
                np.asarray(output[key]).ravel(), true_counts)
    for key in ('cell_gene_emb_average_per_data',
                'spatial_cell_gene_emb_average_per_data',
                'neighborhood_gene_emb_average_per_data'):
        np.testing.assert_allclose(
            outputs[8][key], outputs[len(dataset)][key], rtol=1e-5,
            atol=1e-6)
