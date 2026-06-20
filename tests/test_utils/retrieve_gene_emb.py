import unittest

import torch

from terra.utils import retrieve_gene_emb


class TestRetrieveGeneEmb(unittest.TestCase):
    def test_basic_cell_case(self):
        tokens = torch.tensor([[1, 3, 0, 5, 4, 3],
                               [3, 5, 0, 2, 3, 0]])
        emb = torch.tensor([
            [[0.2, 0.4], [0.3, 0.1], [0.0, 0.0], [0.1, 0.3], [0.5, 0.4], [0.2, 0.6]],
            [[0.4, 0.1], [0.2, 0.2], [0.0, 0.0], [0.2, 0.5], [0.3, 0.1], [0.0, 0.0]]])
        expected_emb = torch.tensor(
            [[0.3, 0.1],
             [0.4, 0.1]])
        computed_emb = retrieve_gene_emb(
            tokens=tokens,
            seq_len_cell=3,
            has_cls=False,
            emb=emb,
            gene_type="cell",
            gene_id=3)
        print(computed_emb)
        torch.testing.assert_close(computed_emb,
                                   expected_emb)

    def test_sparsegene_neighborhood_case(self):
        tokens = torch.tensor([[1, 3, 0, 5, 4, 3],
                               [3, 5, 0, 2, 3, 0],
                               [8, 4, 2, 4, 1, 0]])
        emb = torch.tensor([
            [[0.2, 0.4], [0.3, 0.1], [0.0, 0.0], [0.1, 0.3], [0.5, 0.4], [0.2, 0.6]],
            [[0.4, 0.1], [0.2, 0.2], [0.0, 0.0], [0.2, 0.5], [0.3, 0.1], [0.0, 0.0]],
            [[0.1, 0.5], [0.4, 0.6], [0.2, 0.3], [0.1, 0.1], [0.2, 0.5], [0.0, 0.0]]])
        expected_emb = torch.tensor(
            [[0.5, 0.4],
             [0.0, 0.0],
             [0.1, 0.1]])
        computed_emb = retrieve_gene_emb(
            tokens=tokens,
            seq_len_cell=3,
            has_cls=False,
            emb=emb,
            gene_type="neighborhood",
            gene_id=4)
        print(computed_emb)
        torch.testing.assert_close(computed_emb,
                                   expected_emb)

if __name__ == '__main__':
    unittest.main()
