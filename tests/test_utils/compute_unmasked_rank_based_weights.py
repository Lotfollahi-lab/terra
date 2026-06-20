import unittest

import torch

from terra.utils import compute_unmasked_rank_based_weights


class TestComputeRankBasedWeights(unittest.TestCase):
    def test_basic_case(self):
        tokens = torch.tensor([[1, 3, 2],
                               [4, 5, 0]])
        mask = torch.tensor([[1, 1, 1],
                             [1, 1, 0]])
        expected_weights = torch.tensor([[0.5, 0.3333, 0.1667],
                                         [0.6667, 0.3333, 0.0000]])
        computed_weights = compute_unmasked_rank_based_weights(tokens, mask)
        print(computed_weights)
        torch.testing.assert_close(computed_weights,
                                   expected_weights,
                                   atol=1e-4,
                                   rtol=1e-5)

    def test_all_zero_case(self):
        tokens = torch.tensor([[0, 0, 0],
                               [0, 0, 0]])
        mask = torch.tensor([[0, 0, 0],
                             [0, 0, 0]])
        expected_weights = torch.tensor([[0.0, 0.0, 0.0],
                                         [0.0, 0.0, 0.0]])
        computed_weights = compute_unmasked_rank_based_weights(tokens, mask)
        print(computed_weights)
        torch.testing.assert_close(computed_weights,
                                   expected_weights,
                                   atol=1e-4,
                                   rtol=1e-5)

    def test_single_non_zero_case(self):
        tokens = torch.tensor([[1, 0, 0], [0, 0, 0]])
        mask = torch.tensor([[1, 0, 0],
                             [0, 0, 0]])
        expected_weights = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        computed_weights = compute_unmasked_rank_based_weights(tokens, mask)
        print(computed_weights)
        torch.testing.assert_close(computed_weights,
                                   expected_weights,
                                   atol=1e-4,
                                   rtol=1e-5)

    def test_complex_case(self):
        tokens = torch.tensor([[1, 5, 0, 0],
                               [0, 0, 0, 0],
                               [1, 5, 7, 6],
                               [1, 111, 3, 0]])
        mask = torch.tensor([[0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 1, 1],
                             [0, 0, 1, 0]])
        expected_weights = torch.tensor([[0.0000, 0.0000, 0.0000, 0.0000],
                                         [0.0000, 0.0000, 0.0000, 0.0000],
                                         [0.0000, 0.0000, 0.6667, 0.3333],
                                         [0.0000, 0.0000, 1.0, 0.0000]])
        computed_weights = compute_unmasked_rank_based_weights(tokens, mask)
        print(computed_weights)
        torch.testing.assert_close(computed_weights,
                                   expected_weights,
                                   atol=1e-4,
                                   rtol=1e-5)

if __name__ == '__main__':
    unittest.main()
