import unittest

import torch

from terra.utils import compute_mean_unmasked_emb


class TestComputeMeanUnmaskedEmb(unittest.TestCase):
    def test_mean_unmasked_emb_3d(self):
        # Test case for 3D tensor
        emb_3d = torch.tensor([[[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]],
                               [[7.0, 8.0, 9.0], [1.0, 2.0, 1.0]]],
                              dtype=torch.float32)
        mask_3d = torch.tensor([[1, 0], [1, 1]], dtype=torch.bool)
        expected_mean_3d = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 5.0]],
                                        dtype=torch.float32)
        result_3d = compute_mean_unmasked_emb(emb_3d, mask_3d)
        self.assertTrue(torch.allclose(result_3d, expected_mean_3d),
                        f"Expected {expected_mean_3d}, but got {result_3d}.")

        # Additional test case for a larger 3D tensor
        emb_3d_large = torch.tensor([
            [[1.0, 2.0, 3.0, 4.0], [0.0, -10.0, 1.0, 0.0], [1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
            [[7.0, 8.0, 9.0, 12.0], [1.0, 2.0, 1.0, 2.0], [-12.0, 12.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
            [[5.0, 5.0, 5.0, 5.0], [5.0, 5.0, 5.0, 5.0], [0.0, 0.0, 2.0, 23.0], [4.0, 12.0, 2.0, 31.0], [1.0, 1.0, 1.0, 1.0]],
            [[3.0, 3.0, 3.0, 3.0], [3.0, 3.0, 3.0, 3.0], [4.0, 4.0, 4.0, 4.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
        ], dtype=torch.float32)
        mask_3d_large = torch.tensor([
            [1, 0, 1, 1, 1], # 4 valid rows
            [1, 1, 0, 1, 1], # 4 valid rows
            [1, 1, 0, 0, 1], # 3 valid rows
            [1, 1, 0, 0, 0]  # 4 valid rows
        ], dtype=torch.bool)
        expected_mean_3d_large = torch.tensor([
            [1.0, 1.25, 1.5, 1.5],
            [2.5, 3.0, 3.0, 4.0],
            [3.6667, 3.6667, 3.6667, 3.6667],
            [3.0, 3.0, 3.0, 3.0]
        ], dtype=torch.float32)
        result_3d_large = compute_mean_unmasked_emb(emb_3d_large,
                                                    mask_3d_large)
        self.assertTrue(torch.allclose(result_3d_large, expected_mean_3d_large,
                                       atol=1e-2),
                        f'Expected {expected_mean_3d_large}, '
                        f'but got {result_3d_large}.')

if __name__ == "__main__":
    unittest.main()

