import numpy as np

from terra.normalizers import mean_normalize_by_gene


def test_mean_normalize_by_gene():
    """Check mean_normalize works with a simple example"""

    x = np.array([
        [0, 3, 0],
        [0, 7, 9],
        [4, 0, 4],
        [0, 0, 3],
        [8, 0, 0],
    ])

    x_expected = np.array([
        [0, 3 / 2, 0],
        [0, 7 / 2, 9 / 3.2],
        [4 / 2.4, 0, 4 / 3.2],
        [0, 0, 3 / 3.2],
        [8 / 2.4, 0, 0],
    ])

    x_normalized = mean_normalize_by_gene(x)

    np.testing.assert_array_equal(x_expected, x_normalized)


def test_mean_normalize_by_gene_zero_sum():
    """Check mean_normalize works where the mean for a gene is zero"""

    x = np.array([
        [0, 3],
        [0, 7],
        [0, 0],
        [0, 0],
        [0, 0],
    ])

    x_expected = np.array([
        [np.NaN, 3 / 2],
        [np.NaN, 7 / 2],
        [np.NaN, 0],
        [np.NaN, 0],
        [np.NaN, 0],
    ])

    x_normalized = mean_normalize_by_gene(x)

    np.testing.assert_array_equal(x_expected, x_normalized)
