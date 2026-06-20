import numpy as np

from terra.normalizers import non_zero_median


def test_non_zero_median():
    """Check non_zero_median works with a simple example"""

    x = np.array([
        [0, 3, 0],
        [0, 7, 9],
        [4, 0, 4],
        [0, 0, 3],
        [8, 0, 0],
    ])

    x_expected = np.array([
        [0 / 6, 3 / 5, 0 / 4],
        [0 / 6, 7 / 5, 9 / 4],
        [4 / 6, 0 / 5, 4 / 4],
        [0 / 6, 0 / 5, 3 / 4],
        [8 / 6, 0 / 5, 0 / 4],
    ])

    x_normalized = non_zero_median(x)

    np.testing.assert_array_equal(x_expected, x_normalized)
