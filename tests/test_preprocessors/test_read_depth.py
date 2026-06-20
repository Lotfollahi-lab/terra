import numpy as np
import scipy

from terra.normalizers import read_depth


def test_read_depth():
    """Test read_depth works with a simple example"""

    x = np.array([
        [0, 3, 0],
        [0, 7, 9],
        [4, 0, 4],
        [0, 0, 3],
        [8, 0, 0],
    ])
    x = scipy.sparse.csr_matrix(x)

    x_expected = np.array([
        [0 / 3, 3 / 3, 0 / 3],
        [0 / 16, 7 / 16, 9 / 16],
        [4 / 8, 0 / 8, 4 / 8],
        [0 / 3, 0 / 3, 3 / 3],
        [8 / 8, 0 / 8, 0 / 8],
    ]) * 10_000

    x_normalized = read_depth(x).toarray()

    assert np.testing.assert_allclose(x_normalized, x_expected)
