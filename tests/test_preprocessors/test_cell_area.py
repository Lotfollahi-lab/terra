import numpy as np
import scipy

from terra.normalizers import cell_area


def test_cell_area():
    """Test cell_area works with a simple example"""

    cell_areas = np.array([45.3, 23.2, 45.2, 24.2, 83.1])

    x = np.array([
        [0, 3, 0],
        [0, 7, 9],
        [4, 0, 4],
        [0, 0, 3],
        [8, 0, 0],
    ])
    x = scipy.sparse.csr_matrix(x)

    x_expected = np.array([
        [0 / 45.3, 3 / 45.3, 0 / 45.3],
        [0 / 23.2, 7 / 23.2, 9 / 23.2],
        [4 / 45.2, 0 / 45.2, 4 / 45.2],
        [0 / 24.2, 0 / 24.2, 3 / 24.2],
        [8 / 83.1, 0 / 83.1, 0 / 83.1],
    ])

    x_normalized = cell_area(x, cell_areas=cell_areas).toarray()

    assert np.testing.assert_allclose(x_normalized, x_expected)
