import numpy as np
import scipy

from nichejepa.aggregators import aggregate_by_sum_of_neighbours


def test_aggregate_by_sum_of_neighbours():
    """Check aggregate_by_radius works with a simple example"""

    x = np.array([
        [0, 3, 0, 4, 0, 0],
        [0, 7, 9, 8, 3, 2],
        [4, 0, 4, 6, 8, 0],
        [0, 0, 3, 0, 6, 4],
        [8, 0, 0, 4, 6, 0],
    ])
    x = scipy.sparse.csr_matrix(x)

    coordinates = np.array([
        [0, 1],
        [5, 3],
        [1, 2],
        [7, 9],
        [8, 9]
    ])

    radius = 2

    expected_aggregation = np.array([
        [4, 3, 4, 10,  8, 0],
        [0, 7, 9,  8,  3, 2],
        [4, 3, 4, 10,  8, 0],
        [8, 0, 3,  4, 12, 4],
        [8, 0, 3,  4, 12, 4],
    ])

    aggregation = aggregate_by_sum_of_neighbours(
        x=x,
        coordinates=coordinates,
        radius=radius,
    ).toarray()

    np.testing.assert_array_equal(aggregation, expected_aggregation)
