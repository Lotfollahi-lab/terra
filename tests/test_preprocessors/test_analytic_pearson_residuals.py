import numpy as np
import math
import scipy

from terra.normalizers import analytic_pearson_residuals


def test_analytic_pearson_residuals():
    """Check analytic_pearson_residuals works with a simple example"""

    x = np.column_stack((
        np.random.normal(0, math.sqrt(0), 1000),
        np.random.normal(5000, math.sqrt(50), 1000),
        np.random.normal(10000, math.sqrt(100), 1000),
        np.random.normal(40000, math.sqrt(400), 1000),
        np.random.normal(1000000, math.sqrt(1000), 1000),
        np.random.normal(5000000, math.sqrt(5000), 1000),
        np.random.normal(0, math.sqrt(0), 1000),
        np.random.normal(5000, math.sqrt(50), 1000),
        np.random.normal(10000, math.sqrt(100), 1000),
        np.random.normal(40000, math.sqrt(400), 1000),
        np.random.normal(1000000, math.sqrt(1000), 1000),
        np.random.normal(5000000, math.sqrt(5000), 1000),
    ))
    x = scipy.sparse.csr_matrix(x)

    expected_mean = np.repeat(0.0, 12)

    x_normalized = analytic_pearson_residuals(x)

    x_normalized_mean = np.array(x_normalized.mean(axis=0)).flatten()

    # analytic_pearson_residuals centres the features, so we expect the resultant
    # distribution to have a mean close to zero

    np.testing.assert_equal(np.round(x_normalized_mean, decimals=1), expected_mean)
