import numpy as np
import math
from random import gauss
import scipy

from nichejepa.normalizers import seurat_v3


def test_seurat_v3():
    """Test seurat_v3 works with a simple example"""

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

    expected_std = np.array([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    expected_mean = np.repeat(0.0, 12)

    x_normalized = seurat_v3(x)

    x_normalized_mean = np.array(x_normalized.mean(axis=0)).flatten()
    x_normalized_std = np.array(x_normalized.std(axis=0)).flatten()

    # seurat_v3 centers and scales the features, so the resulting features
    # should have a mean close to zero and a standard deviation close to one

    np.testing.assert_equal(np.round(x_normalized_mean, decimals=1), expected_mean)
    np.testing.assert_equal(np.round(x_normalized_std, decimals=1), expected_std)
