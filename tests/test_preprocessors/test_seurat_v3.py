import numpy as np
import math
from random import gauss
import scipy

from terra.preprocessors import normalize_by_seurat


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

    x_normalized = normalize_by_seurat(x)

    x_normalized_mean = np.array(x_normalized.mean(axis=0)).flatten()
    x_normalized_std = np.array(x_normalized.std(axis=0)).flatten()

    # seurat_v3 centers and scales the features, so the resulting features
    # should have a mean close to zero and a standard deviation close to one

    np.testing.assert_equal(np.round(x_normalized_mean, decimals=1), expected_mean)
    np.testing.assert_equal(np.round(x_normalized_std, decimals=1), expected_std)


def _seurat_reference(x: np.ndarray, span: float) -> np.ndarray:
    """Seurat v3 standardization with an explicit LOESS span."""
    from skmisc.loess import loess
    gene_means = x.mean(axis=0)
    gene_vars = x.var(axis=0)
    not_const = gene_vars > 0
    model = loess(np.log10(gene_means[not_const]),
                  np.log10(gene_vars[not_const]), span=span, degree=2)
    model.fit()
    expected_log_variances = np.zeros(x.shape[1])
    expected_log_variances[not_const] = model.outputs.fitted_values
    return (x - gene_means) / np.sqrt(10 ** expected_log_variances)


def _counts(seed: int = 0) -> np.ndarray:
    # 200 genes spanning a range of means, so the LOESS span matters
    rng = np.random.default_rng(seed)
    means = np.exp(rng.uniform(-3, 3, 200))
    return rng.negative_binomial(5, 5 / (5 + means), (300, 200)).astype(float)


def test_seurat_v3_default_span_unchanged():
    # The default reproduces the previous output, which (because the
    # intended span 0.3 was passed in a way skmisc ignores) used span 0.75.
    x = _counts()
    np.testing.assert_allclose(
        normalize_by_seurat(scipy.sparse.csr_matrix(x)),
        _seurat_reference(x, span=0.75))


def test_seurat_v3_span_is_applied():
    x = _counts()
    y_03 = normalize_by_seurat(scipy.sparse.csr_matrix(x), loess_span=0.3)
    np.testing.assert_allclose(y_03, _seurat_reference(x, span=0.3))
    assert not np.allclose(y_03, normalize_by_seurat(
        scipy.sparse.csr_matrix(x)))
