"""
The per-cell normalizers divide a sparse matrix by a dense column vector,
which returns a COO matrix with recent scipy (e.g. 1.17). AnnData refuses COO
matrices as layers, so tokenization with these normalizations failed. They
now return CSR matrices.
"""

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

from terra.preprocessors.normalizers import (
    normalize_by_cell_area,
    normalize_by_gene_corrected_read_depth,
    normalize_by_read_depth,
)

COUNTS = np.array([
    [0, 3, 0],
    [0, 7, 9],
    [4, 0, 4],
], dtype=np.float32)
CELL_AREAS = np.array([45.3, 23.2, 45.2])

NORMALIZERS = {
    'read_depth': (normalize_by_read_depth, {}),
    'gene_corrected_read_depth': (normalize_by_gene_corrected_read_depth, {}),
    'cell_area': (normalize_by_cell_area, {'cell_areas': CELL_AREAS}),
}


@pytest.mark.parametrize('name', NORMALIZERS)
def test_sparse_input_gives_csr_output_usable_as_layer(name):
    normalizer, kwargs = NORMALIZERS[name]
    x = sp.csr_matrix(COUNTS)

    y = normalizer(x, **kwargs)

    assert sp.issparse(y) and y.format == 'csr'
    # Same values as on dense input (scipy multiplies by the reciprocal)
    np.testing.assert_allclose(
        y.toarray(), np.asarray(normalizer(COUNTS, **kwargs)), rtol=1e-6)
    adata = ad.AnnData(X=x)
    adata.layers['X_rank'] = y
