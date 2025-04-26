
def compute_energy_distance(
    x: np.ndarray,
    y: np.ndarray
) -> float:
    """
    Compute the energy distance using squared Euclidean distances between two multi-dimensional samples.

    Parameters
    ----------
    x : np.ndarray
        Shape (n_samples, n_features) -- first distribution.
    y : np.ndarray
        Shape (n_samples, n_features) -- second distribution.

    Returns
    -------
    e_distance : float
        Energy distance between the two distributions.
    """
    sigma_X = pairwise_distances(x, x, metric="sqeuclidean").mean()
    sigma_Y = pairwise_distances(y, y, metric="sqeuclidean").mean()
    delta = pairwise_distances(x, y, metric="sqeuclidean").mean()
    return 2 * delta - sigma_X - sigma_Y

def compute_maximum_mean_discrepancy(
    x: np.ndarray,
    y: np.ndarray,
    gamma: float = 1.0
) -> float:
    """
    Compute the Maximum Mean Discrepancy (MMD) using the RBF kernel between two samples.

    Parameters
    ----------
    x : np.ndarray
        Shape (n_samples, n_features) -- first distribution.
    y : np.ndarray
        Shape (n_samples, n_features) -- second distribution.
    gamma : float
        RBF kernel bandwidth parameter.

    Returns
    -------
    mmd : float
        Maximum mean discrepancy between the two distributions.
    """
    xx = rbf_kernel(x, x, gamma)
    xy = rbf_kernel(x, y, gamma)
    yy = rbf_kernel(y, y, gamma)
    return xx.mean() + yy.mean() - 2 * xy.mean()

def compute_scalar_mmd(
    x: np.ndarray,
    y: np.ndarray,
    gammas: list[float] = None
) -> float:
    """
    Compute scalar MMD as an average across multiple RBF bandwidths.

    Parameters
    ----------
    x : np.ndarray
        First sample array.
    y : np.ndarray
        Second sample array.
    gammas : list of float, optional
        List of RBF kernel bandwidths. Defaults to [2, 1, 0.5, 0.1, 0.01, 0.005].

    Returns
    -------
    mmd_value : float
        Averaged MMD value across all specified gamma values.
    """
    if gammas is None:
        gammas = [2, 1, 0.5, 0.1, 0.01, 0.005]
    mmds = [compute_maximum_mean_discrepancy(x, y, gamma=g) for g in gammas]
    return np.nanmean(mmds)

def compute_emd(
    x: np.ndarray,
    y: np.ndarray
) -> float:
    """
    Compute the 1D Earth Mover's Distance (Wasserstein-1) between two multidimensional samples
    by averaging the EMD computed on each feature dimension.

    Parameters
    ----------
    x : np.ndarray
        Shape (n_samples, n_features) -- first distribution.
    y : np.ndarray
        Shape (n_samples, n_features) -- second distribution.

    Returns
    -------
    emd_value : float
        The average 1D EMD across all feature dimensions.
    """
    from scipy.stats import wasserstein_distance
    emds = []
    # Compute 1D EMD for each feature
    for dim in range(x.shape[1]):
        emd_dim = wasserstein_distance(x[:, dim], y[:, dim])
        emds.append(emd_dim)
    return float(np.mean(emds))

