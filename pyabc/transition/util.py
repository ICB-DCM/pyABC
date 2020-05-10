import numpy as np


def smart_cov(X_arr: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Create sample covariance matrix.

    Also returns a covariance if X_arr consists of only one single sample
    """
    if X_arr.shape[0] == 1:
        cov_diag = X_arr[0]
        cov = np.diag(np.absolute(cov_diag))
        return cov

    cov = np.cov(X_arr, aweights=w, rowvar=False)
    cov = np.atleast_2d(cov)
    return cov
