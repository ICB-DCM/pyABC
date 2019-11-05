import numpy as np
from typing import Callable


def pdf_norm_from_kernel(
        kernel_val: float,
        **kwargs):
    """
    Just use the pdf_max value passed, usually originating from the distance
    function.
    """
    return kernel_val


def pdf_norm_max_found(
        prev_pdf_norm: float,
        get_weighted_distances: Callable,
        **kwargs):
    """
    Take as pdf_max the maximum over the values found so far in the history,
    and `get_weighted_distances`.
    """
    # execute function (expensive if in calibration)
    df = get_weighted_distances()

    # extract density values
    pdfs = np.array(df['distance'])

    # set previous normalization to dummy if not existent
    if prev_pdf_norm is None:
        prev_pdf_norm = - np.inf

    # take maximum over all normalizations
    pdf_norm = max(prev_pdf_norm, *pdfs)

    return pdf_norm
