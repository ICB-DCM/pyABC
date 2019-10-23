import numpy as np
from typing import Callable


def pdf_norm_from_kernel(
        kernel_val: float,
        **kwargs):
    """
    Just use the pdf_max value passed via default, usually originating
    from the distance function.
    """
    return kernel_val


def pdf_norm_max_found(
        prev_pdf_norm: float,
        get_weighted_distances: Callable,
        **kwargs):
    """
    Take as pdf_max the value found so far in the history, and in
    `get_weighted_distances`.
    """
    # execute function (expensive if in calibration)
    df = get_weighted_distances()
    # extract density values
    pdfs = np.array(df['distance'])

    if prev_pdf_norm is None:
        prev_pdf_norm = - np.inf

    pdf_norm = max(prev_pdf_norm, *pdfs)

    return pdf_norm
