import numpy as np
from typing import Callable, Union
import pandas as pd


def pdf_norm_from_kernel(
        kernel_val: float,
        **kwargs):
    """
    Just use the pdf_max value passed, usually originating from the distance
    function.
    """
    return kernel_val


def pdf_norm_max_found(
        prev_pdf_norm: Union[float, None],
        get_weighted_distances: Callable[[], pd.DataFrame],
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


class ScaledPDFNorm:
    """
    Finds the previously found maximum density value, but then scales it
    by a factor `factor**T` such that the probability of particles getting
    accepted is increased by y value of up to `factor`.

    Some additional rules are applied to make the scheme stable. The scaling
    is in particular applied only after a minimum acceptance rate has been
    violated.

    Parameters
    ----------
    factor:
        The factor by which to effectively rescale the acceptance step's
        normalization constant.
    alpha:
        The ratio by which the subsequent temperature is assumed to be
        reduced relative to the current one. This is only accurate if a
        pyabc.ExponentialDecayFixedRatioScheme with corresponding ratio
        is employed.
    min_acceptance_rate:
        The scaling is applied once the acceptance rates fall below this
        value for the first time.
    """

    def __init__(
            self,
            factor: float = 10,
            alpha: float = 0.5,
            min_acceptance_rate: bool = 0.1):
        self.factor = 10
        self.alpha = alpha
        self.min_acceptance_rate = min_acceptance_rate
        self._hit = False

    def __call__(
            self,
            prev_pdf_norm: Union[float, None],
            get_weighted_distances: Callable[[], pd.DataFrame],
            prev_temp: Union[float, None],
            acceptance_rate: float,
            **kwargs):
        # base: the maximum found temperature
        pdf_norm = pdf_norm_max_found(
            prev_pdf_norm=prev_pdf_norm,
            get_weighted_distances=get_weighted_distances)

        # log-scale
        offset = np.log(self.factor)

        if acceptance_rate >= self.min_acceptance_rate and not self._hit:
            # do not apply scaling yet since acceptance rates still feasible
            return pdf_norm

        # from now on rescale
        self._hit = True

        if prev_temp is None:
            # can't take temperature into account, thus effectively assume T=1
            next_temp = 1
        else:
            # note: this is only accurate if the temperature is based on a
            #  ExponentialDecayFixedRatioScheme with the given alpha value
            next_temp = self.alpha * prev_temp

        # the offset is multiplied by the next temperature so that the
        #  effective resulting factor in the acceptance step is as intended
        scaled_norm = pdf_norm - offset * next_temp

        # used_norm = max(prev_pdf_norm, used_norm)
        return scaled_norm
