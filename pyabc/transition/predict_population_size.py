import logging
from collections import namedtuple

from pyabc.cv.powerlaw import fitpowerlaw

logger = logging.getLogger("CV Estimation")

CVEstimate = namedtuple("CVEstimate", "n_estimated n_samples_list cvs f popt")


def predict_population_size(current_poop_size: int,
                            target_cv: float, calc_cv,
                            n_steps=10, first_step_factor=3) -> CVEstimate:
    """
    Estimate the required nr of particles for a target coefficient of
    variation.

    Parameters
    ----------
    current_pop_size: int
    target_cv: float
    calc_cv: A function mapping population_size -> cv

    Returns
    -------

    suggested_pop_size: int
    """
    if current_poop_size == 1:
        return CVEstimate(1, [], [], None, None)

    start = max(current_poop_size // first_step_factor, 1)
    stop = current_poop_size * 2
    step = max(current_poop_size // n_steps, 1)

    n_samples_list = list(range(start, stop, step))
    cvs = list(map(calc_cv, n_samples_list))

    try:
        popt, f, finv = fitpowerlaw(n_samples_list, cvs)
        suggested_pop_size = finv(target_cv)
        return CVEstimate(suggested_pop_size, n_samples_list, cvs, f, popt)
    except RuntimeError:
        logger.warning("Power law fit failed. "
                       "Falling back to current nr particles {}"
                       .format(current_poop_size))
        return CVEstimate(current_poop_size, n_samples_list, cvs, None, None)
