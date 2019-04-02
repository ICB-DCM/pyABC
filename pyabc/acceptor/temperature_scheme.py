import numpy as np
import scipy as sp
import logging
from ..distance import RET_SCALE_LIN


logger = logging.getLogger("Acceptor")


def scheme_acceptance_rate(**kwargs):
    # required fields
    t = kwargs['t']
    temperatures = kwargs['temperatures']
    temp_init = kwargs['temp_init']
    get_weighted_distances = kwargs['get_weighted_distances']
    pdf_max = kwargs['pdf_max']
    ret_scale = kwargs['ret_scale']
    target_acceptance_rate = kwargs['target_acceptance_rate']

    # is there a pre-defined step to start with?
    if t - 1 not in temperatures and temp_init is not None:
        return temp_init

    # execute function
    df = get_weighted_distances()
    weights = np.array(df['w'])
    pdfs = np.array(df['distance'])

    # compute rescaled posterior densities
    if ret_scale == RET_SCALE_LIN:
        values = pdfs / pdf_max
    else:  # ret_scale == RET_SCALE_LOG
        values = np.exp(pdfs - pdf_max)

    weights /= np.sum(weights)

    # objective function which we wish to find a root for
    def obj(beta):
        val = np.sum(weights * values**beta) - \
            target_acceptance_rate
        return val

    if obj(1) > 0:
        beta_opt = 1.0
    else:
        # perform binary search
        # TODO: take a more efficient optimization approach?
        beta_opt = sp.optimize.bisect(obj, 0, 1)

    # temperature is inverse beta
    temp_opt = 1.0 / beta_opt
    return temp_opt


def scheme_exponential_decay(**kwargs):
    # required fields
    t = kwargs['t']
    temperatures = kwargs['temperatures']
    max_nr_populations = kwargs['max_nr_populations']
    temp_init = kwargs['temp_init']
    alpha = kwargs['alpha']

    if t - 1 in temperatures:
        temp_base = temperatures[t - 1]
    elif temp_init is not None:
        return temp_init
    else:
        return scheme_acceptance_rate(**kwargs)

    if max_nr_populations == np.inf:
        # just decrease by a factor of alpha each round
        temp = alpha * temp_base
        return temp

    # how many steps left?
    t_to_go = (max_nr_populations - 1) - (t - 1)

    if t_to_go < 2:
        return 1.0

    temp = temp_base ** ((t_to_go - 1) / t_to_go)
    return temp


def scheme_decay(**kwargs):
    # required fields
    t = kwargs['t']
    temperatures = kwargs['temperatures']
    max_nr_populations = kwargs['max_nr_populations']
    temp_init = kwargs['temp_init']
    temp_decay_exponent = kwargs['temp_decay_exponent']

    # check if we can compute a decay step
    if max_nr_populations == np.inf:
        raise ValueError("Can only perform decay step with a finite "
                         "max_nr_populations.")

    # how many steps left?
    t_to_go = (max_nr_populations - 1) - (t - 1)
    if t_to_go < 2:
        # have to take exact step, i.e. a temperature of 1, next
        return 1.0

    # get temperature to start with
    if t - 1 in temperatures:
        temp_base = temperatures[t - 1]
    elif temp_init is not None:
        return temp_init
    else:
        return scheme_acceptance_rate(**kwargs)

    temps = np.linspace(1, (temp_base)**(1 / temp_decay_exponent),
                        t_to_go + 1) ** temp_decay_exponent
    logger.debug(f"Temperatures proposed by decay method: {temps}.")

    # pre-last step is the next step
    temp = temps[-2]
    return temp


def scheme_daly(**kwargs):
    """
    Use modified scheme in daly 2017.
    """
    # required fields
    t = kwargs['t']
    temperatures = kwargs['temperatures']
    temp_init = kwargs['temp_init']
    acceptance_rate = kwargs['acceptance_rate']
    min_acceptance_rate = kwargs.get('min_acceptance_rate', 1e-3)

    config = kwargs.get('config', {})
    k = config.setdefault('k', {t: temp_init})

    alpha = kwargs.get('alpha', 0.5)

    if t - 1 in temperatures:
        temp_base = temperatures[t - 1]
        k_base = k[t - 1]
    else:
        if temp_init is not None:
            temp = temp_init
        else:
            temp = scheme_acceptance_rate(**kwargs)
        # k controls reduction in error threshold
        k[t] = temp
        return temp

    if acceptance_rate < min_acceptance_rate:
        k_base = alpha * k_base

    k[t] = min(k_base, alpha * temp_base)
    temp = temp_base - k[t]

    return temp
