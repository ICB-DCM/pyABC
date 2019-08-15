"""
Schemes for choosing the next temperature.

Some schems require a given finite number of populations to work. This is
such as to ensure decrease to a temperature of 1.0 after finitely many
iterations. In such case, choosing an adequate maximum number of populations
can be essential for efficiency. Schemes that directly try to optimize the
number of populations have not been implemented.
"""


import numpy as np
import scipy as sp
import logging
from ..distance import RET_SCALE_LIN


logger = logging.getLogger("Acceptor")


def scheme_acceptance_rate(**kwargs):
    """
    Try to keep the acceptance rate constant at a value of
    `target_acceptance_rate`. Note that this scheme will fail to
    reduce the temperature sufficiently in later iterations, if the
    problem's inherent acceptance rate is lower, but it has been
    observed to give big temperature leaps in early iterations.
    """
    # required fields
    t = kwargs['t']
    temperatures = kwargs['temperatures']
    temp_init = kwargs['temp_init']
    get_weighted_distances = kwargs['get_weighted_distances']
    pdf_max = kwargs['pdf_max']
    ret_scale = kwargs['ret_scale']
    max_nr_populations = kwargs['max_nr_populations']
    target_acceptance_rate = kwargs.get('target_acceptance_rate', 0.5)

    # safety check
    if t >= max_nr_populations - 1:
        # t is last time
        return 1.0

    # is there a pre-defined step to start with?
    if t - 1 not in temperatures and temp_init is not None:
        return temp_init

    # execute function (expensive if in calibration)
    df = get_weighted_distances()

    weights = np.array(df['w'])
    pdfs = np.array(df['distance'])

    # compute rescaled posterior densities
    if ret_scale == RET_SCALE_LIN:
        values = pdfs / pdf_max
    else:  # ret_scale == RET_SCALE_LOG
        values = np.exp(pdfs - pdf_max)

    # to pmf
    weights /= np.sum(weights)

    # objective function which we wish to find a root for
    def obj(beta):
        val = np.sum(weights * values**beta) - \
            target_acceptance_rate
        return val

    if obj(1) > 0:
        # function is mon. dec., smallest possible value already > 0
        beta_opt = 1.0
        # it is obj(0) >= 0 always
    else:
        # perform binary search
        # one could also go for a more efficient optimization approach
        beta_opt = sp.optimize.bisect(obj, 0, 1)

    # temperature is inverse beta
    temp_opt = 1.0 / beta_opt
    return temp_opt


def scheme_exponential_decay(**kwargs):
    """
    If `max_nr_populations` is finite:

    .. math::
        T_j = T_{max}^{(n-j)/n}

    where n denotes the number of populations, and j=1,...,n the iteration.
    This translates to

    .. math::
        T_j = T_{j-1}^{(n-j)/(n-(j-1))}.

    This ensures that a temperature of 1.0 is reached after exactly the
    remaining number of steps.

    If `max_nr_populations` is infinite, the next temperature is

    .. math::
        T_j = \\alpha \\cdot T_{j-1},

    where by default alpha=0.25.

    So, in both cases the sequence of temperatures follows an exponential
    decay, also known as a geometric progression, or a linear progression
    in log-space.

    Note that the formula is applied anew in each iteration. That has the
    advantage that, if also other schemes are used s.t. T_{j-1} is smaller
    than by the above, advantage can be made of this.
    """
    # required fields
    t = kwargs['t']
    temperatures = kwargs['temperatures']
    max_nr_populations = kwargs['max_nr_populations']
    temp_init = kwargs['temp_init']
    alpha = kwargs.get('alpha', 0.25)

    # safety check
    if t >= max_nr_populations - 1:
        # t is last time
        return 1.0

    # find previous temperature, or return if none available
    if t - 1 in temperatures:
        temp_base = temperatures[t - 1]
    elif temp_init is not None:
        return temp_init
    else:
        # should give a good first temperature
        return scheme_acceptance_rate(**kwargs)

    if max_nr_populations == np.inf:
        # just decrease by a factor of alpha each round
        temp = alpha * temp_base
        return temp

    # how many steps left?
    t_to_go = max_nr_populations - t

    # compute next temperature according to exponential decay
    temp = temp_base ** ((t_to_go - 1) / t_to_go)

    return temp


def scheme_polynomial_decay(**kwargs):
    """
    Compute next temperature as pre-last entry in
    >>> np.linspace(1, (temp_base)**(1 / temp_decay_exponent),
    >>>             t_to_go + 1) ** temp_decay_exponent)

    Requires finite `max_nr_populations`.

    Note that this is similar to the `scheme_exponential_decay`, which is
    indeed the limit for `temp_decay_exponent -> infinity`. For smaller
    exponent, the sequence makes larger steps for low temperatures. This
    can be useful in cases, where lower temperatures (which are usually
    more expensive) can be traversed in few larger steps.
    """
    # required fields
    t = kwargs['t']
    temperatures = kwargs['temperatures']
    max_nr_populations = kwargs['max_nr_populations']
    temp_init = kwargs['temp_init']
    temp_decay_exponent = kwargs.get('temp_decay_exponent', 3)

    # check if we can compute a decay step
    if max_nr_populations == np.inf:
        raise ValueError(
            "Can only perform decay step with a finite max_nr_populations.")

    # safety check
    if t >= max_nr_populations - 1:
        # t is last time
        return 1.0

    # get temperature to start with
    if t - 1 in temperatures:
        temp_base = temperatures[t - 1]
    elif temp_init is not None:
        return temp_init
    else:
        # should give a good first temperature
        return scheme_acceptance_rate(**kwargs)

    # how many steps left?
    t_to_go = max_nr_populations - t

    # compute sequence
    temps = np.linspace(1, (temp_base)**(1 / temp_decay_exponent),
                        t_to_go + 1) ** temp_decay_exponent

    logger.debug(f"Temperatures proposed by decay method: {temps}.")

    # pre-last step is the next step
    temp = temps[-2]
    return temp


def scheme_friel_pettitt(**kwargs):
    """
    Basically takes linear steps in lin-space. See [#vyshemirsky2008]_.

    .. [#vyshemirsky2008] Vyshemirsky, Vladislav, and Mark A. Girolami.
        "Bayesian ranking of biochemical system models."
        Bioinformatics 24.6 (2007): 833-839.
    """
    # required fields
    t = kwargs['t']
    temperatures = kwargs['temperatures']
    max_nr_populations = kwargs['max_nr_populations']
    temp_init = kwargs['temp_init']

    # check if we can compute a decay step
    if max_nr_populations == np.inf:
        raise ValueError(
            "Can only perform decaqy step with a finite max_nr_populations.")

    # safety check
    if t >= max_nr_populations - 1:
        # t is last time
        return 1.0

    # get temperature to start with
    if t - 1 in temperatures:
        temp_base = temperatures[t - 1]
        beta_base = 1. / temp_base
    elif temp_init is not None:
        return temp_init
    else:
        # should give a good first temperature
        return scheme_acceptance_rate(**kwargs)

    beta = beta_base + ((1. - beta_base) * 1 / (max_nr_populations - t)) ** 2

    return 1. / beta


def scheme_daly(**kwargs):
    """
    Use modified scheme based on [#daly2017]_.


    .. [#daly2017] Daly Aidan C., Cooper Jonathan, Gavaghan David J. ,
            and Holmes Chris. "Comparing two sequential Monte Carlo samplers
            for exact and approximate Bayesian inference on biological
            models". Journal of The Royal Society Interface, 2017
    """
    # required fields
    t = kwargs['t']
    temperatures = kwargs['temperatures']
    temp_init = kwargs['temp_init']
    acceptance_rate = kwargs['acceptance_rate']
    min_acceptance_rate = kwargs.get('min_acceptance_rate', 1e-4)
    max_nr_populations = kwargs['max_nr_populations']

    config = kwargs.get('config', {})
    k = config.setdefault('k', {t: temp_init})

    alpha = kwargs.get('alpha', 0.5)

    # safety check
    if t >= max_nr_populations - 1:
        # t is last time
        return 1.0

    if t - 1 in temperatures:
        # addressing the std, not the var
        eps_base = np.sqrt(temperatures[t - 1])
        k_base = k[t - 1]
    else:
        if temp_init is not None:
            temp = temp_init
        else:
            # should give a good first temperature
            temp = scheme_acceptance_rate(**kwargs)
        # k controls reduction in error threshold
        k[t] = np.sqrt(temp)
        return temp

    if acceptance_rate < min_acceptance_rate:
        logger.debug("Daly scheduler: Reacting to low acceptance rate.")
        # reduce reduction ke
        k_base = alpha * k_base

    k[t] = min(k_base, alpha * eps_base)
    eps = eps_base - k[t]
    temp = eps**2

    return temp


def scheme_ess(**kwargs):
    """
    Try to keep the effective sample size constant.

    We accept particles with a certain probability (the distance).
    """
    # required fields
    t = kwargs['t']
    temperatures = kwargs['temperatures']
    max_nr_populations = kwargs['max_nr_populations']
    get_weighted_distances = kwargs['get_weighted_distances']
    ret_scale = kwargs['ret_scale']
    pdf_max = kwargs['pdf_max']
    target_ress = kwargs.get('target_ress', 0.8)
    # check if we can compute a decay step
    if max_nr_populations == np.inf:
        raise ValueError(
            "Can only perform decay step with a finite max_nr_populations.")

    # safety check
    if t >= max_nr_populations - 1:
        # t is last time
        return 1.0

    # execute function (expensive if in calibration)
    df = get_weighted_distances()

    weights = np.array(df['w'])
    pdfs = np.array(df['distance'])

    # compute rescaled posterior densities
    if ret_scale == RET_SCALE_LIN:
        values = pdfs / pdf_max
    else:  # ret_scale == RET_SCALE_LOG
        values = np.exp(pdfs - pdf_max)

    # to pmf
    weights /= np.sum(weights)

    target_ess = len(weights) * target_ress
    if t - 1 in temperatures:
        beta_base = 1 / temperatures[t - 1]
    else:
        beta_base = 0.0

    # objective to minimize
    def obj(beta):
        return (ess(values, weights, beta) - target_ess)**2
    res = sp.optimize.minimize(
        obj, x0=np.array([0.5 * (1 + beta_base)]),
        bounds=sp.optimize.Bounds(lb=np.array([beta_base]), ub=np.array([1])))
    beta = res.x
    temp = 1. / beta

    return temp


def ess(pdfs, weights, beta):
    num = np.sum(weights * pdfs**beta)**2
    den = np.sum((weights * pdfs**beta) ** 2)
    return num / den
