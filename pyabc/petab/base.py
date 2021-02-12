from collections.abc import Sequence, Mapping
from typing import Callable, Union
import abc
import logging
import numpy as np
import pandas as pd

import pyabc

logger = logging.getLogger(__name__)

try:
    import petab
except ImportError:
    petab = None
    logger.error("Install petab (see https://github.com/icb-dcm/petab) to use "
                 "the petab functionality.")


class PetabImporter(abc.ABC):
    """Import a PEtab model to parameterize it using pyABC.

    This class provides methods to generate prior, model, and stochastic kernel
    for a pyABC analysis.

    Parameters
    ----------
    petab_problem:
        A PEtab problem containing all information on the parameter estimation
        problem.
    """

    def __init__(
            self,
            petab_problem: petab.Problem):
        self.petab_problem = petab_problem

    def create_prior(self) -> pyabc.Distribution:
        """Create prior.

        Returns
        -------
        prior:
            A valid pyabc.Distribution for the parameters to estimate.
        """
        return create_prior(parameter_df=self.petab_problem.parameter_df)

    @abc.abstractmethod
    def create_model(
        self,
    ) -> Callable[[Union[Sequence, Mapping]], Mapping]:
        """Create model.

        The model takes parameters and simulates data
        for these. Different model simulation formalisms may be employed.

        .. note::
            This method must be overwritten in derived classes.

        Returns
        -------
        model:
            Employs some model formalism to generate simulated data for the
            analyzed system given parameters.
        """

    @abc.abstractmethod
    def create_kernel(
        self,
    ) -> pyabc.StochasticKernel:
        """
        Create acceptance kernel. The kernel takes the simulation result
        and computes a likelihood value by comparing simulated and observed
        data.

        .. note::
            This method must be overwritten in derived classes.

        Returns
        -------
        kernel:
            A pyabc distribution encoding the kernel function.
        """


def create_prior(parameter_df: pd.DataFrame) -> pyabc.Distribution:
    """Create prior.

    Note: For sampling the `PARAMETER_SCALE` is irrelevant, as the distribution
    is fully specified via
    `OBJECTIVE_PRIOR_TYPE` and `OBJECTIVE_PRIOR_PARAMETERS`.

    Parameters
    ----------
    parameter_df:
        The PEtab parameter dataframe.

    Returns
    -------
    prior:
        A valid pyabc.Distribution for the parameters to estimate.
    """
    # add default values
    parameter_df = petab.normalize_parameter_df(parameter_df)

    prior_dct = {}

    # iterate over parameters
    for _, row in parameter_df.reset_index().iterrows():
        if row[petab.C.ESTIMATE] == 0:
            # ignore fixed parameters
            continue

        # pyabc currently only knows objective priors, no
        #  initialization priors
        prior_type = row[petab.C.OBJECTIVE_PRIOR_TYPE]
        pars_str = row[petab.C.OBJECTIVE_PRIOR_PARAMETERS]
        prior_pars = tuple(float(val) for val in pars_str.split(';'))

        # create random variable from table entry
        if prior_type in [petab.C.PARAMETER_SCALE_UNIFORM,
                          petab.C.UNIFORM]:
            lb, ub = prior_pars
            # scipy pars are location, width
            rv = pyabc.RV('uniform', loc=lb, scale=ub-lb)
        elif prior_type in [petab.C.PARAMETER_SCALE_NORMAL,
                            petab.C.NORMAL]:
            mean, std = prior_pars
            # scipy pars are mean, std
            rv = pyabc.RV('norm', loc=mean, scale=std)
        elif prior_type in [petab.C.PARAMETER_SCALE_LAPLACE,
                            petab.C.LAPLACE]:
            mean, b = prior_pars
            # scipy pars are loc=mean, scale=b
            rv = pyabc.RV('laplace', loc=mean, scale=b)
        elif prior_type == petab.C.LOG_NORMAL:
            mean, std = prior_pars
            # petab pars are mean, std of the underlying normal distribution
            # scipy pars are s, loc, scale where s = std, scale = exp(mean)
            #  as a simple calculation shows
            rv = pyabc.RV('lognorm', s=std, loc=0, scale=np.exp(mean))
        elif prior_type == petab.C.LOG_LAPLACE:
            mean, b = prior_pars
            # petab pars are mean, b of the underlying laplace distribution
            # scipy pars are c, loc, scale where c = 1 / b, scale = exp(mean)
            #  as a simple calculation shows
            rv = pyabc.RV('loglaplace', c=1/b, scale=np.exp(mean))
        else:
            raise ValueError(f"Cannot handle prior type {prior_type}.")

        prior_dct[row[petab.C.PARAMETER_ID]] = rv

    # create prior distribution
    prior = pyabc.Distribution(**prior_dct)

    return prior
