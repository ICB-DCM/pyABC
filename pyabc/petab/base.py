import pyabc

from collections.abc import Sequence, Mapping
from typing import Callable, Union
import abc
import logging

logger = logging.getLogger(__name__)

try:
    import petab
except ImportError:

    logger.error("Install petab (see https://github.com/icb-dcm/petab) to use "
                 "the petab functionality.")


class PetabImporter(abc.ABC):
    """
    Import a PEtab model to parameterize it using pyABC.

    This class provides methods to generate prior, model, and stochastic kernel
    for a pyABC analysis.

    Parameters
    ----------

    petab_problem:
        A PEtab problem containing all information on the parameter estimation
        problem.
    free_parameters:
        Whether to estimate free parameters (column ESTIMATE=1 in the
        parameters table).
    fixed_parameters:
        Whether to estimate fixed parameters (column ESTIMATE=0 in the
        parameters table).
    """

    def __init__(
            self,
            petab_problem: petab.Problem,
            free_parameters: bool = True,
            fixed_parameters: bool = False):
        self.petab_problem = petab_problem
        self.free_parameters = free_parameters
        self.fixed_parameters = fixed_parameters

    def create_prior(self) -> pyabc.Distribution:
        """
        Create prior.

        Returns
        -------
        prior:
            A valid pyabc.Distribution for the parameters to estimate.
        """
        # add default values
        parameter_df = petab.normalize_parameter_df(
            self.petab_problem.parameter_df)

        prior_dct = {}

        # iterate over parameters
        for _, row in parameter_df.reset_index().iterrows():
            # check whether we can ignore
            if not self.fixed_parameters and row[petab.C.ESTIMATE] == 0:
                # ignore fixed parameters
                continue
            if not self.free_parameters and row[petab.C.ESTIMATE] == 1:
                # ignore free parameters
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
                rv = pyabc.RV('uniform', lb, ub-lb)
            elif prior_type in [petab.C.PARAMETER_SCALE_NORMAL,
                                petab.C.NORMAL]:
                mean, std = prior_pars
                rv = pyabc.RV('norm', mean, std)
            elif prior_type in [petab.C.PARAMETER_SCALE_LAPLACE,
                                petab.C.LAPLACE]:
                mean, scale = prior_pars
                rv = pyabc.RV('laplace', mean, scale)
            elif prior_type == petab.C.LOG_NORMAL:
                mean, std = prior_pars
                rv = pyabc.RV('lognorm', mean, std)
            elif prior_type == petab.C.LOG_LAPLACE:
                mean, scale = prior_pars
                rv = pyabc.RV('loglaplace', mean, scale)
            else:
                raise ValueError(f"Cannot handle prior type {prior_type}.")

            prior_dct[row[petab.C.PARAMETER_ID]] = rv

        # create prior distribution
        prior = pyabc.Distribution(**prior_dct)

        return prior

    @abc.abstractmethod
    def create_model(
        self,
    ) -> Callable[[Union[Sequence, Mapping]], Mapping]:
        """
        Create model. The model takes parameters and simulates data
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
