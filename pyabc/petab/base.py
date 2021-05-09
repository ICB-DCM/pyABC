from collections.abc import Sequence, Mapping
from typing import Callable, Tuple, Union
from numbers import Number
import abc
import logging
import numpy as np
import pandas as pd

import pyabc

logger = logging.getLogger("ABC.PEtab")

try:
    import petab
    import petab.C as C
except ImportError:
    petab = C = None
    logger.error("Install petab (see https://github.com/icb-dcm/petab) to use "
                 "the petab functionality.")


# priors that are evaluated "on-scale"
SCALED_PRIORS = [C.PARAMETER_SCALE_UNIFORM,
                 C.PARAMETER_SCALE_NORMAL,
                 C.PARAMETER_SCALE_LAPLACE]


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
        self.prior_scales, self.scaled_scales = get_scales(
            self.petab_problem.parameter_df)
        self._sanity_check()

    def create_prior(self) -> pyabc.Distribution:
        """Create prior.

        Note: The returned parameters are on the `objectivePriorType` scale,
        i.e. on `parameterScale` if the prior is one of `parameterScale...`,
        otherwise on linear scale.
        The model must then take care of transforming the parameters from
        prior scale to the right scale.
        Effectively, pyABC thus ignores the `parameterScale`. If this is an
        issue as sampling e.g. on lin or log scale could make a difference,
        this can be revised.

        Returns
        -------
        prior: A valid pyabc.Distribution for the parameters to estimate.
        """
        return create_prior(parameter_df=self.petab_problem.parameter_df)

    @abc.abstractmethod
    def create_model(
        self,
    ) -> Callable[[Union[Sequence, Mapping]], Mapping]:
        """Create model.

        The model takes parameters and simulates data
        for these. Different model simulation formalisms may be employed.
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
        This method must be overwritten in derived classes.

        Returns
        -------
        kernel: A pyabc distribution encoding the kernel function.
        """

    def get_nominal_parameters(
            self, target_scale: str = 'prior') -> pyabc.Parameter:
        """Get nominal parameters.

        Parameters
        ----------
        target_scale:
            Scale on which to return the parameters.
            Values: 'prior'|'scaled'|'lin'.
            If 'prior', they are on the scale the corresponding prior is on.
            If 'scaled', they are scaled, if 'lin', they are unscaled.

        Returns
        -------
        par_nominal:
            The nominal parameters if provided in the petab parameters data
            frame.
        """
        return get_nominal_parameters(
            parameter_df=self.petab_problem.parameter_df,
            target_scale=target_scale,
            prior_scales=self.prior_scales,
            scaled_scales=self.scaled_scales,
        )

    def get_bounds(
            self, target_scale: str = 'prior', use_prior: bool = True) -> dict:
        """Get bounds.

        Parameters
        ----------
        target_scale:
            Scale to get the nominal parameters on.
            Can be 'lin', 'scaled', or 'prior'.
        use_prior: Whether to use prior overrides if of type uniform.

        Returns
        -------
        bounds: Dictionary with a (lower, upper) tuple per parameter.
        """
        return get_bounds(
            parameter_df=self.petab_problem.parameter_df,
            target_scale=target_scale,
            prior_scales=self.prior_scales,
            scaled_scales=self.scaled_scales,
            use_prior=use_prior,
        )

    def get_parameter_names(self, target_scale: str = 'prior'):
        """Get meaningful parameter names, corrected for target scale."""
        parameter_df = petab.normalize_parameter_df(
            self.petab_problem.parameter_df)

        # scale
        if target_scale == C.LIN:
            target_scales = {key: C.LIN for key in self.prior_scales}
        elif target_scale == 'prior':
            target_scales = self.prior_scales
        elif target_scale == 'scaled':
            target_scales = self.scaled_scales
        else:
            raise ValueError(f"Did not recognize target scale {target_scale}")

        names = {}
        for _, row in parameter_df.reset_index().iterrows():
            if row[C.ESTIMATE] == 0:
                continue
            key = row[C.PARAMETER_ID]
            name = str(key)
            if C.PARAMETER_NAME in parameter_df:
                if not petab.is_empty(row[C.PARAMETER_NAME]):
                    name = str(row[C.PARAMETER_NAME])

            target_scale = target_scales[key]
            if target_scale != C.LIN:
                # mini check whether the name might indicate the scale already
                if not name.startswith("log"):
                    name = target_scale + "(" + name + ")"
            names[key] = name
        return names

    def _sanity_check(self):
        """Some checks to identify potential problems."""
        if any(self.scaled_scales[key] != C.LIN
               and self.prior_scales[key] == C.LIN
               for key in self.prior_scales):
            logger.warning(
                "Found parameters with prior scale lin, parameter scale not "
                "lin. Note that pyABC currently ignores the parameter scale "
                "in this case and just performs sampling on the prior scale.")


def create_prior(parameter_df: pd.DataFrame) -> pyabc.Distribution:
    """Create prior.

    Note: The prior generates samples according to
    `OBJECTIVE_PRIOR_TYPE` and `OBJECTIVE_PRIOR_PARAMETERS`.
    These samples are only scaled if the prior type is `PARAMETER_SCALE_...`,
    otherwise unscaled. The model has to take care of converting samples
    accordingly.

    Parameters
    ----------
    parameter_df: The PEtab parameter data frame.

    Returns
    -------
    prior: A valid pyabc.Distribution for the parameters to estimate.
    """
    # add default values
    parameter_df = petab.normalize_parameter_df(parameter_df)

    prior_dct = {}

    # iterate over parameters
    for _, row in parameter_df.reset_index().iterrows():
        if row[C.ESTIMATE] == 0:
            # ignore fixed parameters
            continue

        # pyabc currently only knows objective priors, no
        #  initialization priors
        prior_type = row[C.OBJECTIVE_PRIOR_TYPE]
        pars_str = row[C.OBJECTIVE_PRIOR_PARAMETERS]
        prior_pars = tuple(float(val) for val in pars_str.split(';'))

        # create random variable from table entry
        if prior_type in [C.PARAMETER_SCALE_UNIFORM,
                          C.UNIFORM]:
            lb, ub = prior_pars
            # scipy pars are location, width
            rv = pyabc.RV('uniform', loc=lb, scale=ub-lb)
        elif prior_type in [C.PARAMETER_SCALE_NORMAL,
                            C.NORMAL]:
            mean, std = prior_pars
            # scipy pars are mean, std
            rv = pyabc.RV('norm', loc=mean, scale=std)
        elif prior_type in [C.PARAMETER_SCALE_LAPLACE,
                            C.LAPLACE]:
            mean, b = prior_pars
            # scipy pars are loc=mean, scale=b
            rv = pyabc.RV('laplace', loc=mean, scale=b)
        elif prior_type == C.LOG_NORMAL:
            mean, std = prior_pars
            # petab pars are mean, std of the underlying normal distribution
            # scipy pars are s, loc, scale where s = std, scale = exp(mean)
            #  as a simple calculation shows
            rv = pyabc.RV('lognorm', s=std, loc=0, scale=np.exp(mean))
        elif prior_type == C.LOG_LAPLACE:
            mean, b = prior_pars
            # petab pars are mean, b of the underlying laplace distribution
            # scipy pars are c, loc, scale where c = 1 / b, scale = exp(mean)
            #  as a simple calculation shows
            rv = pyabc.RV('loglaplace', c=1/b, scale=np.exp(mean))
        else:
            raise ValueError(f"Cannot handle prior type {prior_type}.")

        prior_dct[row[C.PARAMETER_ID]] = rv

    # create prior distribution
    prior = pyabc.Distribution(**prior_dct)

    return prior


def get_scales(parameter_df: pd.DataFrame) -> Tuple[dict, dict]:
    """Unravel whether the priors and evaluations are on or off scale.

    Only the `parameterScale...` priors are on-scale, the other priors are on
    linear scale.

    Parameters
    ----------
    parameter_df: The PEtab parameter data frame.

    Returns
    -------
    prior_scales, scaled_scales:
        Scales for each parameter on prior and evaluation level.
    """
    # fill in objective prior columns
    parameter_df = petab.normalize_parameter_df(parameter_df)
    prior_scales = {}
    scaled_scales = {}
    for _, row in parameter_df.reset_index().iterrows():
        if row[C.ESTIMATE] == 0:
            continue
        prior_scales[row[C.PARAMETER_ID]] = (
            row[C.PARAMETER_SCALE]
            if row[C.OBJECTIVE_PRIOR_TYPE] in SCALED_PRIORS
            else C.LIN)
        scaled_scales[row[C.PARAMETER_ID]] = row[C.PARAMETER_SCALE]
    return prior_scales, scaled_scales


def rescale(val: Number, origin_scale: str, target_scale: str) -> Number:
    """Convert parameter value from origin scale to target scale.

    Parameters
    ----------
    val: Parameter value to rescale.
    origin_scale: Origin scale.
    target_scale: Target scale.

    Returns
    -------
    val: The rescaled parameter value.
    """
    if origin_scale == target_scale:
        # nothing to be done
        return val
    # origin to linear to target
    return petab.scale(petab.unscale(val, origin_scale), target_scale)


def get_nominal_parameters(
        parameter_df: pd.DataFrame,
        target_scale: str,
        prior_scales: dict,
        scaled_scales: dict) -> pyabc.Parameter:
    """Get nominal parameters.

    Parameters
    ----------
    parameter_df: The PEtab parameter data frame.
    target_scale:
        Scale to get the nominal parameters on.
        Can be 'lin', 'scaled', or 'prior'.
    prior_scales: Prior scales.
    scaled_scales: On-scale scales.

    Returns
    -------
    par_nominal: The nominal parameters if provided in the data frame.
    """
    # unscaled parameters
    par = pyabc.Parameter(
        {key: parameter_df.loc[key, C.NOMINAL_VALUE]
         for key in prior_scales.keys()})

    # scale
    if target_scale == C.LIN:
        # nothing to be done
        return par
    elif target_scale == 'prior':
        target_scales = prior_scales
    elif target_scale == 'scaled':
        target_scales = scaled_scales
    else:
        raise ValueError(f"Did not recognize target scale {target_scale}")
    # map linear to target scales component-wise
    return map_rescale(par, origin_scales=C.LIN, target_scales=target_scales)


def get_bounds(
        parameter_df: pd.DataFrame,
        target_scale: str,
        prior_scales: dict,
        scaled_scales: dict,
        use_prior: bool) -> dict:
    """Get bounds.

    Parameters
    ----------
    parameter_df: The PEtab parameter data frame.
    target_scale:
        Scale to get the nominal parameters on.
        Can be 'lin', 'scaled', or 'prior'.
    prior_scales: Prior scales.
    scaled_scales: On-scale scales.
    use_prior: Whether to use prior overrides if of type uniform.

    Returns
    -------
    bounds: Dictionary with a (lower, upper) tuple per parameter.
    """
    parameter_df = petab.normalize_parameter_df(parameter_df)

    # scale
    if target_scale == C.LIN:
        target_scales = {key: C.LIN for key in prior_scales}
    elif target_scale == 'prior':
        target_scales = prior_scales
    elif target_scale == 'scaled':
        target_scales = scaled_scales
    else:
        raise ValueError(f"Did not recognize target scale {target_scale}")

    # extract bounds
    bounds = {}
    for _, row in parameter_df.reset_index().iterrows():
        if row[C.ESTIMATE] == 0:
            # ignore fixed parameters
            continue

        key = row[C.PARAMETER_ID]

        # from lower and upper bound
        lower, upper = row[C.LOWER_BOUND], row[C.UPPER_BOUND]
        origin_scale = C.LIN

        # from prior
        prior_type = row[C.OBJECTIVE_PRIOR_TYPE]
        if use_prior and prior_type in [C.UNIFORM, C.PARAMETER_SCALE_UNIFORM]:
            pars_str = row[C.OBJECTIVE_PRIOR_PARAMETERS]
            lower, upper = tuple(float(val) for val in pars_str.split(';'))
            if prior_type == C.PARAMETER_SCALE_UNIFORM:
                origin_scale = row[C.PARAMETER_SCALE]

        # convert to target scale
        lower = rescale(
            lower, origin_scale=origin_scale, target_scale=target_scales[key])
        upper = rescale(
            upper, origin_scale=origin_scale, target_scale=target_scales[key])
        bounds[key] = (lower, upper)
    return bounds


def map_rescale(
        par: pyabc.Parameter,
        origin_scales: Union[dict, str], target_scales: Union[dict, str],
) -> pyabc.Parameter:
    """Rescale parameter dictionary.

    Parameters
    ----------
    par: The parameter to rescale.
    origin_scales: The origin scales.
    target_scales: The target scales.

    Returns
    -------
    par: The rescaled parameter.
    """
    # handle convenience input
    if isinstance(origin_scales, str):
        origin_scales = {key: origin_scales for key in par.keys()}
    if isinstance(target_scales, str):
        target_scales = {key: target_scales for key in par.keys()}

    # rescale each component
    for key, val in par.items():
        par[key] = rescale(
            val=val,
            origin_scale=origin_scales[key],
            target_scale=target_scales[key],
        )

    return par
