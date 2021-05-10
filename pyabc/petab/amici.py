import logging
from collections.abc import Sequence, Mapping
from typing import Callable, Union
import copy

import pyabc
from .base import PetabImporter, rescale

logger = logging.getLogger("ABC.PEtab")

try:
    import petab
    import petab.C as C
except ImportError:
    petab = C = None
    logger.error("Install petab (see https://github.com/icb-dcm/petab) to use "
                 "the petab functionality.")

try:
    import amici
    import amici.petab_import
    from amici.petab_objective import simulate_petab, LLH, RDATAS
except ImportError:
    amici = amici.petab_import = simulate_petab = LLH = RDATAS = None
    logger.error("Install amici (see https://github.com/icb-dcm/amici) to use "
                 "the amici functionality.")


class AmiciPetabImporter(PetabImporter):
    """
    Import a PEtab model using AMICI to simulate it as a deterministic ODE.

    Parameters
    ----------
    petab_problem:
        A PEtab problem containing all information on the parameter estimation
        problem.
    amici_model:
        A corresponding compiled AMICI model that allows simulating data for
        parameters. If not provided, one is created using
        `amici.petab_import.import_petab_problem`.
    amici_solver:
        An AMICI solver to simulate the model. If not provided, one is created
        using `amici_model.getSolver()`.
    """

    def __init__(
            self,
            petab_problem: petab.Problem,
            amici_model: amici.Model = None,
            amici_solver: amici.Solver = None):
        super().__init__(petab_problem=petab_problem)

        if amici_model is None:
            amici_model = amici.petab_import.import_petab_problem(
                petab_problem)
        self.amici_model = amici_model

        if amici_solver is None:
            amici_solver = self.amici_model.getSolver()
        self.amici_solver = amici_solver

    def create_model(
            self,
            return_simulations: bool = False,
            return_rdatas: bool = False,
    ) -> Callable[[Union[Sequence, Mapping]], Mapping]:
        """Create model.

        Note that since AMICI uses deterministic ODE simulations,
        it is usually not necessary to store simulations, as these can
        be reproduced from the parameters.

        Parameters
        ----------
        return_simulations:
            Whether to return the simulations also (large, can be stored
            in database).
        return_rdatas:
            Whether to return the full `List[amici.ExpData]` objects (large,
            cannot be stored in database).

        Returns
        -------
        model:
            The model function, taking parameters and returning simulations.
            The model returns already the likelihood value.
        """
        # parameter ids to consider
        x_free_ids = self.petab_problem.get_x_ids(free=True, fixed=False)

        # fixed parameters
        x_fixed_ids = self.petab_problem.get_x_ids(
            free=False, fixed=True)
        x_fixed_vals = self.petab_problem.get_x_nominal(
            scaled=True, free=False, fixed=True)

        # extract variables for improved pickling
        petab_problem = self.petab_problem
        amici_model = self.amici_model
        amici_solver = self.amici_solver
        prior_scales = self.prior_scales
        scaled_scales = self.scaled_scales

        if set(prior_scales.keys()) != set(x_free_ids):
            # this should not happen
            raise AssertionError("Parameter id mismatch")

        # no gradients for pyabc
        amici_solver.setSensitivityOrder(0)

        def model(par: Union[Sequence, Mapping]) -> Mapping:
            """The model function.

            Note: The parameters are assumed to be passed on prior scale.
            """
            # copy since we add fixed parameters
            par = copy.deepcopy(par)

            # convenience to allow calling model not only with dicts
            if not isinstance(par, Mapping):
                par = {key: val for key, val in zip(x_free_ids, par)}

            # add fixed parameters
            for key, val in zip(x_fixed_ids, x_fixed_vals):
                par[key] = val

            # scale parameters whose priors are not on scale
            for key in prior_scales.keys():
                par[key] = rescale(
                    val=par[key],
                    origin_scale=prior_scales,
                    target_scale=scaled_scales,
                )

            # simulate model
            sim = simulate_petab(
                petab_problem=petab_problem,
                amici_model=amici_model,
                solver=amici_solver,
                problem_parameters=par,
                scaled_parameters=True)

            # return values of interest
            ret = {'llh': sim[LLH]}
            if return_simulations:
                for i_rdata, rdata in enumerate(sim[RDATAS]):
                    ret[f'y_{i_rdata}'] = rdata['y']
            if return_rdatas:
                ret[RDATAS] = sim[RDATAS]

            return ret

        return model

    def create_kernel(
        self,
    ) -> pyabc.StochasticKernel:
        """
        Create acceptance kernel.

        Returns
        -------
        kernel:
            A pyabc distribution encoding the kernel function.
        """
        def kernel_fun(x, x_0, t, par) -> float:
            """The kernel function."""
            # the kernel value is computed by amici already
            return x['llh']

        # create a kernel from function, returning log-scaled values
        kernel = pyabc.distance.SimpleFunctionKernel(
            kernel_fun, ret_scale=pyabc.distance.SCALE_LOG)

        return kernel
