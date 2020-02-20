import logging
from collections.abc import Sequence, Mapping
from typing import Callable, Union
import copy

import pyabc
from .base import PetabImporter

logger = logging.getLogger(__name__)

try:
    import petab
except ImportError:
    logger.error("Install petab (see https://github.com/icb-dcm/petab) to use "
                 "the petab functionality.")

try:
    import amici
    from amici.petab_objective import simulate_petab, LLH, RDATAS
except ImportError:
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
    free_parameters:
        Whether to estimate free parameters (column ESTIMATE=1 in the
        parameters table).
    fixed_parameters:
        Whether to estimate fixed parameters (column ESTIMATE=0 in the
        parameters table).
    amici_model:
        A corresponding compiled AMICI model that allows simulating data for
        parameters. If not provided, one is created using
        `amici.petab_import.import_petab_problem`.
    amici_solver:
        An AMICI solver to simulate the model. If not provided, one is created
        using `amici_model.getSolver()`.
    store_simulations:
        Whether to store performed simulations. Per default, only parameters
        and likelihood valuaes are stored. Since an ODE model
        is deterministic, the trajectories can easily be reproduced.
    """

    def __init__(
            self,
            petab_problem: petab.Problem,
            amici_model: amici.Model = None,
            amici_solver: amici.Solver = None,
            free_parameters: bool = True,
            fixed_parameters: bool = False,
            store_simulations: bool = False):
        super().__init__(
            petab_problem=petab_problem,
            free_parameters=free_parameters,
            fixed_parameters=fixed_parameters)

        if amici_model is None:
            amici_model = amici.getab_import.import_petab_problem(
                petab_problem)
        self.amici_model = amici_model

        if amici_solver is None:
            amici_solver = self.amici_model.getSolver()
        self.amici_solver = amici_solver

        self.store_simulations = store_simulations

    def create_model(
        self
    ) -> Callable[[Union[Sequence, Mapping]], Mapping]:
        """Create model."""
        # parameter ids to consider
        x_ids = self.petab_problem.get_x_ids(
            free=self.free_parameters, fixed=self.fixed_parameters)

        # fixed paramters
        x_fixed_ids = self.petab_problem.get_x_ids(
            free=not self.free_parameters, fixed=not self.fixed_parameters)
        x_fixed_vals = self.petab_problem.get_x_nominal(
            scaled=True,
            free=not self.free_parameters, fixed=not self.fixed_parameters)

        # extract variables for improved pickling
        petab_problem = self.petab_problem
        amici_model = self.amici_model
        amici_solver = self.amici_solver
        store_simulations = self.store_simulations

        # no gradients for pyabc
        amici_solver.setSensitivityOrder(0)

        def model(par: Union[Sequence, Mapping]) -> Mapping:
            """The model function."""
            # copy since we add fixed parameters
            par = copy.deepcopy(par)

            # convenience to allow calling model not only with dicts
            if not isinstance(par, Mapping):
                par = {key: val for key, val in zip(x_ids, par)}

            # add fixed parameters
            for key, val in zip(x_fixed_ids, x_fixed_vals):
                par[key] = val

            # simulate model
            sim = simulate_petab(
                petab_problem=petab_problem,
                amici_model=amici_model,
                solver=amici_solver,
                problem_parameters=par,
                scaled_parameters=True)

            # return values of interest
            ret = {'llh': sim[LLH]}
            if store_simulations:
                for i_rdata, rdata in enumerate(ret[RDATAS]):
                    ret[f'y_{i_rdata}'] = rdata['y']

            return ret

        return model

    def create_kernel(
        self
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
