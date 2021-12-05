"""PEtab import with AMICI simulator."""

import copy
import logging
import os
import tempfile
from collections.abc import Mapping, Sequence
from typing import Callable, Dict, Union

import pyabc

from .base import PetabImporter, rescale

logger = logging.getLogger("ABC.PEtab")

try:
    import petab
    import petab.C as C
except ImportError:
    petab = C = None
    logger.error(
        "Install PEtab (see https://github.com/icb-dcm/petab) to use "
        "the petab functionality, e.g. via `pip install pyabc[petab]`"
    )

try:
    import amici
    from amici import petab_import as amici_petab_import
    from amici.petab_objective import LLH, RDATAS, simulate_petab
except ImportError:
    amici = amici_petab_import = simulate_petab = LLH = RDATAS = None
    logger.error(
        "Install amici (see https://github.com/icb-dcm/amici) to use "
        "the amici functionality, e.g. via `pip install pyabc[amici]`"
    )


class AmiciModel:
    """Amici model."""

    def __init__(
        self,
        petab_problem,
        amici_model,
        amici_solver,
        x_free_ids,
        x_fixed_ids,
        x_fixed_vals,
        prior_scales,
        scaled_scales,
        return_simulations,
        return_rdatas,
    ):
        self.petab_problem = petab_problem
        self.amici_model = amici_model
        self.amici_solver = amici_solver
        self.x_free_ids = x_free_ids
        self.x_fixed_ids = x_fixed_ids
        self.x_fixed_vals = x_fixed_vals
        self.prior_scales = prior_scales
        self.scaled_scales = scaled_scales
        self.return_simulations = return_simulations
        self.return_rdatas = return_rdatas

    def __call__(self, par: Union[Sequence, Mapping]) -> Mapping:
        """The model function.

        Note: The parameters are assumed to be passed on prior scale.
        """
        # copy since we add fixed parameters
        par = copy.deepcopy(par)

        # convenience to allow calling model not only with dicts
        if not isinstance(par, Mapping):
            par = {key: val for key, val in zip(self.x_free_ids, par)}

        # add fixed parameters
        for key, val in zip(self.x_fixed_ids, self.x_fixed_vals):
            par[key] = val

        # scale parameters whose priors are not on scale
        for key in self.prior_scales.keys():
            par[key] = rescale(
                val=par[key],
                origin_scale=self.prior_scales,
                target_scale=self.scaled_scales,
            )

        # simulate model
        sim = simulate_petab(
            petab_problem=self.petab_problem,
            amici_model=self.amici_model,
            solver=self.amici_solver,
            problem_parameters=par,
            scaled_parameters=True,
        )

        # return values of interest
        ret = {'llh': sim[LLH]}
        if self.return_simulations:
            for i_rdata, rdata in enumerate(sim[RDATAS]):
                ret[f'y_{i_rdata}'] = rdata['y']
        if self.return_rdatas:
            ret[RDATAS] = sim[RDATAS]

        return ret

    def __getstate__(self) -> Dict:
        state = {}
        for key in set(self.__dict__.keys()) - {'amici_model', 'amici_solver'}:
            state[key] = self.__dict__[key]

        _fd, _file = tempfile.mkstemp()
        try:
            # write amici solver settings to file
            try:
                amici.writeSolverSettingsToHDF5(self.amici_solver, _file)
            except AttributeError as e:
                e.args += (
                    "Pickling the AmiciObjective requires an AMICI "
                    "installation with HDF5 support.",
                )
                raise
            # read in byte stream
            with open(_fd, 'rb', closefd=False) as f:
                state['amici_solver_settings'] = f.read()
        finally:
            # close file descriptor and remove temporary file
            os.close(_fd)
            os.remove(_file)

        return state

    def __setstate__(self, state: Dict):
        self.__dict__.update(state)

        model = amici_petab_import.import_petab_problem(self.petab_problem)
        solver = model.getSolver()

        _fd, _file = tempfile.mkstemp()
        try:
            # write solver settings to temporary file
            with open(_fd, 'wb', closefd=False) as f:
                f.write(state['amici_solver_settings'])
            # read in solver settings
            try:
                amici.readSolverSettingsFromHDF5(_file, solver)
            except AttributeError as err:
                if not err.args:
                    err.args = ('',)
                err.args += (
                    "Unpickling an AmiciObjective requires an AMICI "
                    "installation with HDF5 support.",
                )
                raise
        finally:
            # close file descriptor and remove temporary file
            os.close(_fd)
            os.remove(_file)

        self.amici_model = model
        self.amici_solver = solver


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
        amici_model: "amici.Model" = None,
        amici_solver: "amici.Solver" = None,
    ):
        super().__init__(petab_problem=petab_problem)

        if amici_model is None:
            amici_model = amici_petab_import.import_petab_problem(
                petab_problem
            )
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
        x_fixed_ids = self.petab_problem.get_x_ids(free=False, fixed=True)
        x_fixed_vals = self.petab_problem.get_x_nominal(
            scaled=True, free=False, fixed=True
        )

        if set(self.prior_scales.keys()) != set(x_free_ids):
            # this should not happen
            raise AssertionError("Parameter id mismatch")

        # no gradients for pyabc
        self.amici_solver.setSensitivityOrder(0)

        model = AmiciModel(
            petab_problem=self.petab_problem,
            amici_model=self.amici_model,
            amici_solver=self.amici_solver,
            x_free_ids=x_free_ids,
            x_fixed_ids=x_fixed_ids,
            x_fixed_vals=x_fixed_vals,
            prior_scales=self.prior_scales,
            scaled_scales=self.scaled_scales,
            return_simulations=return_simulations,
            return_rdatas=return_rdatas,
        )

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
        kernel = pyabc.distance.FunctionKernel(
            kernel_fun, ret_scale=pyabc.distance.SCALE_LOG
        )

        return kernel
