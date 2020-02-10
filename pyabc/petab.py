import sys
import os
import logging

import pyabc

logger = logging.getLogger(__name__)

try:
    import petab
except ImportError:
    logger.error("Install petab (see https://github.com/icb-dcm/petab) to use "
                 "the petab functionality.")

try:
    import amici
    from amici.petab_objective import simulate_petab
except ImportError:
    logger.error("Install amici (see https://github.com/icb-dcm/amici) to use "
                 "the amici functionality.")


class AmiciPetabImporter:

    def __init__(
            self,
            petab_problem: petab.Problem,
            amici_model: amici.Model,
            amici_solver: amici.Solver,
    ):      
        self.petab_problem = petab_problem
        self.amici_model = amici_model
        self.amici_solver = amici_solver

    def get_prior(self):
        lbs = self.petab_problem.get_lb(fixed=False, scaled=True)
        ubs = self.petab_problem.get_ub(fixed=False, scaled=True)
        x_ids = self.petab_problem.get_x_ids(fixed=False)
        dct = {}
        for lb, ub, x_id in zip(lbs, ubs, x_ids):
            dct[x_id] = pyabc.RV('uniform', lb, ub-lb)
        prior = pyabc.Distribution(**dct)
        return prior

    def get_model(self):
        x_ids = self.petab_problem.x_ids

        def model(par):
            if isinstance(par, list):
                par = {key: val for key, val in zip(x_ids, par)}
            ret = simulate_petab(
                petab_problem=self.petab_problem,
                amici_model=self.amici_model,
                solver=self.amici_solver,
                problem_parameters=par,
                scaled_parameters=True)
            return {'llh': ret['llh']}

        return model
    
    def get_kernel(self):
        def kernel_fun(x, x_0, t, par):
            return x['llh']
        kernel = pyabc.distance.SimpleFunctionKernel(
            kernel_fun, ret_scale=pyabc.distance.SCALE_LOG)
        return kernel
