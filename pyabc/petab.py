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
            amici_solver: amici.Solver):
        self.petab_problem = petab_problem
        self.amici_model = amici_model
        self.amici_solver = amici_solver

    def get_prior(self):
        parameter_df = petab.normalize_parameter_df(
            self.petab_problem.parameter_df)

        prior_dct = {}
        for _, row in parameter_df.reset_index().iterrows():
            if row[petab.C.ESTIMATE] == 0:
                continue
            # pyabc currently does not have initialization priors
            prior_type = row[petab.C.OBJECTIVE_PRIOR_TYPE]

            pars_str = row[petab.C.OBJECTIVE_PRIOR_PARAMETERS]
            prior_pars = tuple([float(val) for val in pars_str.split(';')])

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
                raise ValueError(f"Cannot handle rior type {prior_type}.")

            prior_dct[row[petab.C.PARAMETER_ID]] = rv

        prior = pyabc.Distribution(**prior_dct)

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
