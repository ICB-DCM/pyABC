"""Execute petab test suite."""

import logging
import os
import sys

import petabtests
import pytest
from _pytest.outcomes import Skipped

import pyabc

try:
    import amici.petab_import
    import amici.petab_objective
    import petab

    import pyabc.petab
except ImportError:
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture(params=petabtests.CASES_LIST)
def case(request):
    """A single test case."""
    return request.param


def test_petab_suite(case):
    """Execute a given case from the PEtab test suite."""
    try:
        execute_case(case)
        logger.info(f"Case {case} passed")
    except Skipped:
        logger.info(f"Case {case} skipped")
    except Exception as e:
        logger.error(f"Case {case} failed")
        raise e


def execute_case(case):
    """Wrapper for _execute_case for handling test outcomes"""
    try:
        _execute_case(case)
    except Exception as e:
        if isinstance(e, NotImplementedError) or "timepoint specific" in str(
            e
        ):
            logger.info(
                f"Case {case} expectedly failed. Required functionality is "
                f"not implemented: {e}"
            )
            pytest.skip(str(e))
        else:
            raise e


def _execute_case(case):
    """Run a single PEtab test suite case"""
    case = petabtests.test_id_str(case)
    logger.info(f"Case {case}")

    # case folder
    case_dir = os.path.join(petabtests.CASES_DIR, case)

    # load solution
    solution = petabtests.load_solution(case, format='sbml')
    gt_chi2 = solution[petabtests.CHI2]
    gt_llh = solution[petabtests.LLH]
    gt_simulation_dfs = solution[petabtests.SIMULATION_DFS]
    tol_chi2 = solution[petabtests.TOL_CHI2]
    tol_llh = solution[petabtests.TOL_LLH]
    tol_simulations = solution[petabtests.TOL_SIMULATIONS]

    # unique folder for compiled amici model
    output_folder = f'amici_models/model_{case}'

    # import petab problem
    yaml_file = os.path.join(case_dir, petabtests.problem_yaml_name(case))

    # create problem
    petab_problem = petab.Problem.from_yaml(yaml_file)

    # compile amici
    if output_folder not in sys.path:
        sys.path.insert(0, output_folder)
    amici_model = amici.petab_import.import_petab_problem(
        petab_problem=petab_problem,
        model_output_dir=output_folder,
        generate_sensitivity_code=False,
    )
    solver = amici_model.getSolver()

    # import to pyabc
    importer = pyabc.petab.AmiciPetabImporter(
        petab_problem, amici_model, solver
    )
    model = importer.create_model(return_rdatas=True)

    # simulate
    problem_parameters = importer.get_nominal_parameters()
    ret = model(problem_parameters)

    llh = ret['llh']

    # extract results
    rdatas = ret['rdatas']
    chi2 = sum(rdata['chi2'] for rdata in rdatas)
    simulation_df = amici.petab_objective.rdatas_to_measurement_df(
        rdatas, amici_model, importer.petab_problem.measurement_df
    )
    petab.check_measurement_df(
        simulation_df, importer.petab_problem.observable_df
    )
    simulation_df = simulation_df.rename(
        columns={petab.MEASUREMENT: petab.SIMULATION}
    )
    simulation_df[petab.TIME] = simulation_df[petab.TIME].astype(int)

    # check if matches
    chi2s_match = petabtests.evaluate_chi2(chi2, gt_chi2, tol_chi2)
    llhs_match = petabtests.evaluate_llh(llh, gt_llh, tol_llh)
    simulations_match = petabtests.evaluate_simulations(
        [simulation_df], gt_simulation_dfs, tol_simulations
    )

    # log matches
    logger.log(
        logging.INFO if chi2s_match else logging.ERROR,
        f"CHI2: simulated: {chi2}, expected: {gt_chi2},"
        f" match = {chi2s_match}",
    )
    logger.log(
        logging.INFO if simulations_match else logging.ERROR,
        f"LLH: simulated: {llh}, expected: {gt_llh}, " f"match = {llhs_match}",
    )
    logger.log(
        logging.INFO if simulations_match else logging.ERROR,
        f"Simulations: match = {simulations_match}",
    )

    if not all([llhs_match, chi2s_match, simulations_match]):
        logger.error(f"Case {case} failed.")
        raise AssertionError(
            f"Case {case}: Test results do not match " "expectations"
        )

    logger.info(f"Case {case} passed.")
