import os
import numpy as np
import pandas as pd
import pytest
import git
import itertools

import amici.petab_import
import petab
import petab.C as C
import pyabc.petab
import pyabc.petab.base


@pytest.fixture(params=itertools.product(
    [petab.C.LIN, petab.C.LOG, petab.C.LOG10],
    petab.C.PRIOR_TYPES))
def prior_specs(request):
    """A one-line parameter df for a given prior type."""
    scale, prior_type = request.param
    var1, var2 = 0.2, 0.9
    df = pd.DataFrame({
        C.PARAMETER_ID: ['p1'],
        C.ESTIMATE: [1],
        C.PARAMETER_SCALE: [scale],
        C.LOWER_BOUND: [np.nan],
        C.UPPER_BOUND: [np.nan],
        C.OBJECTIVE_PRIOR_TYPE: [prior_type],
        C.OBJECTIVE_PRIOR_PARAMETERS: [f"{var1};{var2}"],
    })
    return scale, prior_type, var1, var2, df


def test_petab_prior(prior_specs):
    """Test whether the prior is correctly defined by sampling from it."""
    # extract settings
    scale, prior_type, var1, var2, parameter_df = prior_specs

    # create prior from petab data frame
    pyabc_prior = pyabc.petab.base.create_prior(parameter_df)

    # generate random samples
    n_samples = 10000
    samples = pyabc_prior.rvs(size=n_samples)['p1']

    # check that uniform parameters fill their domain
    if prior_type in [C.UNIFORM, C.PARAMETER_SCALE_UNIFORM]:
        assert (samples >= var1).all() and (samples <= var2).all()
        assert ((samples >= var2 - (var2-var1)*0.01).any() and
                (samples <= var1 + (var2-var1)*0.01).any())

    # sample mean and variance
    mean = np.mean(samples)
    var = np.var(samples)

    # ground truth mean and variance
    if prior_type in [C.UNIFORM, C.PARAMETER_SCALE_UNIFORM]:
        mean_th = var1 + (var2 - var1) / 2
        var_th = (var2 - var1)**2 / 12
    elif prior_type in [C.NORMAL, C.PARAMETER_SCALE_NORMAL]:
        mean_th = var1
        var_th = var2**2
    elif prior_type in [C.LAPLACE, C.PARAMETER_SCALE_LAPLACE]:
        mean_th = var1
        var_th = 2 * var2**2
    elif prior_type == C.LOG_NORMAL:
        # just log-transform all
        mean = np.mean(np.log(samples))
        var = np.var(np.log(samples))
        mean_th = var1
        var_th = var2**2
    elif prior_type == C.LOG_LAPLACE:
        # just log-transform all
        mean = np.mean(np.log(samples))
        var = np.var(np.log(samples))
        mean_th = var1
        var_th = 2 * var2**2
    else:
        raise ValueError(f"Unexpected prior type: {prior_type}")

    # multiplicative tolerance of sample vs ground truth variables
    tol = 0.8
    assert mean_th * tol < mean < mean_th * 1 / tol
    assert var_th * tol < var < var_th * 1 / tol


def test_parameter_fixing():
    """Test that only free parameters are exposed to pyABC."""
    # define problem with fixed parameters
    parameter_df = pd.DataFrame({
        C.PARAMETER_ID: ['p1', 'p2', 'p3'],
        C.ESTIMATE: [1, 0, 1],
        C.PARAMETER_SCALE: [C.LIN] * 3,
        C.LOWER_BOUND: [0] * 3,
        C.UPPER_BOUND: [1] * 3,
        C.OBJECTIVE_PRIOR_TYPE: [C.PARAMETER_SCALE_UNIFORM] * 3,
    })

    # create prior from petab data frame
    pyabc_prior = pyabc.petab.base.create_prior(parameter_df)

    # create a sample
    sample = pyabc_prior.rvs()

    # check the entries
    assert set(sample.keys()) == {'p1', 'p3'}


def test_pipeline():
    """Test the petab pipeline on an application model."""
    # download archive
    benchmark_dir = "doc/examples/tmp/benchmark-models-petab"
    if not os.path.exists(benchmark_dir):
        git.Repo.clone_from(
            "https://github.com/benchmarking-initiative"
            "/benchmark-models-petab.git",
            benchmark_dir, depth=1)
    g = git.Git(benchmark_dir)

    # update repo if online
    try:
        g.pull()
    except git.exc.GitCommandError:
        pass

    # create problem
    petab_problem = petab.Problem.from_yaml(os.path.join(
        benchmark_dir, "Benchmark-Models",
        "Boehm_JProteomeRes2014", "Boehm_JProteomeRes2014.yaml"))

    # compile amici
    model = amici.petab_import.import_petab_problem(petab_problem)
    solver = model.getSolver()

    # import to pyabc
    importer = pyabc.petab.AmiciPetabImporter(petab_problem, model, solver)

    # extract required objects
    prior = importer.create_prior()
    model = importer.create_model()
    kernel = importer.create_kernel()

    # call model
    assert np.isclose(
        model(petab_problem.x_nominal_free_scaled)['llh'], -138.221996)

    # mini analysis, just to run it
    temperature = pyabc.Temperature(
        enforce_exact_final_temperature=False,
        schemes=[pyabc.AcceptanceRateScheme()])
    acceptor = pyabc.StochasticAcceptor()

    abc = pyabc.ABCSMC(model, prior, kernel, eps=temperature,
                       acceptor=acceptor, population_size=10)
    abc.new(pyabc.storage.create_sqlite_db_id(), None)
    abc.run(max_nr_populations=1)
