import sys
import os
import numpy as np
import scipy.stats
import pandas as pd
import pytest
import git
import itertools
import matplotlib.pyplot as plt

import amici.petab_import
import petab
import petab.C as C
import pyabc.petab
import pyabc.petab.base


@pytest.fixture(params=itertools.product(
    [petab.C.LIN, petab.C.LOG, petab.C.LOG10],
    [*petab.C.PRIOR_TYPES, None]))
def prior_specs(request):
    """A one-line parameter df for a given prior type."""
    scale, prior_type = request.param
    var1, var2 = 0.2, 0.9
    if prior_type:
        # dataframe with objective prior
        df = pd.DataFrame({
            C.PARAMETER_ID: ['p1'],
            C.ESTIMATE: [1],
            C.PARAMETER_SCALE: [scale],
            C.LOWER_BOUND: [np.nan],
            C.UPPER_BOUND: [np.nan],
            C.OBJECTIVE_PRIOR_TYPE: [prior_type],
            C.OBJECTIVE_PRIOR_PARAMETERS: [f"{var1};{var2}"],
        })
    else:
        # also consider the case that no prior is specified, resulting in a
        #  parameter scale uniform prior within the rescaled bounds

        # unscale variables
        unscaled_var1, unscaled_var2 = var1, var2
        if scale == C.LOG:
            unscaled_var1, unscaled_var2 = np.exp([var1, var2])
        elif scale == C.LOG10:
            unscaled_var1, unscaled_var2 = 10.**var1, 10.**var2
        # dataframe without objective prior
        df = pd.DataFrame({
            C.PARAMETER_ID: ['p1'],
            C.ESTIMATE: [1],
            C.PARAMETER_SCALE: [scale],
            C.LOWER_BOUND: [unscaled_var1],
            C.UPPER_BOUND: [unscaled_var2],
        })
        # expected default if objective type not set
        prior_type = C.PARAMETER_SCALE_UNIFORM
    yield scale, prior_type, var1, var2, df


def test_petab_prior(prior_specs):
    """Test whether the prior is correctly defined by sampling from it."""
    # need to fix random seed due to stochastics of multiple testing
    np.random.seed(0)

    # extract settings
    scale, prior_type, var1, var2, parameter_df = prior_specs

    # create prior from petab data frame
    pyabc_prior = pyabc.petab.base.create_prior(parameter_df)

    # generate random samples
    n_samples = 10000
    samples = pyabc_prior.rvs(size=n_samples)['p1']

    # -- UNIFORM COVERAGE -- #

    # check that uniform parameters fill their domain
    if prior_type in [C.UNIFORM, C.PARAMETER_SCALE_UNIFORM]:
        assert (samples >= var1).all() and (samples <= var2).all()
        assert ((samples >= var2 - (var2-var1)*0.01).any() and
                (samples <= var1 + (var2-var1)*0.01).any())

    # -- MEAN AND VARIANCE -- #

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

    # compare means and variances
    assert mean_th * tol < mean < mean_th * 1 / tol
    assert var_th * tol < var < var_th * 1 / tol

    # -- KOLMOGOROV-SMIRNOV CDF COMPARISON -- #

    # create distribution object
    if prior_type in [C.UNIFORM, C.PARAMETER_SCALE_UNIFORM]:
        distr = scipy.stats.uniform(loc=var1, scale=var2-var1)
    elif prior_type in [C.NORMAL, C.PARAMETER_SCALE_NORMAL]:
        distr = scipy.stats.norm(loc=var1, scale=var2)
    elif prior_type in [C.LAPLACE, C.PARAMETER_SCALE_LAPLACE]:
        distr = scipy.stats.laplace(loc=var1, scale=var2)
    elif prior_type == C.LOG_NORMAL:
        distr = scipy.stats.lognorm(s=var2, loc=0, scale=np.exp(var1))
    elif prior_type == C.LOG_LAPLACE:
        distr = scipy.stats.loglaplace(c=1/var2, scale=np.exp(var1))
    else:
        raise ValueError(f"Unexpected prior type: {prior_type}")

    # perform KS test
    _, p_value = scipy.stats.kstest(samples, distr.cdf)
    # at least check that there are no highly significant differences
    assert p_value > 1e-2

    # dummy check that the test recognizes use of the wrong distribution
    if prior_type in [C.NORMAL, C.PARAMETER_SCALE_NORMAL]:
        distr = scipy.stats.laplace(loc=var1, scale=var2)
        _, p_value = scipy.stats.kstest(samples, distr.cdf)
        assert p_value < 1e-5


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
    }).set_index(C.PARAMETER_ID)

    # create prior from petab data frame
    pyabc_prior = pyabc.petab.base.create_prior(parameter_df)

    # create a sample
    sample = pyabc_prior.rvs()

    # check the entries
    assert set(sample.keys()) == {'p1', 'p3'}


def test_get_nominal_parameters():
    """Test extraction of nominal parameters."""
    parameter_df = pd.DataFrame({
        C.PARAMETER_ID: ['p1', 'p2', 'p3'],
        C.NOMINAL_VALUE: [2] * 3,
        C.LOWER_BOUND: [1] * 3,
        C.UPPER_BOUND: [3] * 3,
        C.ESTIMATE: [1] * 3,
        C.PARAMETER_SCALE: [C.LIN, C.LOG, C.LOG10],
        C.OBJECTIVE_PRIOR_TYPE: [
            C.PARAMETER_SCALE_UNIFORM, C.PARAMETER_SCALE_UNIFORM, C.UNIFORM],
        C.OBJECTIVE_PRIOR_PARAMETERS: ["1;4", "1;3", "0;0.7"],
    }).set_index(C.PARAMETER_ID)

    # expected nominal parameters
    expected = {
        C.LIN: pyabc.Parameter({'p1': 2, 'p2': 2, 'p3': 2}),
        'prior': pyabc.Parameter({'p1': 2, 'p2': np.log(2), 'p3': 2}),
        'scaled': pyabc.Parameter({'p1': 2, 'p2': np.log(2),
                                   'p3': np.log10(2)}),
    }

    # get scales
    prior_scales, scaled_scales = pyabc.petab.base.get_scales(parameter_df)

    # run for all target_scales
    for scale in expected:
        x_nominal = pyabc.petab.base.get_nominal_parameters(
            parameter_df, scale, prior_scales, scaled_scales)
        assert x_nominal == expected[scale]

    # raise
    with pytest.raises(ValueError):
        pyabc.petab.base.get_nominal_parameters(
            parameter_df, C.LOG, prior_scales, scaled_scales)


def test_get_bounds():
    """Test that bounds are extracted correctly."""
    parameter_df = pd.DataFrame({
        C.PARAMETER_ID: ['p1', 'p2', 'p3', 'p4'],
        C.ESTIMATE: [1] * 4,
        C.PARAMETER_SCALE: [C.LIN, C.LOG, C.LOG10, C.LOG10],
        C.LOWER_BOUND: [1] * 4,
        C.UPPER_BOUND: [3] * 4,
        C.OBJECTIVE_PRIOR_TYPE: [
            C.PARAMETER_SCALE_UNIFORM, C.UNIFORM, C.PARAMETER_SCALE_UNIFORM,
            C.LAPLACE],
        C.OBJECTIVE_PRIOR_PARAMETERS: ["1;4", "1;3", "0;0.7", "1;4"],
    }).set_index(C.PARAMETER_ID)

    # most common use case
    prior_scales, scaled_scales = pyabc.petab.base.get_scales(parameter_df)
    bounds = pyabc.petab.base.get_bounds(
        parameter_df, 'prior', prior_scales, scaled_scales, use_prior=True)
    assert bounds == {'p1': (1, 4), 'p2': (1, 3), 'p3': (0, 0.7), 'p4': (1, 3)}

    # no prior parameter overrides
    prior_scales, scaled_scales = pyabc.petab.base.get_scales(parameter_df)
    bounds = pyabc.petab.base.get_bounds(
        parameter_df, 'prior', prior_scales, scaled_scales, use_prior=False)
    assert bounds == {'p1': (1, 3), 'p2': (1, 3),
                      'p3': (np.log10(1), np.log10(3)), 'p4': (1, 3)}

    # all on scale
    prior_scales, scaled_scales = pyabc.petab.base.get_scales(parameter_df)
    bounds = pyabc.petab.base.get_bounds(
        parameter_df, 'scaled', prior_scales, scaled_scales, use_prior=False)
    assert bounds == {'p1': (1, 3), 'p2': (np.log(1), np.log(3)),
                      'p3': (np.log10(1), np.log10(3)),
                      'p4': (np.log10(1), np.log10(3))}

    # all off scale
    prior_scales, scaled_scales = pyabc.petab.base.get_scales(parameter_df)
    bounds = pyabc.petab.base.get_bounds(
        parameter_df, C.LIN, prior_scales, scaled_scales, use_prior=False)
    assert bounds == {'p1': (1, 3), 'p2': (1, 3), 'p3': (1, 3), 'p4': (1, 3)}

    # raise
    with pytest.raises(ValueError):
        pyabc.petab.base.get_bounds(
            parameter_df, C.LOG, prior_scales, scaled_scales, use_prior=True)


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
    model_name = 'Boehm_JProteomeRes2014'
    petab_problem = petab.Problem.from_yaml(os.path.join(
        benchmark_dir, 'Benchmark-Models', model_name, model_name + '.yaml'))

    # compile amici
    output_folder = f'amici_models/{model_name}'
    if output_folder not in sys.path:
        sys.path.insert(0, output_folder)
    model = amici.petab_import.import_petab_problem(
        petab_problem, model_output_dir=output_folder)
    solver = model.getSolver()

    # import to pyabc
    importer = pyabc.petab.AmiciPetabImporter(petab_problem, model, solver)

    # extract required objects
    prior = importer.create_prior()
    model = importer.create_model()
    kernel = importer.create_kernel()

    # call model
    assert np.isclose(
        model(importer.get_nominal_parameters())['llh'], -138.221996)

    # mini analysis, just to run it
    temperature = pyabc.Temperature(
        enforce_exact_final_temperature=False,
        schemes=[pyabc.AcceptanceRateScheme()])
    acceptor = pyabc.StochasticAcceptor()

    abc = pyabc.ABCSMC(model, prior, kernel, eps=temperature,
                       acceptor=acceptor, population_size=10)
    abc.new(pyabc.storage.create_sqlite_db_id(), None)
    h = abc.run(max_nr_populations=1)

    # visualize
    pyabc.visualization.plot_kde_matrix_highlevel(
        h, limits=importer.get_bounds(),
        refval=importer.get_nominal_parameters(), refval_color='grey',
        names=importer.get_parameter_names(),
    )
    plt.close()
