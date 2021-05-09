import multiprocessing
import pytest
import numpy as np
import scipy.stats as st
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pyabc.sampler import (SingleCoreSampler,
                           MappingSampler,
                           MulticoreParticleParallelSampler,
                           DaskDistributedSampler,
                           ConcurrentFutureSampler,
                           MulticoreEvalParallelSampler,
                           RedisEvalParallelSamplerServerStarter,
                           RedisStaticSamplerServerStarter)
import pyabc
import logging
import os
import tempfile

logger = logging.getLogger(__name__)


def multi_proc_map(f, x):
    with multiprocessing.Pool() as pool:
        res = pool.map(f, x)
    return res


class GenericFutureWithProcessPool(ConcurrentFutureSampler):
    def __init__(self, map_=None):
        cfuture_executor = ProcessPoolExecutor(max_workers=8)
        client_max_jobs = 8
        super().__init__(cfuture_executor, client_max_jobs)


class GenericFutureWithProcessPoolBatch(ConcurrentFutureSampler):
    def __init__(self, map_=None):
        cfuture_executor = ProcessPoolExecutor(max_workers=8)
        client_max_jobs = 8
        batch_size = 15
        super().__init__(cfuture_executor, client_max_jobs,
                         batch_size=batch_size)


class GenericFutureWithThreadPool(ConcurrentFutureSampler):
    def __init__(self, map_=None):
        cfuture_executor = ThreadPoolExecutor(max_workers=8)
        client_max_jobs = 8
        super().__init__(cfuture_executor, client_max_jobs)


class MultiProcessingMappingSampler(MappingSampler):
    def __init__(self, map_=None):
        super().__init__(multi_proc_map)


class DaskDistributedSamplerBatch(DaskDistributedSampler):
    def __init__(self, map_=None):
        batch_size = 20
        super().__init__(batch_size=batch_size)


class WrongOutputSampler(SingleCoreSampler):
    def sample_until_n_accepted(
            self, n, simulate_one, t, *,
            max_eval=np.inf, all_accepted=False, ana_vars=None):
        return super().sample_until_n_accepted(
            n+1, simulate_one, t, max_eval=max_eval,
            all_accepted=all_accepted, ana_vars=ana_vars)


def RedisEvalParallelSamplerWrapper(**kwargs):
    kwargs.setdefault('batch_size', 5)
    kwargs.setdefault('catch', False)
    return RedisEvalParallelSamplerServerStarter(**kwargs)


def RedisEvalParallelSamplerWaitForAllWrapper(**kwargs):
    kwargs.setdefault('batch_size', 5)
    kwargs.setdefault('catch', False)
    return RedisEvalParallelSamplerServerStarter(
        wait_for_all_samples=True, **kwargs)


def RedisEvalParallelSamplerLookAheadDelayWrapper(**kwargs):
    kwargs.setdefault('catch', False)
    return RedisEvalParallelSamplerServerStarter(
        look_ahead=True, look_ahead_delay_evaluation=True, **kwargs)


def RedisStaticSamplerWrapper(**kwargs):
    kwargs.setdefault('catch', False)
    return RedisStaticSamplerServerStarter(**kwargs)


def PicklingMulticoreParticleParallelSampler():
    return MulticoreParticleParallelSampler(pickle=True)


def PicklingMulticoreEvalParallelSampler():
    return MulticoreEvalParallelSampler(pickle=True)


@pytest.fixture(params=[SingleCoreSampler,
                        RedisEvalParallelSamplerWrapper,
                        RedisEvalParallelSamplerWaitForAllWrapper,
                        RedisEvalParallelSamplerLookAheadDelayWrapper,
                        RedisStaticSamplerWrapper,
                        MulticoreEvalParallelSampler,
                        MultiProcessingMappingSampler,
                        MulticoreParticleParallelSampler,
                        PicklingMulticoreParticleParallelSampler,
                        PicklingMulticoreEvalParallelSampler,
                        MappingSampler,
                        DaskDistributedSampler,
                        DaskDistributedSamplerBatch,
                        GenericFutureWithThreadPool,
                        GenericFutureWithProcessPool,
                        GenericFutureWithProcessPoolBatch,
                        ])
def sampler(request):
    s = request.param()
    try:
        yield s
    finally:
        # release all resources
        try:
            s.shutdown()
        except AttributeError:
            pass


@pytest.fixture
def redis_starter_sampler(request):
    s = RedisEvalParallelSamplerServerStarter(batch_size=5)
    try:
        yield s
    finally:
        # release all resources
        s.shutdown()


def basic_testcase():
    """A simple test model."""
    def model(p):
        return {"y": p['p0'] + 0.1 * np.random.randn(10)}

    prior = pyabc.Distribution(
        p0=pyabc.RV('uniform', -5, 10), p1=pyabc.RV('uniform', -2, 2))

    def distance(y1, y2):
        return np.abs(y1['y'] - y2['y']).sum()

    obs = {'y': 1}
    return model, prior, distance, obs


def test_two_competing_gaussians_multiple_population(db_path, sampler):
    two_competing_gaussians_multiple_population(
        db_path, sampler, 1)


def test_two_competing_gaussians_multiple_population_2_evaluations(
        db_path, sampler):
    two_competing_gaussians_multiple_population(db_path, sampler, 2)


def two_competing_gaussians_multiple_population(db_path, sampler, n_sim):
    # Define a gaussian model
    sigma = .5

    def model(args):
        return {"y": st.norm(args['x'], sigma).rvs()}

    # We define two models, but they are identical so far
    models = [model, model]
    models = list(map(pyabc.SimpleModel, models))

    # However, our models' priors are not the same. Their mean differs.
    mu_x_1, mu_x_2 = 0, 1
    parameter_given_model_prior_distribution = [
        pyabc.Distribution(x=pyabc.RV("norm", mu_x_1, sigma)),
        pyabc.Distribution(x=pyabc.RV("norm", mu_x_2, sigma)),
    ]

    # We plug all the ABC setups together
    nr_populations = 2
    pop_size = pyabc.ConstantPopulationSize(23)
    abc = pyabc.ABCSMC(models, parameter_given_model_prior_distribution,
                       pyabc.PercentileDistance(measures_to_use=["y"]),
                       pop_size,
                       eps=pyabc.MedianEpsilon(),
                       sampler=sampler)

    # Finally we add meta data such as model names and
    # define where to store the results
    # y_observed is the important piece here: our actual observation.
    y_observed = 1
    abc.new(db_path, {"y": y_observed})

    # We run the ABC with 3 populations max
    minimum_epsilon = .05
    history = abc.run(minimum_epsilon, max_nr_populations=nr_populations)

    # Evaluate the model probabilities
    mp = history.get_model_probabilities(history.max_t)

    def p_y_given_model(mu_x_model):
        res = st.norm(mu_x_model, np.sqrt(sigma**2 + sigma**2)).pdf(y_observed)
        return res

    p1_expected_unnormalized = p_y_given_model(mu_x_1)
    p2_expected_unnormalized = p_y_given_model(mu_x_2)
    p1_expected = p1_expected_unnormalized / (p1_expected_unnormalized
                                              + p2_expected_unnormalized)
    p2_expected = p2_expected_unnormalized / (p1_expected_unnormalized
                                              + p2_expected_unnormalized)
    assert history.max_t == nr_populations-1
    # the next line only tests if we obtain correct numerical types
    try:
        mp0 = mp.p[0]
    except KeyError:
        mp0 = 0

    try:
        mp1 = mp.p[1]
    except KeyError:
        mp1 = 0

    assert abs(mp0 - p1_expected) + abs(mp1 - p2_expected) < np.inf

    # check that sampler only did nr_particles samples in first round
    pops = history.get_all_populations()
    # since we had calibration (of epsilon), check that was saved
    pre_evals = pops[pops['t'] == pyabc.History.PRE_TIME]['samples'].values
    assert pre_evals >= pop_size.nr_particles
    # our samplers should not have overhead in calibration, except batching
    batch_size = sampler.batch_size if hasattr(sampler, 'batch_size') else 1
    max_expected = pop_size.nr_particles + batch_size - 1
    if pre_evals > max_expected:
        # Violations have been observed occasionally for the redis server
        # due to runtime conditions with the increase of the evaluations
        # counter. This could be overcome, but as it usually only happens
        # for low-runtime models, this should not be a problem. Thus, only
        # print a warning here.
        logger.warning(
            f"Had {pre_evals} simulations in the calibration iteration, "
            f"but a maximum of {max_expected} would have been sufficient for "
            f"the population size of {pop_size.nr_particles}.")


def test_progressbar(sampler):
    """Test whether using a progress bar gives any errors."""
    model, prior, distance, obs = basic_testcase()

    abc = pyabc.ABCSMC(
        model, prior, distance, sampler=sampler, population_size=20)
    abc.new(db=pyabc.create_sqlite_db_id(), observed_sum_stat=obs)
    abc.run(max_nr_populations=3)


def test_in_memory(redis_starter_sampler):
    db_path = "sqlite://"
    two_competing_gaussians_multiple_population(db_path,
                                                redis_starter_sampler, 1)


def test_wrong_output_sampler():
    sampler = WrongOutputSampler()

    def simulate_one():
        return pyabc.Particle(m=0, parameter={}, weight=0,
                              sum_stat={}, distance=42,
                              accepted=True)
    with pytest.raises(AssertionError):
        sampler.sample_until_n_accepted(5, simulate_one, 0)


def test_redis_multiprocess():
    def simulate_one():
        accepted = np.random.randint(2)
        return pyabc.Particle(0, {}, 0.1, {}, 0, accepted)

    sampler = RedisEvalParallelSamplerServerStarter(
        batch_size=3, workers=1, processes_per_worker=2)
    try:
        # id needs to be set
        sampler.set_analysis_id("ana_id")

        sample = sampler.sample_until_n_accepted(10, simulate_one, 0)
        assert 10 == len(sample.get_accepted_population())
    finally:
        sampler.shutdown()


def test_redis_catch_error():
    def model(pars):
        if np.random.uniform() < 0.1:
            raise ValueError("error")
        return {'s0': pars['p0'] + 0.2 * np.random.uniform()}

    def distance(s0, s1):
        return abs(s0['s0'] - s1['s0'])

    prior = pyabc.Distribution(p0=pyabc.RV("uniform", 0, 10))
    sampler = RedisEvalParallelSamplerServerStarter(
        batch_size=3, workers=1, processes_per_worker=1)
    try:
        abc = pyabc.ABCSMC(
            model, prior, distance, sampler=sampler, population_size=10)

        db_file = "sqlite:///" + os.path.join(tempfile.gettempdir(), "test.db")
        data = {'s0': 2.8}
        abc.new(db_file, data)
        abc.run(minimum_epsilon=.1, max_nr_populations=3)
    finally:
        sampler.shutdown()


def test_redis_pw_protection():
    def simulate_one():
        accepted = np.random.randint(2)
        return pyabc.Particle(0, {}, 0.1, {}, 0, accepted)

    sampler = RedisEvalParallelSamplerServerStarter(  # noqa: S106
        password="daenerys")
    try:
        # needs to be always set
        sampler.set_analysis_id("ana_id")
        sample = sampler.sample_until_n_accepted(10, simulate_one, 0)
        assert 10 == len(sample.get_accepted_population())
    finally:
        sampler.shutdown()


def test_redis_continuous_analyses():
    """Test correct behavior of the redis server with multiple analyses."""
    sampler = RedisEvalParallelSamplerServerStarter()
    try:
        sampler.set_analysis_id("id1")
        # try "starting a new run while the old one has not finished yet"
        with pytest.raises(AssertionError) as e:
            sampler.set_analysis_id("id2")
        assert "busy with an analysis " in str(e.value)
        # after stopping it should work
        sampler.stop()
        sampler.set_analysis_id("id2")
    finally:
        sampler.shutdown()


def test_redis_subprocess():
    """Test whether the instructed redis sampler allows worker subprocesses."""
    # print worker output
    logging.getLogger("Redis-Worker").addHandler(logging.StreamHandler())

    def model_process(p, pipe):
        """The actual model."""
        pipe.send({"y": p['p0'] + 0.1 * np.random.randn(10)})

    def model(p):
        """Model calling a subprocess."""
        parent, child = multiprocessing.Pipe()
        proc = multiprocessing.Process(target=model_process, args=(p, child))
        proc.start()
        res = parent.recv()
        proc.join()
        return res

    prior = pyabc.Distribution(
        p0=pyabc.RV('uniform', -5, 10), p1=pyabc.RV('uniform', -2, 2))

    def distance(y1, y2):
        return np.abs(y1['y'] - y2['y']).sum()

    obs = {'y': 1}
    # False as daemon argument is ok, True and None are not allowed
    sampler = RedisEvalParallelSamplerServerStarter(
        workers=1, processes_per_worker=2, daemon=False)
    try:
        abc = pyabc.ABCSMC(
            model, prior, distance, sampler=sampler,
            population_size=10)
        abc.new(pyabc.create_sqlite_db_id(), obs)
        # would just never return if model evaluation fails
        abc.run(max_nr_populations=3)
    finally:
        sampler.shutdown()


def test_redis_look_ahead():
    """Test the redis sampler in look-ahead mode."""
    model, prior, distance, obs = basic_testcase()
    eps = pyabc.ListEpsilon([20, 10, 5])
    # spice things up with an adaptive population size
    pop_size = pyabc.AdaptivePopulationSize(
        start_nr_particles=50, mean_cv=0.5, max_population_size=50)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv') as fh:
        sampler = RedisEvalParallelSamplerServerStarter(
            look_ahead=True, look_ahead_delay_evaluation=False,
            log_file=fh.name)
        try:
            abc = pyabc.ABCSMC(
                model, prior, distance, sampler=sampler,
                population_size=pop_size, eps=eps)
            abc.new(pyabc.create_sqlite_db_id(), obs)
            h = abc.run(max_nr_populations=3)
        finally:
            sampler.shutdown()

        assert h.n_populations == 3

        # read log file
        df = pd.read_csv(fh.name, sep=',')
        assert (df.n_lookahead > 0).any()
        assert (df.n_lookahead_accepted > 0).any()
        assert (df.n_preliminary == 0).all()

        # check history proposal ids
        for t in range(0, h.max_t + 1):
            pop = h.get_population(t=t)
            pop_size = len(pop)
            n_lookahead_pop = len(
                [p for p in pop.get_list() if p.proposal_id == -1])
            assert min(
                pop_size, int(df.loc[df.t == t, 'n_lookahead_accepted'])) \
                == n_lookahead_pop


def test_redis_look_ahead_error():
    """Test whether the look-ahead mode fails as expected."""
    model, prior, distance, obs = basic_testcase()
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv') as fh:
        sampler = RedisEvalParallelSamplerServerStarter(
            look_ahead=True, look_ahead_delay_evaluation=False,
            log_file=fh.name)
        args_list = [
            {'eps': pyabc.MedianEpsilon()},
            {'distance_function': pyabc.AdaptivePNormDistance()}]
        for args in args_list:
            if 'distance_function' not in args:
                args['distance_function'] = distance
            try:
                with pytest.raises(AssertionError) as e:
                    abc = pyabc.ABCSMC(
                        model, prior, sampler=sampler,
                        population_size=10, **args)
                    abc.new(pyabc.create_sqlite_db_id(), obs)
                    abc.run(max_nr_populations=3)
                assert "cannot be used in look-ahead mode" in str(e.value)
            finally:
                sampler.shutdown()


def test_redis_look_ahead_delayed():
    """Test the look-ahead sampler with delayed evaluation in an adaptive
    setup."""
    model, prior, distance, obs = basic_testcase()
    # spice things up with an adaptive population size
    pop_size = pyabc.AdaptivePopulationSize(
        start_nr_particles=50, mean_cv=0.5, max_population_size=50)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv') as fh:
        sampler = RedisEvalParallelSamplerLookAheadDelayWrapper(
            log_file=fh.name)
        try:
            abc = pyabc.ABCSMC(
                model, prior, distance, sampler=sampler,
                population_size=pop_size)
            abc.new(pyabc.create_sqlite_db_id(), obs)
            h = abc.run(max_nr_populations=3)
        finally:
            sampler.shutdown()
        # read log file
        df = pd.read_csv(fh.name, sep=',')
        assert (df.n_lookahead > 0).any()
        assert (df.n_lookahead_accepted > 0).any()
        # in delayed mode, all look-aheads must have been preliminary
        assert (df.n_lookahead == df.n_preliminary).all()
        print(df)

        # check history proposal ids
        for t in range(0, h.max_t + 1):
            pop = h.get_population(t=t)
            pop_size = len(pop)
            n_lookahead_pop = len(
                [p for p in pop.get_list() if p.proposal_id == -1])
            assert min(
                pop_size, int(df.loc[df.t == t, 'n_lookahead_accepted'])) \
                == n_lookahead_pop
