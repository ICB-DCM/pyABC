import pandas as pd
import numpy as np
import pytest

from pyabc.transition import NotEnoughParticles, Transition
from pyabc import (
    ABCSMC, Distribution, GridSearchCV, MultivariateNormalTransition,
    LocalTransition, AggregatedTransition, DiscreteJumpTransition,
    Parameter, RV, create_sqlite_db_id)


class SimpleAggregatedTransition(AggregatedTransition):
    """Two different transitions for the keys."""

    def __init__(self):
        mapping = {'a': LocalTransition(),
                   ('b',): MultivariateNormalTransition()}
        super().__init__(mapping=mapping)


class SimpleAggregatedTransition2(AggregatedTransition):
    """Only a single transitions with two keys."""

    def __init__(self):
        mapping = {('a', 'b'): MultivariateNormalTransition()}
        super().__init__(mapping=mapping)


class SimpleAggregatedTransitionSingle(AggregatedTransition):
    """For a single key."""

    def __init__(self):
        mapping = {'a': MultivariateNormalTransition()}
        super().__init__(mapping=mapping)


@pytest.fixture(params=[LocalTransition,
                        MultivariateNormalTransition,
                        SimpleAggregatedTransition,
                        SimpleAggregatedTransition2])
def transition(request):
    return request.param()


@pytest.fixture(params=[LocalTransition,
                        MultivariateNormalTransition,
                        SimpleAggregatedTransitionSingle])
def transition_single(request):
    return request.param()


def data(n):
    df = pd.DataFrame({"a": np.random.rand(n), "b": np.random.rand(n)})
    w = np.ones(len(df)) / len(df)
    return df, w


def data_single(n):
    df = pd.DataFrame({"a": np.random.rand(n)})
    w = np.ones(len(df)) / len(df)
    return df, w


def test_rvs_return_type(transition: Transition):
    df, w = data(20)
    transition.fit(df, w)
    sample = transition.rvs()
    assert (sample.index == pd.Index(["a", "b"])).all()


def test_pdf_return_types(transition: Transition):
    df, w = data(20)
    transition.fit(df, w)
    single = transition.pdf(df.iloc[0])
    multiple = transition.pdf(df)
    assert isinstance(single, float)
    assert multiple.shape == (20,)


def test_many_particles_single_par(transition_single: Transition):
    df, w = data_single(20)
    transition_single.fit(df, w)
    transition_single.required_nr_samples(.1)


def test_variance_estimate(transition: Transition):
    var_list = []
    for n in [20, 250]:
        df, w = data(n)
        transition.fit(df, w)
        var = transition.mean_cv()
        var_list.append(var)

    assert var_list[0] >= var_list[1]


def test_variance_estimate_higher_n_than_sample(transition: Transition):
    n = 100
    df, w = data(n)
    transition.fit(df, w)

    var_list = []
    for n_test in [n, n*4, n*10]:
        var = transition.mean_cv(n_test)
        var_list.append(var)

    for lower, upper in zip(var_list[:-1], var_list[1:]):
        # add a little buffer to overcome slight random fluctuations
        assert lower + 1e-2 >= upper


def test_variance_no_side_effect(transition: Transition):
    df, w = data(60)
    transition.fit(df, w)
    # very intrusive test. touches internals of m. not very nice.
    X_orig_id = id(transition.X)
    transition.mean_cv()  # this has to be called here
    assert id(transition.X) == X_orig_id


def test_particles_no_parameters(transition: Transition):
    df = pd.DataFrame(index=[0, 1, 2, 3])
    assert len(df) == 4
    w = np.array([1, 1, 1, 1]) / 4
    transition.fit(df, w)
    with pytest.raises(NotEnoughParticles):
        transition.required_nr_samples(.1)


def test_empty(transition):
    # TODO define proper behavior
    df = pd.DataFrame()
    w = np.array([])
    transition.fit(df, w)
    with pytest.raises(NotEnoughParticles):
        transition.required_nr_samples(.1)


def test_0_particles_fit(transition: Transition):
    # TODO define proper behavior
    df, w = data(0)
    with pytest.raises(NotEnoughParticles):
        transition.fit(df, w)


def test_single_particle_fit(transition: Transition):
    # TODO define proper behavior
    df, w = data(1)
    transition.fit(df, w)


def test_single_particle_required_nr_samples(transition: Transition):
    # TODO define proper behavior
    df, w = data(1)
    transition.fit(df, w)
    transition.required_nr_samples(.1)


def test_two_particles_fit(transition: Transition):
    # TODO define proper behavior
    df, w = data(2)
    transition.fit(df, w)


def test_two_particles_required_nr_samples(transition: Transition):
    # TODO define proper behavior
    df, w = data(2)
    transition.fit(df, w)
    transition.required_nr_samples(.1)


def test_many_particles(transition: Transition):
    df, w = data(20)
    transition.fit(df, w)
    transition.required_nr_samples(.1)


def test_argument_order(transition: Transition):
    """
    Dataframes passed to the transition kernels are generated from dicts.
    Order of parameter names is no guaranteed.
    The Transition kernel has to take care of the correct sorting.
    """
    df, w = data(20)
    transition.fit(df, w)
    test = df.iloc[0]
    reversed_ = test[::-1]
    # works b/c of even nr of parameters
    assert (np.array(test) != np.array(reversed_)).all()
    assert transition.pdf(test) == transition.pdf(reversed_)


def test_score(transition: Transition):
    df, w = data(20)
    transition.fit(df, w)
    transition.score(df, w)  # just call it


def test_grid_search_multivariate_normal():
    m = MultivariateNormalTransition()
    m_grid = GridSearchCV(m, {"scaling": np.logspace(-5, 1.5, 5)}, n_jobs=1)
    df, w = data(20)
    m_grid.fit(df, w)


def test_grid_search_two_samples_multivariate_normal():
    """
    Supposed to run into problems b/c nr splits > then nr_samples
    """
    cv = 5
    m = MultivariateNormalTransition()
    m_grid = GridSearchCV(m, {"scaling": np.logspace(-5, 1.5, 5)}, cv=cv,
                          n_jobs=1)
    df, w = data(2)
    m_grid.fit(df, w)
    assert m_grid.cv == cv


def test_grid_search_single_sample_multivariate_normal():
    """
    Supposed to run into problems b/c nr splits > then nr_samples
    """
    cv = 5
    m = MultivariateNormalTransition()
    m_grid = GridSearchCV(m, {"scaling": np.logspace(-5, 1.5, 5)}, cv=cv,
                          n_jobs=1)
    df, w = data(1)
    m_grid.fit(df, w)
    assert m_grid.cv == cv


def test_mean_coefficient_of_variation_sample_not_full_rank(
        transition: Transition):
    """
    This is a test created after I encountered this kind of bug
    """
    n = 13
    df = pd.DataFrame({"a": np.ones(n) * 2,
                       "b": np.ones(n)})
    w = np.ones(len(df)) / len(df)
    transition.fit(df, w)
    transition.mean_cv()


def test_discrete_jump_transition():
    """Test that the DiscreteJumpTransition does what it's supposed to do."""
    # domain
    domain = np.array([5, 4, 2.5, 0.5])
    n_domain = len(domain)

    # sampler
    p_stay = 0.7
    p_move = (1 - p_stay) / (n_domain - 1)
    trans = DiscreteJumpTransition(domain, p_stay=p_stay)

    # fit to distribution
    df = pd.DataFrame({'a': [5, 4, 4, 4, 2.5, 2.5, 0.5]})
    w = np.array([0.0, 0.2, 0.2, 0.1, 0.1, 0.1, 0.3])
    trans.fit(df, w)

    # test sampling
    n_sample = 1000
    res = trans.rvs(n_sample)

    def freq(weight):
        return p_stay * weight + p_move * (1 - weight)

    assert 0 < sum(res.a == 5) / n_sample <= p_move + 0.05
    assert freq(0.45) < sum(res.a == 4) / n_sample < freq(0.55)
    assert freq(0.15) < sum(res.a == 2.5) / n_sample < freq(0.25)
    assert freq(0.25) < sum(res.a == 0.5) / n_sample < freq(0.35)

    # test density calculation
    assert np.isclose(trans.pdf(pd.Series({'a': 5})), freq(0.))
    assert np.isclose(trans.pdf(pd.Series({'a': 4})), freq(0.5))
    assert np.isclose(trans.pdf(pd.Series({'a': 2.5})), freq(0.2))
    assert np.isclose(trans.pdf(pd.Series({'a': 0.5})), freq(0.3))


def test_discrete_jump_transition_errors():
    """Test that the DiscreteJumpTransition correctly raises."""
    # stay probability
    with pytest.raises(ValueError):
        DiscreteJumpTransition(np.array([1, 2, 3]), p_stay=1.1)
    with pytest.raises(ValueError):
        DiscreteJumpTransition(np.array([1, 2, 3]), p_stay=-0.1)

    # fitting
    trans = DiscreteJumpTransition(np.array([1, 2, 3]))
    with pytest.raises(ValueError):
        trans.fit(pd.DataFrame({'a': [1, 1, 2], 'b': [2, 2, 3]}),
                  np.array([1., 1., 1.]))

    # density calculation
    trans.fit(pd.DataFrame({'a': [42, 42, 43]}), np.ones(3))
    with pytest.raises(ValueError):
        trans.pdf(pd.Series({'a': 44}))


def test_model_gets_parameter(transition_single: Transition):
    """Check that we use Parameter objects as model input throughout.

    This should be the case both when the parameter is created from the prior,
    and from the transition.
    """
    def model(p):
        assert isinstance(p, Parameter)
        return {'s0': p['a'] + 0.1 * np.random.normal()}
    prior = Distribution(a=RV('uniform', -5, 10))

    abc = ABCSMC(
        model, prior, transitions=transition_single, population_size=10)
    abc.new(create_sqlite_db_id(), {'s0': 3.5})
    abc.run(max_nr_populations=3)


def test_pipeline(transition: Transition):
    """Test the various transitions in a full pipeline."""
    def model(p):
        return {'s0': p['a'] + p['b'] * np.random.normal()}
    prior = Distribution(a=RV('uniform', -5, 10), b=RV('uniform', 0.01, 0.09))

    abc = ABCSMC(
        model, prior, transitions=transition, population_size=10)
    abc.new(create_sqlite_db_id(), {'s0': 3.5})
    abc.run(max_nr_populations=3)
