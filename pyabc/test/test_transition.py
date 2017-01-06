from pyabc.transition import NotEnoughParticles
from pyabc import MultivariateNormalTransition
import pandas as pd
import numpy as np
import pytest
from pyabc import GridSearchCV

def data(n):
    df = pd.DataFrame({"a": np.random.rand(n), "b": np.random.rand(n)})
    w = np.ones(len(df)) / len(df)
    return df, w


def data_single(n):
    df = pd.DataFrame({"a": np.random.rand(n)})
    w = np.ones(len(df)) / len(df)
    return df, w


def test_many_particles_single_par():
    df, w = data_single(20)
    m = MultivariateNormalTransition()
    m.fit(df, w)
    m.required_nr_samples(.1)


def test_variance_estimate():
    var_list = []
    for n in [50, 100, 200]:
        m = MultivariateNormalTransition()
        df, w = data(n)
        m.fit(df, w)
        var = m.mean_coefficient_of_variation()
        var_list.append(var)

    assert var_list[0] >= var_list[1]
    assert var_list[1] >= var_list[2]


def test_variance_no_side_effect():
    m = MultivariateNormalTransition()
    df, w = data(60)
    m.fit(df, w)
    # very intrusive test. touches internals of m. not very nice.
    X_orig_id = id(m.X)
    var = m.mean_coefficient_of_variation()
    assert id(m.X) == X_orig_id


def test_particles_no_parameters():
    df = pd.DataFrame(index=[0, 1, 2, 3])
    assert len(df) == 4
    w = np.array([1, 1, 1, 1]) / 4
    m = MultivariateNormalTransition()
    m.fit(df, w)
    with pytest.raises(NotEnoughParticles):
        m.required_nr_samples(.1)


def test_empty():
    # TODO define proper behavior
    df = pd.DataFrame()
    w = np.array([])
    m = MultivariateNormalTransition()
    m.fit(df, w)
    with pytest.raises(NotEnoughParticles):
        m.required_nr_samples(.1)


def test_0_particles():
    # TODO define proper behavior
    df, w = data(0)
    print(df)
    m = MultivariateNormalTransition()
    with pytest.raises(NotEnoughParticles):
        m.fit(df, w)
    with pytest.raises(NotEnoughParticles):
        m.required_nr_samples(.1)


def test_single_particle():
    # TODO define proper behavior
    df, w = data(1)
    m = MultivariateNormalTransition()
    m.fit(df, w)
    m.required_nr_samples(.1)


def test_two_particles():
    # TODO define proper behavior
    df, w = data(2)
    m = MultivariateNormalTransition()
    m.fit(df, w)
    m.required_nr_samples(.1)


def test_many_particles():
    df, w = data(20)
    m = MultivariateNormalTransition()
    m.fit(df, w)
    m.required_nr_samples(.1)


def test_argument_order():
    """
    Dataframes passed to the transition kernels are generated from dicts.
    Order of parameter names is no guaranteed.
    The Transition kernel has to take care of the correct sorting.
    """
    m = MultivariateNormalTransition()
    df, w = data(20)
    m.fit(df, w)
    test = df.iloc[0]
    reversed = test[::-1]
    assert (np.array(test) != np.array(reversed)).all()   # works b/c of even nr of parameters
    assert m.pdf(test) == m.pdf(reversed)


def test_score():
    m = MultivariateNormalTransition()
    df, w = data(20)
    m.fit(df, w)
    score = m.score(df, w)


def test_grid_search():
    m = MultivariateNormalTransition()
    m_grid = GridSearchCV(m, {"scaling": np.logspace(-5, 1.5, 5)})
    df, w = data(20)
    m_grid.fit(df, w)


def test_grid_search_two_samples():
    """
    Supposed to run into problems b/c nr splits > then nr_samples
    """
    cv = 5
    m = MultivariateNormalTransition()
    m_grid = GridSearchCV(m, {"scaling": np.logspace(-5, 1.5, 5)}, cv=cv)
    df, w = data(2)
    m_grid.fit(df, w)
    assert m_grid.cv == cv


def test_grid_search_single_sample():
    """
    Supposed to run into problems b/c nr splits > then nr_samples
    """
    cv = 5
    m = MultivariateNormalTransition()
    m_grid = GridSearchCV(m, {"scaling": np.logspace(-5, 1.5, 5)}, cv=cv)
    df, w = data(1)
    m_grid.fit(df, w)
    assert m_grid.cv == cv