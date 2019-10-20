import numpy as np

import pyabc
from pyabc.acceptor import AcceptorResult


def test_simple_function_acceptor():

    def distance(x, x_0):
        return sum(abs(x[key] - x_0[key]) for key in x_0)

    def dummy_accept(dist, eps, x, x_0, t, par):
        d = dist(x, x_0)
        return AcceptorResult(d, d < eps(t))

    x = {'s0': 1, 's1': 0}
    y = {'s0': 2, 's1': 2}

    acceptor = pyabc.SimpleFunctionAcceptor(dummy_accept)

    ret = acceptor(distance_function=distance, eps=lambda t: 0.1,
                   x=x, x_0=y, t=0, par=None)

    assert isinstance(ret, AcceptorResult)
    assert ret.distance == 3

    # test integration

    def model(par):
        return {'s0': par['p0'] + 1, 's1': 42}

    prior = pyabc.Distribution(p0=pyabc.RV('uniform', -5, 10))
    abc = pyabc.ABCSMC(
        model, prior, distance, population_size=2)
    abc.new(pyabc.create_sqlite_db_id(), model({'p0': 1}))
    h = abc.run(max_nr_populations=2)

    df = h.get_weighted_distances()
    assert np.isfinite(df['distance']).all()


def test_uniform_acceptor():
    def dist(x, x_0):
        return sum(abs(x[key] - x_0[key]) for key in x_0)

    distance = pyabc.SimpleFunctionDistance(dist)
    acceptor = pyabc.UniformAcceptor()
    eps = pyabc.ListEpsilon([1, 4, 2])

    x = {'s0': 1.5}
    x_0 = {'s0': 0}

    ret = acceptor(distance_function=distance,
                   eps=eps, x=x, x_0=x_0, t=2, par=None)

    assert ret.accept

    # now let's test again, including previous time points

    acceptor = pyabc.UniformAcceptor(use_complete_history=True)

    ret = acceptor(distance_function=distance,
                   eps=eps, x=x, x_0=x_0, t=2, par=None)

    assert not ret.accept
