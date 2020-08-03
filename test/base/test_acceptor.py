import numpy as np
import pandas as pd
import tempfile

import pyabc
from pyabc.acceptor import AcceptorResult


def test_simple_function_acceptor():
    """Test the simple function acceptor."""

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
    """Test the uniform acceptor."""
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


def test_stochastic_acceptor():
    """Test the stochastic acceptor's features."""
    # store pnorms
    pnorm_file = tempfile.mkstemp(suffix=".json")[1]
    acceptor = pyabc.StochasticAcceptor(
        pdf_norm_method=pyabc.pdf_norm_max_found,
        log_file=pnorm_file)
    eps = pyabc.Temperature(initial_temperature=1)
    distance = pyabc.IndependentNormalKernel(var=np.array([1, 1]))

    def model(par):
        return {'s0': par['p0'] + np.array([0.3, 0.7])}
    x_0 = {'s0': np.array([0.4, -0.6])}

    # just run
    prior = pyabc.Distribution(p0=pyabc.RV('uniform', -1, 2))
    abc = pyabc.ABCSMC(model, prior, distance, eps=eps,
                       acceptor=acceptor, population_size=10)
    abc.new(pyabc.create_sqlite_db_id(), x_0)
    h = abc.run(max_nr_populations=1, minimum_epsilon=1.)

    # check pnorms
    pnorms = pyabc.storage.load_dict_from_json(pnorm_file)
    assert len(pnorms) == h.max_t + 2  # +1 t0, +1 one final update
    assert isinstance(list(pnorms.keys())[0], int)
    assert isinstance(pnorms[0], float)

    # use no initial temperature and adaptive c
    acceptor = pyabc.StochasticAcceptor()
    eps = pyabc.Temperature()
    abc = pyabc.ABCSMC(model, prior, distance, eps=eps,
                       acceptor=acceptor, population_size=20)
    abc.new(pyabc.create_sqlite_db_id(), x_0)
    abc.run(max_nr_populations=3)


def test_pdf_norm_methods_integration():
    """Test integration of pdf normalization methods in ABCSMC."""
    def model(par):
        return {'s0': par['p0'] + np.array([0.3, 0.7])}

    x_0 = {'s0': np.array([0.4, -0.6])}

    for pdf_norm in [pyabc.pdf_norm_max_found,
                     pyabc.pdf_norm_from_kernel,
                     pyabc.ScaledPDFNorm(),
                     ]:
        # just run
        acceptor = pyabc.StochasticAcceptor(pdf_norm_method=pdf_norm)
        eps = pyabc.Temperature()
        distance = pyabc.IndependentNormalKernel(var=np.array([1, 1]))
        prior = pyabc.Distribution(p0=pyabc.RV('uniform', -1, 2))

        abc = pyabc.ABCSMC(model, prior, distance, eps=eps, acceptor=acceptor,
                           population_size=20)
        abc.new(pyabc.create_sqlite_db_id(), x_0)
        abc.run(max_nr_populations=3)


def test_pdf_norm_methods():
    """Test pdf normalization methods standalone."""
    # preparations

    def _get_weighted_distances():
        return pd.DataFrame({
            'distance': [1, 2, 3, 4],
            'w': [2, 1, 1, 0]})

    pdf_norm_args = dict(
        kernel_val=42,
        prev_pdf_norm=3.5,
        get_weighted_distances=_get_weighted_distances,
        prev_temp=10.3,
        acceptance_rate=0.3,
    )

    # run functions
    max_found = max(pdf_norm_args['get_weighted_distances']()['distance'])
    assert pyabc.pdf_norm_max_found(**pdf_norm_args) == max_found
    assert pyabc.pdf_norm_from_kernel(**pdf_norm_args) == 42
    assert pyabc.ScaledPDFNorm()(**pdf_norm_args) == max_found

    # test additional setups
    pdf_norm_args['prev_pdf_norm'] = 4.5
    pdf_norm_args['acceptance_rate'] = 0.05
    assert pyabc.pdf_norm_max_found(**pdf_norm_args) == 4.5
    offsetted_pdf = 4.5 - np.log(10) * 0.5 * pdf_norm_args['prev_temp']
    assert pyabc.ScaledPDFNorm()(**pdf_norm_args) == offsetted_pdf
