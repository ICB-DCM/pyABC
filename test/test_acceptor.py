from pyabc import (
    Distribution, RV,
    SimpleFunctionAcceptor,
    NoDistance,
    ABCSMC
)
from pyabc.acceptor import AcceptorResult


def test_simple_acceptor():
    def dummy_accept(dist, eps, x, x_0, t, par):
        d = sum(abs(x[key] - x_0[key]) for key in x_0)
        return AcceptorResult(d, d < 1)

    x = {'s0': 1, 's1': 0}
    y = {'s0': 2, 's1': 2}

    acceptor = SimpleFunctionAcceptor(dummy_accept)

    ret = acceptor(distance_function=None, eps=None,
                   x=x, x_0=y, t=0, par=None)

    assert isinstance(ret, AcceptorResult)
    assert ret.distance == 3

    # test integration

    def model(par):
        return {'s0': par['p0'] + 1, 's1': 42}

    prior = Distribution(p0=RV('uniform', -5, 10))
    abc = ABCSMC(model, prior, NoDistance())
    abc.new(
