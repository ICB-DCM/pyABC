from pyabc import (SimpleAcceptor,
                   UniformAcceptor,
                   StochasticAcceptor)


def test_simple_acceptor():
    def dummy_accept(t, dist_t, eps, x, x_0, pars):
        dist = sum(abs(x[key] - x_0[key]) for key in x)
        return dist, dist < 1

    acceptor = SimpleAcceptor(dummy_accept)

    x = {'s0': 1, 's1': 0}
    y = {'s0': 2, 's1': 2}

    dist, accept = acceptor(0, None, None, x, y, None)
