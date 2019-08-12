from pyabc import (SimpleFunctionAcceptor)


def test_simple_acceptor():
    def dummy_accept(t, dist_t, eps, x, x_0, pars):
        dist = sum(abs(x[key] - x_0[key]) for key in x)
        return dist, dist < 1

    acceptor = SimpleFunctionAcceptor(dummy_accept)

    x = {'s0': 1, 's1': 0}
    y = {'s0': 2, 's1': 2}

    acc_result = acceptor(0, None, None, x, y, None)
