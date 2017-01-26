import scipy as sp


def fast_random_choice(weights):
    """
    this is at least for small arrays much faster
    than numpy.random.choice.
    For the Gillespie overall this brings for 3 reaction a speedup
    of a factor of 2
    """
    cs = 0
    u = sp.random.rand()
    for k in range(weights.size):
        cs += weights[k]
        if u <= cs:
            return k
    raise Exception("Random choice error {}".format(weights))
