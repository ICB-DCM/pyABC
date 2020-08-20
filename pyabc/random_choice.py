import numpy as np


def fast_random_choice(weights):
    """
    This is at least for small arrays much faster than numpy.random.choice.
    For the Gillespie, overall this brings for 3 reactions a speedup factor
    of 2.
    """
    # rough heuristic when it makes sense to use numpy's implementation
    if len(weights) >= 15:
        return np.random.choice(len(weights), p=weights)

    # cumulative weights
    cs = 0
    # draw a uniform random number
    u = np.random.rand()
    # return weight index at random variable
    for k in range(len(weights)):
        cs += weights[k]
        if u <= cs:
            return k

    # error when u > sum(weights) < 1 (not checked pro-actively)
    raise ValueError("Random choice error {}".format(weights))
