import numpy as np
from scipy import stats as st
import copy


def weights(n_per_model, transitions, test_transitions, test_X):
    """
    For each model m, sample `n_per_model[m]` points from `transitions[m]`,
    fit `test_transitions[m]` to them, and then compute the weights of all
    test points in `test_X[m]` for `test_transitions[m]`.
    Return those weights.

    Parameters
    ----------

    n_per_model: np.ndarray
        Number of samples per model

    transitions: List[Transition]
        List of transitions used to sample the bootstrapped points

    test_transitions: List[Transition]
        List of transitions used to fit to the bootstrapped points

    test_X: List[np.ndarray]
        Test points at which to evaluate the bootstrapped fitted KDEs

    Returns
    -------

    bootstr_w_at_test_X: List[np.ndarray]
        Weights for each model at the test points
    """
    bootstr_w_at_test_X = []
    for trans, test_trans, n, X in zip(transitions, test_transitions,
                                       n_per_model, test_X):
        bootstr_X = trans.rvs(size=n)
        test_trans.fit(bootstr_X, np.ones(len(bootstr_X)) / len(bootstr_X))
        bootstr_w_at_test_X.append(test_trans.pdf(X))
    return bootstr_w_at_test_X


def calc_cv(nr_particles, model_weights, N_BOOTSTR, test_w,
            transitions, test_X):
    """
    Calculate the Coefficient of Variation.

    Parameters
    ----------

    nr_particles: int
        Number of particles to estimate the CV for

    model_weights: np.ndarray
        array of model weights

    N_BOOTSTR: int
        Nr of bootstrapped KDEs to take to estimate the CV

    test_w: List[np.ndarray]
        test_w[m] are the weights of the test points test_X[m] of model m

    transitions: List[Transition]
        List of transitions

    test_X: List[np.ndarray]
        test_X[m] are the test points with weights test_w[m]

    Returns
    -------

    cv, variations_at_X: float, List[np.ndarray]
        * cv is the mean variation
        * variations_at_X are the variations at the test_X

    """
    # create deep copies of the transitions which will be refitted
    test_transitions = copy.deepcopy(transitions)

    # how many particles to draw for each model
    n_per_model = np.random.multinomial(nr_particles, model_weights)

    # N_BOOTSTR times, train test_transitions on n_per_model points, and
    # calculate the weights associated with test_X, for each model
    bootstr_w_at_test_X = [weights(n_per_model, transitions, test_transitions,
                                   test_X)
                           for _ in range(N_BOOTSTR)]

    # sort by model
    per_model_w = [np.array(arr) for arr in zip(*bootstr_w_at_test_X)]

    # calculate the cv of the bootstrapped weights for each model
    variations_at_X = [st.variation(ws, axis=0) for ws in per_model_w]

    # normalize by number of samples per model
    model_weighted_variations_at_X = [
        var * n / n_per_model.sum() for
        var, n in zip(variations_at_X, n_per_model)
    ]

    # weight cvs by the point weights
    point_weighted_var_at_X = [var * w for var, w in
                               zip(model_weighted_variations_at_X, test_w)]

    # compute an "average coefficient of variation":
    # for each model, sum up the weighted cvs over the test points
    # then, take the sum over all models
    cv = sum(var.sum() for var in point_weighted_var_at_X)

    return float(cv), variations_at_X
