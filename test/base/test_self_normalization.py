import pyabc
import numpy as np


def create_population_weights(subpopulation_sizes: np.array):
    """Create arbitrary weights, with different average for each subpopulation
    """
    weights = np.zeros(subpopulation_sizes.sum())
    for i in range(len(subpopulation_sizes)):
        subpop_weight_expectation = 2/subpopulation_sizes.sum() * np.random.uniform()
        if i != 0:
            already_assigned = subpopulation_sizes[:i].sum()
        else:
            already_assigned = 0
        for j in range(subpopulation_sizes[i]):
            weights[already_assigned + j] = max(0.00001, np.random.normal(subpop_weight_expectation,
                                                                          subpop_weight_expectation))

    return weights


def compare_opt_alphas(weights, subpopulation_sizes):
    """Compute optimum alpha once using analytical and once scipy.minimizer approach
    """
    alpha_analytical = pyabc.sampler.redis_eps.sampler.analytical_solution(weights,
                                                                           subpopulation_sizes)

    prop_ids = [-i for i in range(len(subpopulation_sizes))]
    alpha_minimized =\
        pyabc.sampler.redis_eps.sampler.solution_by_minimizer(pyabc.sampler.redis_eps.sampler.weighted_ess,
                                                              prop_ids,
                                                              weights,
                                                              subpopulation_sizes)

    return alpha_analytical, alpha_minimized


def print_results(subpopulation_sizes: np.array):
    """Create and print test results
    """
    weights = create_population_weights(subpopulation_sizes)
    alpha_analytical, alpha_minimized = compare_opt_alphas(weights, subpopulation_sizes)
    ess_analytical = pyabc.sampler.redis_eps.sampler.weighted_ess(np.array([alpha_analytical]),
                                                                  weights,
                                                                  subpopulation_sizes)

    ess_minimized = pyabc.sampler.redis_eps.sampler.weighted_ess(np.array([alpha_minimized]),
                                                                 weights,
                                                                 subpopulation_sizes)
    print("Analytical solution: alpha=", alpha_analytical, "; ESS=", ess_analytical)
    print("Minimizer solution: alpha=", alpha_minimized, "; ESS=", ess_minimized)


print_results(np.array([200, 800]))


"""
TODO test full normalize_with_opt_ess function
How to generate sample object?
"""