from typing import List
from pyabc.parameters import Parameter
from abc import ABCMeta, abstractmethod

class MapWrapper():
    @abstractmethod
    def wrap_map_sample_from_population(self, abc_context,parameter_perturbation_kernels,
                                        nr_samples_per_particle, t, t0, current_eps):
        pass

    @abstractmethod
    def wrap_map_sample_from_prior(self, abc_context) -> List[dict]:
        pass

class MapWrapperDefault(MapWrapper):
    """
        This is the basic map_wrapper implementation required for code compatibility reasons.
        There should be no need do initialize this on the user-level.
    """
    def __init__(self, map_fun=map):
        self.map_fun = map_fun;

    def wrap_map_sample_from_prior(self, abc_context) -> List[dict]:
        """
        Only sample from prior and return results without changing
        the history of the Epsilon. This can be used to get initial samples
        for the distance function or the epsilon to calibrate them.

        .. warning::

            The sample is cached.
        """
        if abc_context._points_sampled_from_prior is None:
            # not saved as attribute b/c Mapper of type "ipython_cluster" is not pickable
            abc_context._points_sampled_from_prior = list(self.map_fun(
                                                            lambda _: self.sample_particle_from_prior(
                                                                abc_context),
                                                            list(range(abc_context.nr_particles))
                                                                       )
                                                          )
        return abc_context._points_sampled_from_prior

    def wrap_map_sample_from_population(self, abc_context, parameter_perturbation_kernels,
                                        nr_samples_per_particle, t, t0, current_eps):
        pop_samples = list(
                            self.map_fun(
                                  lambda _: self.sample_particle_from_pertubation(
                                                abc_context, parameter_perturbation_kernels, nr_samples_per_particle,
                                                t, t0, current_eps),
                                  [None] * abc_context.nr_particles)
                          )
        return pop_samples

    #@staticmethod
    def sample_particle_from_prior(self, abc_context):
        m = abc_context.model_prior_distribution.rvs()
        par = abc_context.parameter_given_model_prior_distribution[m].rvs()
        model_result = abc_context.models[m].summary_statistics(par, abc_context.summary_statistics)
        return model_result.sum_stats

    #@staticmethod
    def sample_particle_from_pertubation(self, abc_context, parameter_perturbation_kernels,
                                           nr_samples_per_particle: int,
                                           t: int,
                                           t0: int,
                                           current_eps: float) -> (int, Parameter, float, List[float], int, List[dict]):
        """
        This is where the actual model evaluation happens.

        Parameters
        ----------
        abc_context
        parameter_perturbation_kernels
        nr_samples_per_particle
        t: int population number
        t0: int initial population
        current_eps

        Returns
        -------

        """
        while True:  # find valid theta_ss and (corresponding b) according to data x_0
            m_ss, theta_ss = abc_context.generate_valid_proposal(parameter_perturbation_kernels, t)
            eval_res = abc_context.evaluate_proposal(m_ss, theta_ss, nr_samples_per_particle, t, t0, current_eps)
            distance_list = eval_res['distance_list']
            simulation_counter = eval_res['simulation_counter']
            summary_statistics_list = eval_res['summary_statistics_list']
            if len(distance_list) > 0:
                break
        weight = abc_context.calc_proposal_weight(distance_list, m_ss, theta_ss, parameter_perturbation_kernels,
                                                  nr_samples_per_particle, t, t0)
        return m_ss, theta_ss, weight, distance_list, simulation_counter, summary_statistics_list


class MapWrapperDistribDemo(map):
    """
        This implementation of the map wrapper demonstrates the idea behind the map_wrapper object.
    """
    def __init__(self, map_fun=map, test_string=""):
        self.map_fun = map_fun;
        self.test_string = test_string

    def set_map_fun(self, map_fun):
        self.map_fun = map_fun;

    def wrap_map(self, lambdafun, args):
        # sanity checks
        print(self.test_string)
        print("If I was a real distributed mapper, I'd now start an army of workers...")
        print("They'd do my bidding... I'd leave orders for them in a DB, they'd gather them and " +
              "return their results...")
        print("Once I'd have decided that my results are sufficient, I'd return them as a list")
        print("However, I'm a poor imitation and still only do a simple map() call and return the result")
        print("Now I'm depressed. Are you happy?")
        result = self.map_fun(lambdafun, args)
        return result


