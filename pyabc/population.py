"""
Particles and Populations
=========================

A particle contains the sampled parameters and simulated data.
A population gathers all particles collected in one SMC
iteration.
"""


from typing import List, Callable
import pandas as pd
from pyabc.parameters import Parameter
import logging

logger = logging.getLogger("Population")


class Particle:
    """
    An (accepted or rejected) particle, containing the information that will
    also be stored in the database.
    Stores all summary statistics that
    were generated during the creation of this particle, and a flag
    indicating whether this particle was accepted or not.

    Parameters
    ----------

    m: int
        The model index.

    parameter: Parameter
        The model specific parameter.

    weight: float, 0 <= weight <= 1
        The weight of the particle.

    accepted_sum_stats: List[dict]
        List of accepted summary statistics.
        This list is usually of length 1. This list is longer only if more
        than one sample is taken for a particle.
        This list has length 0 if the particle is rejected.

    accepted_distances: List[float]
        A particle can contain more than one sample.
        If so, the distances of the individual samples
        are stored in this list. In the most common case of a single
        sample, this list has length 1.

    rejected_sum_stats: List[dict]
        List of rejected summary statistics.

    rejected_distances: List[float]
        List of rejected distances.

    accepted: bool
        True if particle was accepted, False if not.


    .. note::
        There are two different ways of weighting particles: First, the weights
        can be calculated as emerges from the importance sampling. Second, the
        weights of particles belonging to one model can be summed to, after
        normalization, find model probabilities. Then, the weights of all
        particles belonging to one model can be summed to one.
        Weighting is transferred to the second way in _normalize_weights() in
        order to also have access to model probabilities. This mode is also
        stored in the database. If one needs access to the first weighting
        scheme later on again, one has to perform backwards transformation,
        multiplying the weights with the model probabilities.
    """

    def __init__(self,
                 m: int,
                 parameter: Parameter,
                 weight: float,
                 accepted_sum_stats: List[dict],
                 accepted_distances: List[float],
                 rejected_sum_stats: List[dict] = None,
                 rejected_distances: List[float] = None,
                 accepted: bool = True):

        self.m = m
        self.parameter = parameter
        self.weight = weight
        self.accepted_sum_stats = accepted_sum_stats
        self.accepted_distances = accepted_distances
        if rejected_sum_stats is None:
            rejected_sum_stats = []
        self.rejected_sum_stats = rejected_sum_stats
        if rejected_distances is None:
            rejected_distances = []
        self.rejected_distances = rejected_distances
        self.accepted = accepted


class Population:
    """
    A population contains a list of particles and offers standardized access
    to them. Upon initialization, the particle weights are normalized and model
    probabilities computed as described in _normalize_weights.
    """

    def __init__(self, particles: List[Particle]):
        self._list = particles.copy()
        self._model_probabilities = None
        self._normalize_weights()

    def __len__(self):
        return len(self._list)

    def get_list(self) -> List[Particle]:
        """
        Returns
        -------

        A copy of the underlying particle list.
        """

        return self._list.copy()

    def _normalize_weights(self):
        """
        Normalize the cumulative weight of the particles belonging to a model
        to 1, and compute the model probabilities. Should only be called once.
        """

        store = self.to_dict()

        model_total_weights = {m: sum(particle.weight for particle in plist)
                               for m, plist in store.items()}
        population_total_weight = sum(model_total_weights.values())
        model_probabilities = {m: w / population_total_weight
                               for m, w in model_total_weights.items()}

        # update model_probabilities attribute
        self._model_probabilities = model_probabilities

        # normalize weights within each model
        for m in store:
            model_total_weight = model_total_weights[m]
            plist = store[m]
            for particle in plist:
                particle.weight /= model_total_weight

    def update_distances(self,
                         distance_to_ground_truth: Callable[[dict], float]):
        """
        Update the distances of all summary statistics of all particles
        according to the passed distance function (which is typically
        different from the distance function with which the original
        distances were computed).

        :param distance_to_ground_truth:
            Distance function to the observed summary statistics.
        """

        for particle in self._list:
            for i in range(len(particle.accepted_distances)):
                particle.accepted_distances[i] = distance_to_ground_truth(
                    particle.accepted_sum_stats[i], particle.parameter)

    def get_model_probabilities(self) -> dict:
        """
        Get probabilities of the individual models.

        Returns
        -------

        model_probabilities: List
            The model probabilities.
        """

        # _model_probabilities are assigned during normalization
        return self._model_probabilities

    def get_weighted_distances(self) -> pd.DataFrame:
        """
        Create DataFrame of (distance, weight)'s. The particle weights are
        multiplied by the model probabilities. If one simulation per particle
        was performed, the weights thus sum to 1. If more than one simulation
        per particle was performed, this does not have to be the case,
        and post-normalizing may be necessary.

        Returns
        -------

        weighted_distances: pd.DataFrame:
            A pd.DataFrame containing in column 'distance' the distances
            and in column 'w' the scaled weights.
        """
        rows = []
        for particle in self._list:
            model_probability = self._model_probabilities[particle.m]
            for distance in particle.accepted_distances:
                rows.append({'distance': distance,
                             'w': particle.weight * model_probability})
        weighted_distances = pd.DataFrame(rows)

        return weighted_distances

    def get_weighted_sum_stats(self) -> tuple:
        """
        Get weights and summary statistics.

        Returns
        -------
        weights, sum_stats: 2-Tuple of lists
        """
        weights = []
        sum_stats = []
        for particle in self._list:
            model_probability = self._model_probabilities[particle.m]
            normalized_weight = particle.weight * model_probability
            for sum_stat in particle.accepted_sum_stats:
                weights.append(normalized_weight)
                sum_stats.append(sum_stat)
        return weights, sum_stats

    def get_accepted_sum_stats(self) -> List[dict]:
        """
        Return a list of all accepted summary statistics.
        """
        sum_stats = []
        for particle in self._list:
            sum_stats.extend(particle.accepted_sum_stats)

        return sum_stats

    def get_for_keys(self, keys):
        """
        Get dataframe of population values. Possible entries of keys:
        weight, distance, sum_stat, parameter.

        Returns
        -------

        Dictionary where the keys are associated to same-ordered lists
        of the corresponding values.
        """
        # check input
        allowed_keys = ['weight', 'distance', 'parameter', 'sum_stat']
        for key in keys:
            if key not in allowed_keys:
                raise ValueError(f"Key {key} not in {allowed_keys}.")

        ret = {key: [] for key in keys}
        for particle in self._list:
            n_accepted = len(particle.accepted_distances)
            if 'weight' in keys:
                model_probability = self._model_probabilities[particle.m]
                weight = particle.weight * model_probability
                ret['weight'].extend([weight] * n_accepted)
            if 'parameter' in keys:
                ret['parameter'].extend([particle.parameter] * n_accepted)
            if 'distance' in keys:
                for distance in particle.accepted_distances:
                    ret['distance'].append(distance)
            if 'sum_stat' in keys:
                for sum_stat in particle.accepted_sum_stats:
                    ret['sum_stat'].append(sum_stat)

        return ret

    def to_dict(self) -> dict:
        """
        Create a dictionary representation, creating a list of particles for
        each model.

        Returns
        -------

        store: dict
            A dictionary with the models as keys and a list of particles for
            each model as values.
        """

        store = {}

        for particle in self._list:
            if particle is not None:
                # append particle for key particle.m, create empty list
                # if key not yet existent
                store.setdefault(particle.m, []).append(particle)
            else:
                logger.warning("Empty particle.")

        return store
