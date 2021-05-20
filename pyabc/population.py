"""
Particles and Populations
=========================

A particle contains the sampled parameters and simulated data.
A population gathers all particles collected in one SMC
iteration.
"""


from typing import Callable, Dict, List, Tuple
import numpy as np
import pandas as pd
import logging
from .parameters import Parameter

logger = logging.getLogger("ABC.Population")


class Particle:
    """
    An (accepted or rejected) particle, containing the information that will
    also be stored in the database.
    Stores all summary statistics that
    were generated during the creation of this particle, and a flag
    indicating whether this particle was accepted or not.

    Parameters
    ----------
    m:
        The model index.
    parameter:
        The model specific parameter.
    weight:
        The weight of the particle. 0 <= weight <= 1.
    sum_stat:
        Model simulation.
    distance:
        Distance of simulated and measured data.
    accepted:
        True if particle was accepted, False if not.
    proposal_id:
        An identifier for the proposal the particle was generated from.
        This allows grouping particles accordingly.
    preliminary:
        Whether this particle is only preliminarily accepted. Must be False
        eventually for all particles.

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
                 sum_stat: dict,
                 distance: float,
                 accepted: bool = True,
                 proposal_id: int = 0,
                 preliminary: bool = False):

        self.m = m
        self.parameter = parameter
        self.weight = weight
        self.sum_stat = sum_stat
        self.distance = distance
        self.accepted = accepted
        self.proposal_id = proposal_id
        self.preliminary = preliminary


class Population:
    """
    A population contains a list of particles and offers standardized access
    to them. Upon initialization, the particle weights are normalized and model
    probabilities computed as described in _normalize_weights.

    Parameters
    ----------
    particles:
        Particles that constitute the accepted population.
    """

    def __init__(self, particles: List[Particle]):
        self.particles = particles
        self._model_probabilities = None

        # checks
        if any(not p.accepted for p in particles):
            raise AssertionError(
                "A population should only consist of accepted particles")
        if not np.isclose(sum(p.weight for p in particles), 1):
            raise AssertionError(
                "The population total weight is not normalized.")

        self.calculate_model_probabilities()

    def __len__(self):
        return len(self.particles)

    def calculate_model_probabilities(self):
        """Compute the model probabilities and ensure normalization.

        Computes model weights as relative sums of particles belonging to
        a model.
        Also ensures that the total weight is 1, raising if this was not the
        case before already since usually normalization should happen already
        for the entire sample including rejected particles.
        """
        store = self.get_particles_by_model()

        # calculate weight per model
        model_total_weights = {m: sum(particle.weight for particle in plist)
                               for m, plist in store.items()}

        # calculate total weight
        population_total_weight = sum(model_total_weights.values())  # 1

        # model probabilities are weights per model divided by total weight
        model_probabilities = {m: w / population_total_weight
                               for m, w in model_total_weights.items()}

        # cache model probabilities
        self._model_probabilities = model_probabilities

    def update_distances(
            self,
            distance_to_ground_truth: Callable[[dict, Parameter], float],
    ) -> None:
        """
        Update the distances of all summary statistics of all particles
        according to the passed distance function (which is typically
        different from the distance function with which the original
        distances were computed).

        :param distance_to_ground_truth:
            Distance function to the observed summary statistics.
        """

        for particle in self.particles:
            particle.distance = distance_to_ground_truth(
                particle.sum_stat, particle.parameter)

    def get_model_probabilities(self) -> pd.DataFrame:
        """Get probabilities of the individual models.

        Returns
        -------
        model_probabilities: List
            The model probabilities.
        """
        # _model_probabilities are cached at the beginning
        vars = [(key, val) for key, val in self._model_probabilities.items()]
        ms = [var[0] for var in vars]
        ps = [var[1] for var in vars]
        return pd.DataFrame({'m': ms, 'p': ps}).set_index('m')

    def get_alive_models(self) -> List:
        return self._model_probabilities.keys()

    def nr_of_models_alive(self) -> int:
        return len(self.get_alive_models())

    def get_distribution(self, m: int) -> Tuple[pd.DataFrame, np.ndarray]:
        particles = self.get_particles_by_model()[m]
        parameters = pd.DataFrame([p.parameter for p in particles])
        weights = np.array([p.weight for p in particles])
        weights /= weights.sum()
        return parameters, weights

    def get_weighted_distances(self) -> pd.DataFrame:
        """
        Create DataFrame of (distance, weight)'s. The particle weights are
        multiplied by the model probabilities. The weights thus sum to 1.

        Returns
        -------
        weighted_distances: pd.DataFrame:
            A pd.DataFrame containing in column 'distance' the distances
            and in column 'w' the scaled weights.
        """
        rows = []
        for particle in self.particles:
            rows.append({'distance': particle.distance, 'w': particle.weight})
        weighted_distances = pd.DataFrame(rows)
        return weighted_distances

    def get_weighted_sum_stats(self) -> tuple:
        """Get weights and summary statistics.

        Returns
        -------
        weights, sum_stats: 2-Tuple of lists
        """
        weights = []
        sum_stats = []
        for particle in self.particles:
            weights.append(particle.weight)
            sum_stats.append(particle.sum_stat)
        return weights, sum_stats

    def get_accepted_sum_stats(self) -> List[dict]:
        """Return a list of all accepted summary statistics."""
        return [particle.sum_stat for particle in self.particles]

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
        for particle in self.particles:
            if 'weight' in keys:
                ret['weight'].append(particle.weight)
            if 'parameter' in keys:
                ret['parameter'].append(particle.parameter)
            if 'distance' in keys:
                ret['distance'].append(particle.distance)
            if 'sum_stat' in keys:
                ret['sum_stat'].append(particle.sum_stat)

        return ret

    def get_particles_by_model(self) -> Dict[int, List[Particle]]:
        """Get particles by model.

        Returns
        -------
        particles_by_model: dict
            A dictionary with the models as keys and a list of particles for
            each model as values.
        """
        particlees_by_model = {}

        for particle in self.particles:
            if particle is not None:
                # append particle for key particle.m, create empty list
                # if key not yet existent
                particlees_by_model.setdefault(particle.m, []).append(particle)
            else:
                logger.warning("Empty particle.")

        return particlees_by_model


class Sample:
    """Contains all particles generated during the sampling process.

    Contains all accepted particles and can also contain rejected particles
    if requested.

    Parameters
    ----------
    record_rejected:
        Whether to record rejected particles as well, along with accepted
        ones.
    is_look_ahead:
        Whether this sample consists of particles generated in look-ahead mode.
    max_nr_rejected:
        Maximum number or rejected particles to store.
    ok:
        Whether the sampling process succeeded (usually in generating the
        requested number of particles).
    """

    def __init__(self,
                 record_rejected: bool = False,
                 max_nr_rejected: int = np.inf,
                 is_look_ahead: bool = False,
                 ok: bool = True):
        self.accepted_particles: List[Particle] = []
        self.rejected_particles: List[Particle] = []
        self.record_rejected: bool = record_rejected
        self.max_nr_rejected: int = max_nr_rejected
        self.is_look_ahead: bool = is_look_ahead
        self.ok: bool = ok

    @staticmethod
    def from_population(population: Population) -> 'Sample':
        """Utility function to create a dummy sample from a population.

        Obviously, the sample will then only consist of accepted particles.

        Parameters
        ----------
        population: Population to create a sample from.

        Returns
        -------
        sample: The generated sample containing all population particles.
        """
        sample = Sample()
        for particle in population.particles:
            if not particle.accepted:
                raise AssertionError(
                    "A population should only consist of accepted particles")
            sample.append(particle)
        return sample

    @property
    def all_particles(self):
        return self.accepted_particles + self.rejected_particles

    @property
    def all_sum_stats(self):
        """Get all summary statistics.

        Returns
        -------
        all_sum_stats: List
            Concatenation of all the all_sum_stats lists of all
            particles added and accepted to this sample via append().
        """
        return [particle.sum_stat for particle in self.all_particles]

    def append(self, particle: Particle):
        """Add new particle to the sample.

        Parameters
        ----------
        particle: Particle
            Sampled particle containing all information needed later.
        """
        # add to population if accepted, else maybe add to rejected
        if particle.accepted:
            self.accepted_particles.append(particle)
        elif self.record_rejected and \
                len(self.rejected_particles) < self.max_nr_rejected:
            self.rejected_particles.append(particle)

    def __add__(self, other: "Sample"):
        sample = Sample(
            record_rejected=self.record_rejected,
            max_nr_rejected=self.max_nr_rejected)

        sample.accepted_particles = \
            self.accepted_particles + other.accepted_particles
        sample.rejected_particles = \
            self.rejected_particles + other.rejected_particles
        # restrict to max nr rejected
        if len(sample.rejected_particles) > sample.max_nr_rejected:
            sample.rejected_particles = \
                sample.rejected_particles[:sample.max_nr_rejected]

        # the other attributes may keep their defaults
        return sample

    @property
    def n_accepted(self) -> int:
        """
        Returns
        -------
        n_accepted: int
            Number of accepted particles.
        """
        return len(self.accepted_particles)

    def get_accepted_population(self) -> Population:
        """
        Returns
        -------
        population: Population
            A population of only the accepted particles.
        """
        return Population(self.accepted_particles.copy())

    def normalize_weights(self):
        """Normalize weights to sum(accepted weights) = 1.

        This is done at the end of a sampling run. Normalizing rejected
        weights by the same factor ensures that weights stay comparable.
        """
        total_weight_accepted = sum(p.weight for p in self.accepted_particles)
        if np.isclose(total_weight_accepted, 0):
            raise AssertionError("The total population weight is zero")
        for p in self.all_particles:
            p.weight /= total_weight_accepted


class SampleFactory:
    """
    The SampleFactory class serves as a factory class to create empty samples
    based on the parameters stored in the SampleFactory object.

    This is the class that components (like the distance function and
    epsilon) should refer to when they want to influence the sampling process.

    Parameters
    ----------
    record_rejected: bool
        Corresponds to Sample.record_rejected.
    """

    def __init__(
            self,
            record_rejected: bool = False,
            max_nr_rejected: int = np.inf):
        self._record_rejected = record_rejected
        self._max_nr_rejected = max_nr_rejected

    def record_rejected(self, record: bool = True):
        """Switch whether to record rejected particles."""
        logger.info(f"Recording also rejected particles: {record}")
        self._record_rejected = record

    def __call__(self, is_look_ahead: bool = False):
        """Create a new empty sample."""
        return Sample(
            record_rejected=self._record_rejected,
            max_nr_rejected=self._max_nr_rejected,
            is_look_ahead=is_look_ahead)
