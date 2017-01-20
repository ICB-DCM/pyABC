"""
Population stratgy
==================

Strategies to choose the population size.

At the moment, only constant population size is supported. But this might
change in the future.
"""

from .transition import NotEnoughParticles
import logging
import json
adaptation_logger = logging.getLogger("Adaptation")


class PopulationStrategy:
    """
    Size of the diffrent populations

    This is a non-functional base implementation. Do not use this class
    directly. Subclasses must override the `adapt_population_size` method.

    Parameters
    ----------

    nr_particles: int
       Number of particles per populations

    nr_populations: int
        Maximum number of populations

    nr_samples_per_parameter: int
        Number of samples to draw for a proposed parameter
    """
    def __init__(self, nr_particles: int, nr_populations: int, nr_samples_per_parameter: int=1):
        self.nr_particles = nr_particles
        self.nr_populations = nr_populations
        self.nr_samples_per_parameter = nr_samples_per_parameter

    def adapt_population_size(self, perturbers, model_weights):
        """

        Parameters
        ----------
        perturbers
        model_weights

        Returns
        -------

        """
        raise NotImplementedError

    def get_config(self):
        """
        Returns
        -------
        dict
            Configuration of the class as dictionary
        """
        return {"name": self.__class__.__name__,
                "nr_particles": self.nr_particles,
                "nr_populations": self.nr_populations}

    def to_json(self):
        """
        Returns
        -------

        str
            Configuration of the class as json string.
        """
        return json.dumps(self.get_config())


class ConstantPopulationStrategy(PopulationStrategy):
    """
    Constant size of the diffrent populations

    Parameters
    ----------

    nr_particles: int
       Number of particles per populations

    nr_populations: int
        Maximum number of populations

    nr_samples_per_parameter: int
        Number of samples to draw for a proposed parameter
    """
    def adapt_population_size(self, perturbers, model_weights):
        pass


class AdaptivePopulationStrategy(PopulationStrategy):
    def __init__(self, nr_particles, nr_populations, nr_samples_per_parameter=1,
                 mean_cv=0.05, max_population_size=float("inf")):
        super().__init__(nr_particles, nr_populations, nr_samples_per_parameter)
        self.max_population_size = max_population_size
        self.mean_cv = mean_cv

    def get_config(self):
        return {"name": self.__class__.__name__,
                "max_population_size": self.max_population_size,
                "mean_cv": self.mean_cv}

    def adapt_population_size(self, transitions, model_weights):
        nr_required_samples = []
        for trans in transitions:
            try:
                nr_required_samples.append(trans.required_nr_samples(coefficient_of_variation=self.mean_cv))
            except NotEnoughParticles:
                pass

        if len(nr_required_samples) > 0:
            old_particles = self.nr_particles
            try:
                aggregated_nr_particles = sum(nr_required_samples)
                self.nr_particles = min(int(aggregated_nr_particles), self.max_population_size)
            except TypeError:
                print("DEBUGTYPEERROR", nr_required_samples, self.max_population_size)
                raise
            adaptation_logger.debug("Change nr particles {} -> {}".format(old_particles, self.nr_particles))
