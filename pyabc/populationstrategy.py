from .transition import NotEnoughParticles
import logging

adaptation_logger = logging.getLogger("Adaptation")


class PopulationStrategy:
    """
    max_nr_allowed_sample_attempts_per_particle: int
        The maximum number of sample attempts allowed for each particle.
        If this number is reached, the sampling for a particle is stopped.
        Hence, a population may return with less particles than started.
        This is an approximation to the ABCSMC algorithm which ensures, that
        the algorithm terminates.

    min_nr_particles_per_population: int
        Minimum number of samples which have to be accepted for a population.
        If this number is not reached, the algorithm stops.
        This option, together with the ``max_nr_allowed_sample_attempts_per_particle``
        ensures that the algorithm terminates.

        More precisely, this parameter determines to which extend an approximation to the
        ABCSMC algorithm is allowed.
    """
    def __init__(self, nr_particles, nr_populations, nr_samples_per_parameter=1):
        self.nr_particles = nr_particles
        self.nr_populations = nr_populations
        self.nr_samples_per_parameter = nr_samples_per_parameter

    def min_nr_particles(self):
        return self.nr_particles // 2

    def max_nr_allowed_sample_attempts_per_particle(self):
        return self.nr_particles * 2

    def adapt_population_size(self, perturbers, model_weights):
        raise NotImplementedError


class ConstantPopulationStrategy(PopulationStrategy):
    def adapt_population_size(self, perturbers, model_weights):
        pass


class AdaptivePopulationStrategy(PopulationStrategy):
    def __init__(self, nr_particles, nr_populations, nr_samples_per_parameter=1,
                 mean_cv=0.05, max_population_size=float("inf")):
        super().__init__(nr_particles, nr_populations, nr_samples_per_parameter)
        self.max_population_size = max_population_size
        self.mean_cv = mean_cv

    def adapt_population_size(self, transitions, model_weights):
        nr_required_samples = []
        for trans in transitions:
            try:
                nr_required_samples.append(trans.required_nr_samples(coefficient_of_variation=self.mean_cv))
            except NotEnoughParticles:
                pass

        if len(nr_required_samples) > 0:
            old_particles = self.nr_particles
            self.nr_particles = min(int(sum(nr_required_samples)), self.max_population_size)
            adaptation_logger.debug("Change nr particles {} -> {}".format(old_particles, self.nr_particles))
