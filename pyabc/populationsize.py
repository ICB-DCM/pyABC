from .transition import CVNotPossibleException
import logging

adaptation_logger = logging.getLogger("Adaptation")


class Size:
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


class ConstantSize(Size):
    def adapt_population_size(self, perturbers, model_weights):
        pass


class AdaptiveSize(Size):  # TODO rename ConstantSize. This inheritance is weird
    def __init__(self, nr_particles, nr_populations, nr_samples_per_parameter=1, mean_cv=0.05, max_population_size=None):
        super().__init__(nr_particles, nr_populations, nr_samples_per_parameter)
        self.max_population_size = max_population_size if max_population_size is not None else nr_particles * 2
        self.mean_cv = mean_cv

    def adapt_population_size(self, perturbers, model_weights):
        nr_required_samples = []
        for pert in perturbers:
            try:
                nr_required_samples.append(pert.cv(cv=self.mean_cv))
            except CVNotPossibleException:
                pass

        if len(nr_required_samples) > 0:
            old_particles = self.nr_particles
            self.nr_particles = min(int(sum(nr_required_samples)), self.max_population_size)
            adaptation_logger.debug("Change nr particles {} -> {}".format(old_particles, self.nr_particles))
