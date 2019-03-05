"""
Population strategy
===================

Strategies to choose the population size.

The population size can be constant or can change over the course
of the generations.
"""

import json
import logging

import numpy as np
from typing import List

from pyabc.cv.bootstrap import calc_cv
from .transition import Transition
from .transition.predict_population_size import predict_population_size

logger = logging.getLogger("Adaptation")


class PopulationStrategy:
    """
    Strategy to select the sizes of the populations.

    This is a non-functional abstract base implementation. Do not use this
    class directly. Subclasses must override the `adapt_population_size`
    method.

    Parameters
    ----------

    nr_particles: int
       Number of particles per populations

    nr_samples_per_parameter: int, optional
        Number of samples to draw for a proposed parameter.
        Default is 1.
    """

    def __init__(self, nr_particles: int, *,
                 nr_samples_per_parameter: int = 1):
        self.nr_particles = nr_particles
        self.nr_samples_per_parameter = nr_samples_per_parameter

    def adapt_population_size(self, transitions: List[Transition],
                              model_weights: np.ndarray):
        """
        Select the population size for the next population.

        Parameters
        ----------
        transitions: List of Transitions
        model_weights: array of model weights

        Returns
        -------
        n: int
            The new population size
        """
        raise NotImplementedError

    def get_config(self):
        """
        Get the configuration of this object.

        Returns
        -------
        dict
            Configuration of the class as dictionary
        """
        return {"name": self.__class__.__name__,
                "nr_particles": self.nr_particles}

    def to_json(self):
        """
        Return the configuration as json string.
        Per default, this converts the dictionary returned
        by get_config to json.

        Returns
        -------

        str
            Configuration of the class as json string.
        """
        return json.dumps(self.get_config())


class ConstantPopulationSize(PopulationStrategy):
    """
    Constant size of the different populations

    Parameters
    ----------

    nr_particles: int
       Number of particles per populations

    nr_samples_per_parameter: int
        Number of samples to draw for a proposed parameter
    """

    def adapt_population_size(self, transitions, model_weights):
        pass


class AdaptivePopulationSize(PopulationStrategy):
    """
    Adapt the population size according to the mean coefficient of variation
    error criterion, as detailed in [#klingerhasenaueradaptive]_ .
    This strategy tries to respond to the shape of the
    current posterior approximation by selecting the population size such
    that the variation of the density estimates matches the target
    variation given via the mean_cv argument.

    Parameters
    ----------

    start_nr_particles: int
        Number of particles in the first populations

    mean_cv: float, optional
        The error criterion. Defaults to 0.05.
        A smaller value leads generally to larger populations.
        The error criterion is the mean coefficient of variation of
        the estimated KDE.

    max_population_size: int, optional
        Max nr of allowed particles in a population.
        Defaults to infinity.

    min_population_size: int, optional
        Min number of particles allowed in a population.
        Defaults to 10

    nr_samples_per_parameter: int, optional
        Defaults to 1.

    n_bootstrap: int, optional
        Number of bootstrapped populations to use to estimate the CV.
        Defaults to 10.


    .. [#klingerhasenaueradaptive] Klinger, Emmanuel, and Jan Hasenauer.
            “A Scheme for Adaptive Selection of Population Sizes in "
            Approximate Bayesian Computation - Sequential Monte Carlo."
            Computational Methods in Systems Biology, 128-44.
            Lecture Notes in Computer Science.
            Springer, Cham, 2017.
            https://doi.org/10.1007/978-3-319-67471-1_8.
    """

    def __init__(self, start_nr_particles, mean_cv=0.05,
                 *,
                 max_population_size=float("inf"),
                 min_population_size=10,
                 nr_samples_per_parameter=1,
                 n_bootstrap=10):
        super().__init__(start_nr_particles,
                         nr_samples_per_parameter=nr_samples_per_parameter)
        self.max_population_size = max_population_size
        self.min_population_size = min_population_size
        self.mean_cv = mean_cv
        self.n_bootstrap = n_bootstrap

    def get_config(self):
        return {"name": self.__class__.__name__,
                "max_population_size": self.max_population_size,
                "mean_cv": self.mean_cv}

    def adapt_population_size(self, transitions: List[Transition],
                              model_weights: np.ndarray):
        test_X = [trans.X for trans in transitions]
        test_w = [trans.w for trans in transitions]

        reference_nr_part = self.nr_particles
        target_cv = self.mean_cv
        cv_estimate = predict_population_size(
            reference_nr_part, target_cv,
            lambda nr_particles: calc_cv(nr_particles, model_weights,
                                         self.n_bootstrap, test_w, transitions,
                                         test_X)[0])

        if not np.isnan(cv_estimate.n_estimated):
            self.nr_particles = max(min(int(cv_estimate.n_estimated),
                                        self.max_population_size),
                                    self.min_population_size)

        logger.info("Change nr particles {} -> {}"
                    .format(reference_nr_part, self.nr_particles))
