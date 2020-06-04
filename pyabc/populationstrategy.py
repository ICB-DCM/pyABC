"""
Population strategies
=====================

Strategies to choose the population size.

The population size can be constant or can change over the course
of the generations.
"""

from abc import ABC, abstractmethod
import json
import logging
import numpy as np
from typing import Dict, List, Union
import warnings

from pyabc.cv.bootstrap import calc_cv
from .transition import Transition
from .transition.predict_population_size import predict_population_size

logger = logging.getLogger("Adaptation")


class PopulationStrategy(ABC):
    """
    Strategy to select the sizes of the populations.

    This is a non-functional abstract base implementation. Do not use this
    class directly. Subclasses must override the `update` method.

    Parameters
    ----------
    nr_calibration_particles:
        Number of calibration particles.
    nr_samples_per_parameter:
        Number of samples to draw for a proposed parameter.
        Default is 1.
    """

    def __init__(self,
                 nr_calibration_particles: int = None,
                 nr_samples_per_parameter: int = 1):
        self.nr_calibration_particles = nr_calibration_particles
        if nr_samples_per_parameter != 1:
            warnings.warn(
                "A nr_samples_per_parameter != 1 is deprecated "
                "since version 0.9.23, the parameter will be removed "
                "in a future release.", DeprecationWarning)
        self.nr_samples_per_parameter = nr_samples_per_parameter

    def update(self, transitions: List[Transition],
               model_weights: np.ndarray, t: int = None):
        """
        Select the population size for the next population.

        Parameters
        ----------
        transitions:
            List of transitions.
        model_weights:
            Array of model weights.
        t:
            Time to adapt for.
        """

    @abstractmethod
    def __call__(self, t: int = None) -> int:
        raise NotImplementedError()

    def get_config(self) -> dict:
        """
        Get the configuration of this object.

        Returns
        -------
        config:
            Configuration of the class as dictionary
        """
        return {"name": self.__class__.__name__,
                "nr_calibration_particles": self.nr_calibration_particles,
                "nr_samples_per_parameter": self.nr_samples_per_parameter}

    def to_json(self) -> str:
        """
        Return the configuration as json string.
        Per default, this converts the dictionary returned
        by get_config to json.

        Returns
        -------
        config:
            Configuration of the class as json string.
        """
        return json.dumps(self.get_config())


class ConstantPopulationSize(PopulationStrategy):
    """
    Constant size of the different populations

    Parameters
    ----------
    nr_particles:
        Number of particles per population.
    nr_calibration_particles:
        Number of calibration particles.
    nr_samples_per_parameter:
        Number of samples to draw for a proposed parameter.
    """

    def __init__(self,
                 nr_particles: int,
                 nr_calibration_particles: int = None,
                 nr_samples_per_parameter: int = 1):
        super().__init__(
            nr_calibration_particles=nr_calibration_particles,
            nr_samples_per_parameter=nr_samples_per_parameter)
        self.nr_particles = nr_particles

    def __call__(self, t: int = None) -> int:
        if t == -1 and self.nr_calibration_particles is not None:
            return self.nr_calibration_particles
        return self.nr_particles

    def get_config(self) -> dict:
        config = super().get_config()
        config["nr_particles"] = self.nr_particles
        return config


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
    start_nr_particles:
        Number of particles in the first populations
    mean_cv:
        The error criterion. Defaults to 0.05.
        A smaller value leads generally to larger populations.
        The error criterion is the mean coefficient of variation of
        the estimated KDE.
    max_population_size:
        Max nr of allowed particles in a population.
        Defaults to infinity.
    min_population_size:
        Min number of particles allowed in a population.
        Defaults to 10
    nr_samples_per_parameter:
        Defaults to 1.
    n_bootstrap:
        Number of bootstrapped populations to use to estimate the CV.
        Defaults to 10.
    nr_calibration_particles:
        Number of calibration particles.


    .. [#klingerhasenaueradaptive] Klinger, Emmanuel, and Jan Hasenauer.
            “A Scheme for Adaptive Selection of Population Sizes in "
            Approximate Bayesian Computation - Sequential Monte Carlo."
            Computational Methods in Systems Biology, 128-44.
            Lecture Notes in Computer Science.
            Springer, Cham, 2017.
            https://doi.org/10.1007/978-3-319-67471-1_8.
    """

    def __init__(self,
                 start_nr_particles,
                 mean_cv: float = 0.05,
                 max_population_size: int = np.inf,
                 min_population_size: int = 10,
                 nr_samples_per_parameter: int = 1,
                 n_bootstrap: int = 10,
                 nr_calibration_particles: int = None):
        super().__init__(
            nr_calibration_particles=nr_calibration_particles,
            nr_samples_per_parameter=nr_samples_per_parameter)
        self.start_nr_particles = start_nr_particles
        self.max_population_size = max_population_size
        self.min_population_size = min_population_size
        self.mean_cv = mean_cv
        self.n_bootstrap = n_bootstrap

        # to hold the current value
        self.nr_particles = start_nr_particles

    def get_config(self) -> dict:
        config = super().get_config()
        config["start_nr_particles"] = self.start_nr_particles
        config["max_population_size"] = self.max_population_size
        config["min_population_size"] = self.min_population_size
        config["mean_cv"] = self.mean_cv
        config["n_bootstrap"] = self.n_bootstrap
        return config

    def update(self, transitions: List[Transition],
               model_weights: np.ndarray, t: int = None):
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

    def __call__(self, t: int = None) -> int:
        if t == -1 and self.nr_calibration_particles is not None:
            return self.nr_calibration_particles
        return self.nr_particles


class ListPopulationSize(PopulationStrategy):
    """
    Return population size values from a predefined list. For every time point
    enquired later (specified by time t), an entry must exist in the list.

    Parameters
    ----------
    values: List[float]
        List of population size values.
        ``values[t]`` is the value for population t.
    nr_calibration_particles:
        Number of calibration particles.
    """

    def __init__(self,
                 values: Union[List[int], Dict[int, int]],
                 nr_calibration_particles: int = None,
                 nr_samples_per_parameter: int = 1):
        super().__init__(
            nr_calibration_particles=nr_calibration_particles,
            nr_samples_per_parameter=nr_samples_per_parameter)
        self.values = values

    def get_config(self) -> dict:
        config = super().get_config()
        config["population_values"] = self.population_values
        return config

    def __call__(self, t: int = None) -> int:
        if t == -1 and self.nr_calibration_particles is not None:
            return self.nr_calibration_particles
        return self.values[t]
