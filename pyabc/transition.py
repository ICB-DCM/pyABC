"""
Perturbation
============

Perturbation strategies.
"""
from abc import ABCMeta, abstractmethod
import pandas as pd
import numpy as np
import scipy.stats as st
import copy
from typing import Union
from scipy.optimize import curve_fit

# TODO decide what to do if no parameters there, i.e. if len(X.columns) == 0
# Possible options include
# 1. Have a specific perturber for no particles
# 2. Explicitly hande that case in each concrete perturber implementation
# 3. Make a metaclass which takes care of the no parameters case. This metaclass could also check if w is sane


def fit_wrap(self, X, w):
    if len(X.columns) == 0:
        self.no_parameters = True
        return
    self.no_parameters = False
    if w.size > 0:
        assert np.isclose(w.sum(), 1)
    self.fit_old(X, w)


def pdf_wrap(self, x):
    if self.no_parameters:
        return 1
    return self.pdf_old(x)


def rvs_wrap(self):
    if self.no_parameters:  # TODO better no parameter handling. metaclass?
        return pd.Series()
    return self.rvs_old()


class TransitionMeta(ABCMeta):
    """
    This metaclass handles the special case of no parameters.
    Transition classes do not have to check for it anymore
    """
    def __init__(cls, name, bases, attrs):
        ABCMeta.__init__(cls, name, bases, attrs)
        cls.fit_old = cls.fit
        cls.fit = fit_wrap

        cls.pdf_old = cls.pdf
        cls.pdf = pdf_wrap

        cls.rvs_old = cls.rvs
        cls.rvs = rvs_wrap


class Transition(metaclass=TransitionMeta):
    @abstractmethod
    def fit(self, X: pd.DataFrame, w: np.ndarray):
        """
        Fit the perturber to the sampled data.
        Concrete implementations might do something like fitting a KDE.

        Parameters
        ----------
        X: pd.DataFrame
            The parameters.
        w: array
            The corresponding weights
        """

    @abstractmethod
    def rvs(self) -> pd.Series:
        """
        Random variable sample (rvs).

        Sample from the fitted distribution.

        Returns
        -------
        sample: pd.Series
            A sample from the fitted model
        """

    @abstractmethod
    def pdf(self, x: Union[pd.Series, pd.DataFrame]) -> Union[float, np.ndarray]:
        """
        Evaluate the probability density function (PDF) at x.

        Parameters
        ----------
        x: pd.Series
            Parameter

        Returns
        -------

        density: float
            Probability density at .
        """
    @abstractmethod
    def mean_coefficient_of_variation(self) -> float:
        pass

    @abstractmethod
    def required_nr_samples(self, coefficient_of_variation: float) -> int:
        pass


class NotEnoughParticles(Exception):
    pass


class MultivariateNormalTransition(Transition):
    """
    Pretty stupid but should in principle be functional
    """

    def fit(self, X: pd.DataFrame, w: np.ndarray):
        """
        Fit the transition kernel

        Parameters
        ----------
        X: DataFrame
            The parameters.
        w: array
            Array of weights

        It holds that len(X) == len(w).
        """
        if len(X) == 0:
            raise NotEnoughParticles("Fitting not possible.")

        self.X = X
        self.X_arr = X.as_matrix()
        self.w = w
        if len(X) > 1:
            cov = np.cov(self.X_arr, aweights=w, rowvar=False)
        else:
            cov_diag = self.X_arr[0]
            cov = np.diag(cov_diag)
        if len(cov.shape) == 0:
            cov = cov.reshape((1,1))
        self.cov = cov * scott_rule_of_thumb(len(X) / (1 + w.var()), cov.shape[0])
        import scipy.stats as st
        if not self.no_parameters:
            self.normal = st.multivariate_normal(cov=self.cov, allow_singular=True)

    def required_nr_samples(self, coefficient_of_variation):
        if not hasattr(self, "X") or not hasattr(self, "w"):
            raise NotEnoughParticles
        return variance_list(self.__class__(), self.X, self.w)[0](coefficient_of_variation)

    def mean_coefficient_of_variation(self):
        if not self.no_parameters:
            if not hasattr(self, "X") or not hasattr(self, "w"):
                raise NotEnoughParticles
            return variance(self.__class__(), self.X, self.w)

    def rvs(self):
        sample = self.X.sample(weights=self.w).iloc[0]
        perturbed = sample + np.random.multivariate_normal(np.zeros(self.cov.shape[0]), self.cov)
        return perturbed

    def pdf(self, x: Union[pd.Series, pd.DataFrame]):
        x = x[self.X.columns]
        x = np.array(x)
        if len(x.shape) == 1:
            x = x[None,:]
        dens = np.array([(self.normal.pdf(xs - self.X_arr) * self.w).sum() for xs in x])
        return dens if dens.size != 1 else float(dens)


def scott_rule_of_thumb(n_samples, dimension):
    return n_samples ** (-1. / (dimension + 4))


def silverman_rule_of_thumb(n_samples, dimension):
    return (n_samples * (dimension + 2) / 4.) ** (-1. / (dimension + 4))


def fitpowerlaw(x, y):
    x = np.array(x)
    y = np.array(y)

    def f(x, a, b):
        return a * x ** (-b)

    popt, cov = curve_fit(f, x, y, p0=[.5, 1 / 5])

    return popt, lambda x: f(x, *popt), lambda y: finverse(y, *popt)


def finverse(y, a, b):
    return (a / y) ** (1 / b)


def variance(transition: Transition, X: pd.DataFrame, w: np.ndarray):
    # TODO: This does lots of unnecessary calculation
    # Is for testing for the moment
    return variance_list(transition, X, w)[-1][-1]


def variance_list(transition: Transition, X: pd.DataFrame, w: np.ndarray):
    """
    Calculate mean coefficient of variation of the KDE.
    """
    NR_CROSS_VAL = 10
    START_FACTOR = 3
    NR_STEPS = 30

    if len(X) == 1:
        return lambda x: 1, np.array([0])

    transition_cp = copy.copy(transition)
    transition_cp.varprinted = True

    start = max(len(X)//START_FACTOR, 1)
    stop = len(X)
    step = max(len(X)//NR_STEPS, 1)
    n_samples_list = list(range(start, stop, step)) + [len(X)]
    cvs = np.zeros(len(n_samples_list))
    for ind, n_samples in enumerate(n_samples_list):
        uniform_weights = np.ones(n_samples) / n_samples

        density_values = []
        for k in range(NR_CROSS_VAL):
            bootstrapped_points = X.sample(n_samples, replace=True, weights=w)
            transition_cp.fit(bootstrapped_points, uniform_weights)
            density_values.append(transition_cp.pdf(X))

        density_values = np.array(density_values)
        variation = st.variation(density_values, 0)
        mean_variation = (variation * w).sum()
        cvs[ind] = mean_variation

    popt, f, finv = fitpowerlaw(n_samples_list, cvs)
    return finv, cvs
