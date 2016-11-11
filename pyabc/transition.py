"""
Perturbation
============

Perturbation strategies.
"""
from abc import ABC, abstractmethod
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

class Transition(ABC):
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
    def pdf(self, x: Union[pd.Series, pd.DataFrame]) -> float:
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
        if len(X.columns) == 0:
            self.no_parameters = True
            return
        self.no_parameters = False

        self.X = X
        self.X_arr = X.as_matrix()
        self.w = w  # TODO assert that w is normalized? use metaclass?
        assert np.isclose(w.sum(), 1)
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

    def cv(self, cv=None):
        if cv is None:
            if not self.no_parameters:
                return variance(self.__class__(), self.X, self.w)
        else:
            return variance_list(self.__class__(), self.X, self.w)[0](cv)

    def rvs(self):
        if self.no_parameters:  # TODO better no parameter handling. metaclass?
            return pd.Series()
        sample = self.X.sample(weights=self.w).iloc[0]
        perturbed = sample + np.random.multivariate_normal(np.zeros(self.cov.shape[0]), self.cov)
        return perturbed

    def pdf(self, x: Union[pd.Series, pd.DataFrame]):
        if self.no_parameters:  # TODO better no parameter handling. metaclass?
            return 1
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


def timeit(f):
    def g(*args):
        import time
        start = time.time()
        res = f(*args)
        end = time.time()
        print(end - start, "s for ", f.__name__, " = ", res)
        return res
    return g


def variance(transition: Transition, X: pd.DataFrame, w: np.ndarray):
    # TODO: This does lots of unnecessary calculation
    # Is for testing for the moment
    return variance_list(transition, X, w)[-1][-1]


def variance_list(transition: Transition, X: pd.DataFrame, w: np.ndarray):
    """
    Calculate mean coefficient of variation of the KDE.
    """
    if len(X) == 1:
        return lambda x: 1, np.array([0])

    transition_cp = copy.copy(transition)
    transition_cp.varprinted = True
    nr_cross_val = 10

    # TODO make this always work. Does not work foll app nr samples
    start = max(len(X)//2, 1)
    stop = len(X)
    step = max(len(X)//30, 1)
    n_samples_list = list(range(start, stop, step)) + [len(X)]
    cvs = np.zeros(len(n_samples_list))
    for ind, n_samples in enumerate(n_samples_list):
        uniform_weights = np.ones(n_samples) / n_samples

        density_values = []
        for k in range(nr_cross_val):
            bootstrapped_points = X.sample(n_samples, replace=True, weights=w)
            transition_cp.fit(bootstrapped_points, uniform_weights)
            density_values.append(transition_cp.pdf(X))

        density_values = np.array(density_values)
        variation = st.variation(density_values, 0)
        mean_variation = (variation * w).sum()
        cvs[ind] = mean_variation

    import matplotlib.pyplot as plt
    import time

    fig, ax = plt.subplots()
    import seaborn as sns
    ax.set_xlabel("Nr bootstrapped samples")
    ax.set_ylabel("Mean KDE PDF coefficient of variation")
    ax.plot(n_samples_list, cvs)
    popt, f, finv = fitpowerlaw(n_samples_list, cvs)
    pltpoints = np.linspace(n_samples_list[0], n_samples_list[-1])
    ax.plot(pltpoints, f(pltpoints))
    ax.set_title(str(popt) + " " + str(X.columns))
    fig.savefig("/home/emmanuel/tmp/" + str(id(transition)) + " " + str(time.time()) + " " + ".png")
    print("popt", popt, "id", id(transition))
    return finv, cvs
