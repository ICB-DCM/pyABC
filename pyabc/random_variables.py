"""
Random variables
================
"""


import logging
from abc import ABC, abstractmethod
from functools import reduce
from typing import Union
from .parameters import Parameter, ParameterStructure

rv_logger = logging.getLogger("RV")


class RVBase(ABC):
    """
    Random variable abstract base class.

    .. note::

        Why introduce another random variable class and not just use
        the one's provided in
        ``scipy.stats``?

        This funny construction is done because ``scipy.stats``
        distributions are not pickleable.
        This class is really a very thin wrapper around ``scipy.stats``
        distributions to make them pickleable.
        It is important to be able to pickle them to execute the ACBSMC
        algorithm in a distributed cluster
        environment
    """

    @abstractmethod
    def copy(self) -> "RVBase":
        """
        Copy the random variable.

        Returns
        -------
        copied_rv: RVBase
            A copy of the random variable.
        """

    @abstractmethod
    def rvs(self, *args, **kwargs) -> float:
        """
        Sample from the RV.

        Returns
        -------

        sample: float
            A sample from the random variable.
        """

    @abstractmethod
    def pmf(self, x, *args, **kwargs) -> float:
        """
        Probability mass function

        Parameters
        ----------

        x: int
            Probability mass at ``x``.

        Returns
        -------

        mass: float
            The mass at ``x``.
        """

    @abstractmethod
    def pdf(self, x: float, *args, **kwargs) -> float:
        """
        Probability density function

        Parameters
        ----------
        x: float
            Probability density at x.

        Returns
        -------

        density: float
            Probability density at x.
        """

    @abstractmethod
    def cdf(self, x: float, *args, **kwargs) -> float:
        """
        Cumulative distribution function.

        Parameters
        ----------
        x: float
            Cumulative distribution function at x.

        Returns
        -------

        density: float
            Cumulative distribution function at x.
        """


class RV(RVBase):
    """
    Concrete random variable.

    Parameters
    ----------

    name: str
        Name of the distribution as in ``scipy.stats``

    args:
        Arguments as in ``scipy.stats`` matching the distribution
        with name "name".

    kwargs:
        Keyword arguments as in ``scipy.stats``
        matching the distribution with name "name".
    """

    @classmethod
    def from_dictionary(cls, dictionary: dict) -> "RV":
        """
        Construct random variable from dictionary.

        Parameters
        ----------

        dictionary: dict
            A dictionary with the keys

               * "name" (mandatory)
               * "args" (optional)
               * "kwargs" (optional)

            as in scipy.stats.



        .. note::

            Either the "args" or the "kwargs" key has to be present.
        """

        return cls(dictionary['type'], *dictionary.get('args', []),
                   **dictionary.get('kwargs', {}))

    def __init__(self, name: str, *args, **kwargs):
        self.name = name
        self.args = args
        self.kwargs = kwargs
        self.distribution = None
        "the scipy.stats. ... distribution object"
        self.__setstate__(self.__getstate__())

    def __getattr__(self, item):
        return getattr(self.distribution, item)

    def __getstate__(self):
        return self.name, self.args, self.kwargs

    def __setstate__(self, state):
        self.name = state[0]
        self.args = state[1]
        self.kwargs = state[2]
        import scipy.stats as st
        distribution = getattr(st, self.name)
        self.distribution = distribution(*self.args, **self.kwargs)

    def copy(self):
        return self.__class__(self.name, *self.args, **self.kwargs)

    def rvs(self, *args, **kwargs):
        return self.distribution.rvs(*args, **kwargs)

    def pmf(self, x, *args, **kwargs):
        return self.distribution.pmf(x, *args, **kwargs)

    def pdf(self, x, *args, **kwargs):
        return self.distribution.pdf(x, *args, **kwargs)

    def cdf(self, x, *args, **kwargs):
        return self.distribution.cdf(x, *args, **kwargs)

    def __repr__(self):
        return ("<RV(name={name}, args={args} kwargs={kwargs})>"
                .format(name=self.name, args=self.args, kwargs=self.kwargs))


class RVDecorator(RVBase):
    """
    Random variable decorater base class.

    Implement a decorator pattern.

    Further decorators should derive from this class.

    It stores the decorated random variable in ``self.component``

    Overwrite the method ``decorator_repr`` the represent the decorator type.
    The decorated variable will then be automatically included in
    the call to ``__repr__``.

    Parameters
    ----------

    component: RVBase
        The random variable to be decorated.
    """

    def __init__(self, component: RVBase):
        self.component = component  #: The decorated random variable

    def rvs(self, *args, **kwargs):
        return self.component.rvs(*args, **kwargs)

    def pmf(self, x, *args, **kwargs):
        return self.component.pmf(x, *args, **kwargs)

    def pdf(self, x, *args, **kwargs):
        return self.component.pdf(x, *args, **kwargs)

    def cdf(self, x, *args, **kwargs):
        return self.component.cdf(x, *args, **kwargs)

    def copy(self):
        return self.__class__(self.component.copy())

    def decorator_repr(self) -> str:  # pylint: disable=R0201
        """
        Represent the decorator itself.

        Template method.

        The ``__repr__`` method used ``decorator_repr`` and the
        ``__repr__`` of the
        decorated RV to build a combined representation.

        Returns
        -------

        decorator_repr: str
            A string representing the decorator only.
        """

        return "Decorator"

    def __repr__(self):
        return ("[{decorator_repr}]"
                .format(decorator_repr=self.decorator_repr())
                + self.component.__repr__())


class LowerBoundDecorator(RVDecorator):
    """
    Impose a strict lower bound on a random variable.
    Condition RV X to X > lower bound.
    In particular P(X = lower_bound) = 0.

    .. note::

        Sampling is done via rejection. Up to 10000 samples are taken
        from the decorated RV.
        The first sample within the permitted range is then taken.
        Otherwise None is returned.

    Parameters
    ----------

    component: RV
        The decorated random variable.

    lower_bound: float
        The lower bound.
    """

    MAX_TRIES = 10000

    def __init__(self, component: RV, lower_bound: float):
        if component.cdf(lower_bound) == 1:
            raise Exception(
                "LowerBoundDecorator: Conditioning on a set of measure zero.")
        self.lower_bound = lower_bound
        super(LowerBoundDecorator, self).__init__(component)

    def copy(self):
        return self.__class__(self.component.copy(), self.lower_bound)

    def decorator_repr(self):
        return "Lower: X > {lower:2f}".format(lower=self.lower_bound)

    def rvs(self, *args, **kwargs):
        for _ in range(LowerBoundDecorator.MAX_TRIES):
            sample = self.component.rvs()
            # not sure whether > is the exact opposite. but <= is consistent
            if not (sample <= self.lower_bound):
                return sample  # with the other functions
        return None

    def pdf(self, x, *args, **kwargs):
        if x <= self.lower_bound:
            return 0.
        return (self.component.pdf(x)
                / (1 - self.component.cdf(self.lower_bound)))

    def pmf(self, x, *args, **kwargs):
        if x <= self.lower_bound:
            return 0.
        return (self.component.pmf(x)
                / (1 - self.component.cdf(self.lower_bound)))

    def cdf(self, x, *args, **kwargs):
        if x <= self.lower_bound:
            return 0.
        lower_mass = self.component.cdf(self.lower_bound)
        return (self.component.cdf(x) - lower_mass) / (1 - lower_mass)


class Distribution(ParameterStructure):
    """
    Distribution of parameters for a model.

    A distribution is a collection of RVs and/or distributions.
    Essentially something like a dictionary
    of random variables or distributions.
    The variables from which the distribution is initialized are
    independent.

    This should be used to define a prior.
    """

    def __repr__(self):
        return "<Distribution {keys}>".format(
            keys=str(list(self.get_parameter_names()))[1:-1])

    @classmethod
    def from_dictionary_of_dictionaries(cls,
                                        dict_of_dicts: dict) -> "Distribution":
        """
        Create distribution from dictionary of dictionaries

        Parameters
        ----------
        dict_of_dicts: dict
            The keys of the dict indicate the parameters names.
            The values are itself dictionaries representing scipy.stats
            distribution. I.e. the have the key "name" and at least one
            of the keys "args" or "kwargs".

        Returns
        -------

        distribution: Distribution
            Created distribution.
        """

        rv_dictionary = {}
        for key, value in dict_of_dicts.items():
            rv_dictionary[key] = RV.from_dictionary(value)
        return cls(rv_dictionary)

    def copy(self) -> "Distribution":
        """
        Copy the distribution

        Returns
        -------

        copied_distribution: Distribution
            A copy of the distribution.
        """

        return self.__class__(**{key: value.copy()
                                 for key, value in self.items()})

    def update_random_variables(self, **random_variables):
        """
        Update random variables within the distribution

        Parameters
        ----------

        **random_variables:
            keywords are the parameters' names, the values are random variable.

        """

        self.update(random_variables)

    def get_parameter_names(self) -> list:
        """
        Sorted list of parameter names.

        Returns
        -------

        sorted_names: list
            Sorted list of parameter names.
        """

        return sorted(self.keys())

    def rvs(self) -> Parameter:
        """
        Sample from joint distribution

        Returns
        -------

        parameter: Parameter
            A parameter which was sampled.
        """

        return Parameter(**{key: val.rvs() for key, val in self.items()})

    def pdf(self, x: Union[Parameter, dict]):
        """
        Get combination of probability density function (for continuous
        variables) and
        probability mass function (for discrete variables) at point x

        Parameters
        ----------
        x : Union[Parameter, dict]
            Evaluate at the given Parameter ``x``.
        """
        # check if the parameters match
        if sorted(x.keys()) != sorted(self.keys()):
            raise Exception("Random variable parameter mismatch. Expected: " +
                            str(sorted(self.keys())) +
                            " got " + str(sorted(x.keys())))
        if len(self) > 0:
            res = []
            for key, val in x.items():
                try:
                    # works for continuous variables
                    res.append(self[key].pdf(val))
                except AttributeError:
                    # discrete variables do not have a pdf but a pmf
                    res.append(self[key].pmf(val))
            return reduce(lambda s, t: s * t, res)
        else:
            return 1
