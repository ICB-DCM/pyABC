"""
Models
======

A model defines how input parameters relate to output simulated data.
"""

from .parameters import Parameter
from typing import Callable, Any
from .epsilon import Epsilon
from .distance import Distance
from .acceptor import Acceptor


class ModelResult:
    """
    Result of a model evaluation.
    Allows to flexibly return summary statistics,
    distances and accepted/rejected.
    """

    def __init__(self,
                 sum_stats: dict = None,
                 distance: float = None,
                 accepted: bool = None,
                 weight: float = 1.0):
        self.sum_stats = sum_stats if sum_stats is not None else {}
        self.distance = distance
        self.accepted = accepted
        self.weight = weight


class Model:
    """
    General model. This is the most flexible model class, but
    also the most complicated one to use.
    This is an abstract class and not functional on its own.
    Derive concrete subclasses for actual usage.

    The individual steps

      * sample
      * summary_statistics
      * distance
      * accept

    can be overwritten.

    To use this class, at least the sample method has to be overriden.

    .. note::

        Most likely you do not want to use this class directly, but the
        :class:`SimpleModel` instead, or even just pass a plain function
        as model.

    Parameters
    ----------
    name: str, optional (default = "model")
        A descriptive name of the model. This name can simplify further
        analysis for the user as it is stored in the database.
    """

    def __init__(self, name: str = "Model"):
        self.name = name

    def __repr__(self):
        return "<{} {}>".format(self.__class__.__name__, self.name)

    def sample(self, pars: Parameter):
        """
        Return a sample from the model evaluated at parameters ``pars``. This
        can be raw data, or already summarized statistics thereof.

        This method has to be implemented by any subclass.

        Parameters
        ----------
        pars: Parameter
            Dictionary of parameters.

        Returns
        -------
        sample: any
            The sampled data.
        """
        raise NotImplementedError()

    def summary_statistics(self,
                           t: int,
                           pars: Parameter,
                           sum_stats_calculator: Callable) -> ModelResult:
        """
        Sample, and then calculate the summary statistics.

        Called from within ABCSMC during the initialization process.

        Parameters
        ----------
        t: int
            Current time point.
        pars: Parameter
            Model parameters.
        sum_stats_calculator: Callable
            A function which calculates summary statistics, as passed to
            :class:`pyabc.smc.ABCSMC`.
            The user is free to use or ignore this function.

        Returns
        -------
        model_result: ModelResult
            The result with filled summary statistics.
        """
        raw_data = self.sample(pars)
        sum_stats = sum_stats_calculator(raw_data)
        return ModelResult(sum_stats=sum_stats)

    def distance(self,
                 t: int,
                 pars: Parameter,
                 sum_stats_calculator: Callable,
                 distance_calculator: Distance,
                 x_0: dict) -> ModelResult:
        """
        Sample, calculate summary statistics, and then calculate the distance.

        Not required in the current implementation.

        Parameters
        ----------
        t: int
            Current time point.
        pars: Parameter
            Model parameters.
        sum_stats_calculator: Callable
            A function which calculates summary statistics, as passed to
            :class:`pyabc.smc.ABCSMC`.
            The user is free to use or ignore this function.
        distance_calculator: Callable
            A function which calculates the distance, as passed to
            :class:`pyabc.smc.ABCSMC`.
            The user is free to use or ignore this function.
        x_0: dict
            Observed summary statistics.

        Returns
        -------
        model_result: ModelResult
            The result with filled distance.
        """

        sum_stats_result = self.summary_statistics(t,
                                                   pars,
                                                   sum_stats_calculator)
        distance = distance_calculator(sum_stats_result.sum_stats,
                                       x_0,
                                       t,
                                       pars)
        sum_stats_result.distance = distance

        return sum_stats_result

    def accept(self,
               t: int,
               pars: Parameter,
               sum_stats_calculator: Callable,
               distance_calculator: Distance,
               eps_calculator: Epsilon,
               acceptor: Acceptor,
               x_0: dict):
        """
        Sample, calculate summary statistics, calculate distance, and then
        accept or not accept a parameter.

        Called from within ABCSMC in each iteration to evaluate a parameter.


        Parameters
        ----------
        t: int
            Current time point.
        pars: Parameter
            The model parameters.
        sum_stats_calculator: Callable
            A function which calculates summary statistics.
            The user is free to use or ignore this function.
        distance_calculator: pyabc.Distance
            The distance function.
            The user is free to use or ignore this function.
        eps_calculator: pyabc.Epsilon
            The acceptance thresholds.
        acceptor: pyabc.Acceptor
            The acceptor judging whether to accept, based on distance and
            epsilon.
        x_0: dict
            The observed summary statistics.

        Returns
        -------
        model_result: ModelResult
            The result with filled accepted field.

        """
        result = self.summary_statistics(t,
                                         pars,
                                         sum_stats_calculator)
        acc_res = acceptor(
            distance_function=distance_calculator,
            eps=eps_calculator,
            x=result.sum_stats,
            x_0=x_0,
            t=t,
            par=pars)
        result.distance = acc_res.distance
        result.accepted = acc_res.accept
        result.weight = acc_res.weight

        return result


class SimpleModel(Model):
    """
    A model which is initialized with a function which generates the samples.
    For most cases this class will be adequate.
    Note that you can also pass a plain function to the ABCSMC class, which
    then gets automatically converted to a SimpleModel.

    Parameters
    ----------
    sample_function: Callable[[Parameter], Any]
        Returns the sample to be passed to the summary statistics method.
        This function as a single argument which is a Parameter.
    name: str. optional
        The name of the model. If not provided, the names if inferred from
        the function name of `sample_function`.
    """

    def __init__(self,
                 sample_function: Callable[[Parameter], Any],
                 name: str = None):
        if name is None:
            name = sample_function.__name__
        super().__init__(name)
        self.sample_function = sample_function

    def sample(self, pars: Parameter):
        return self.sample_function(pars)

    @staticmethod
    def assert_model(model_or_function):
        """
        Alternative constructor. Accepts either a Model instance or a
        function and returns always a Model instance.

        Parameters
        ----------
        model_or_function: Model, function
            Constructs a SimpleModel instance if a function is passed.
            If a Model instance is passed, the Model instance itself is
            returned.

        Returns
        -------
        model: SimpleModel or Model

        """
        if isinstance(model_or_function, Model):
            return model_or_function
        else:
            return SimpleModel(model_or_function)


class IntegratedModel(Model):
    """
    A model class which integrates simulation, distance calculation
    and rejection/acceptance.

    This can bring performance improvements if the user can calculate
    the distance function on the fly during model simulation and interrupt
    the simulation if the current acceptance threshold cannot be satisfied
    anymore.

    Subclass this model and implement ``integrated_simulate`` to define
    your own integrated model..
    """

    def integrated_simulate(self,
                            pars: Parameter,
                            eps: float) -> ModelResult:
        """
        Method which integrates simulation and acceptance/rejection
        in a single method.

        Parameters
        ----------
        pars: Parameter
            Parameters at which to evaluate the model
        eps: float
            Current acceptance threshold. If required, it is effortlessly
            possible to instead use the entire epsilon_calculator object
            passed to accept().

        Returns
        -------
        model_result: ModelResult
            In case the parameter evaluation is rejected, this method
            should simply return ``ModelResult(accepted=False)``.
            If the parameter was accepted, this method should return either
            ``ModelResult(accepted=True, distance=distance)`` or
            ``ModelResult(accepted=True, distance=distance,
            sum_stats=sum_stats)``
            in which ``distance`` denotes the achieved
            distance and ``sum_stats`` the summary statistics (e.g. simulated
            data) of the run. Note that providing the summary statistics
            is optional. If they are provided, then they are also logged in
            the database.
        """
        raise NotImplementedError()

    def accept(self,
               t: int,
               pars: Parameter,
               sum_stats_calculator: Callable,
               distance_calculator: Distance,
               eps_calculator: Epsilon,
               acceptor: Acceptor,
               x_0: dict):
        return self.integrated_simulate(pars, eps_calculator(t))
