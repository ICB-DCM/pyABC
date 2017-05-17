"""
Models
======

Models for ABCSMC
"""

from .parameters import Parameter
from typing import Callable, Any

__all__ = ["Model", "SimpleModel", "ModelResult", "IntegratedModel"]


class ModelResult:
    """
    Result of a model evaluation.
    Allows to flexibly return everything from summary_statistics to
    accepted/rejected.
    """
    def __init__(self, sum_stats=None, distance=None, accepted=None):
        self.sum_stats = sum_stats if sum_stats is not None else {}
        self.distance = distance
        self.accepted = accepted


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

        name: str
            A descriptive name of the model. This name can simplify further
            analysis for the user as it is stored in the database.
    """
    def __init__(self, name: str="model"):
        self.name = name

    def __repr__(self):
        return "<{} {}>".format(self.__class__.__name__, self.name)

    def sample(self, pars):
        """
        Return a sample from the model evaluated ar parameters ``pars``.

        This method has to be implemented by any subclass.

        Parameters
        ----------
        pars: dictionary of parameters

        Returns
        -------

        sample: any
            The sampled data.
        """
        raise NotImplementedError()

    def summary_statistics(self, pars, sum_stats_calculator) -> ModelResult:
        """
        Calculate the summary statistics.

        Parameters
        ----------
        pars: Model parameters
        sum_stats_calculator: A function which calculates summary statistics
            the user is free to use or ignore this function

        Returns
        -------

        model_result: ModelResult
            The result filled with summary statistics
        """
        raw_data = self.sample(pars)
        sum_stats = sum_stats_calculator(raw_data)
        return ModelResult(sum_stats=sum_stats)

    def distance(self, pars, sum_stats_calculator, distance_calculator) \
            -> ModelResult:
        """
        Calculate the distance

        Parameters
        ----------
        pars: Model parameters
        sum_stats_calculator: A function which calculates summary statistics.
            The user is free to use or ignore this function.
        distance_calculator: A function which calculates the distance.
            The user is free to use or ignore this function.

        Returns
        -------

        model_result: ModelResult
            The result filled with the distance
        """
        sum_stats_result = self.summary_statistics(pars, sum_stats_calculator)
        distance = distance_calculator(sum_stats_result.sum_stats)
        sum_stats_result.distance = distance
        return sum_stats_result

    def accept(self, pars, sum_stats_calculator, distance_calculator, eps) \
            -> ModelResult:
        """
        Accept or not accept a parameter.

        Parameters
        ----------
        pars: Model parameters
        sum_stats_calculator: A function which calculates summary statistics.
            The user is free to use or ignore this function.
        distance_calculator: A function which calculates the distance.
            The user is free to use or ignore this function.
        eps: float
            Acceptance threshold

        Returns
        -------

        model_result: ModelResult
            Result filled with the accepted field.

        """
        distance_result = self.distance(pars, sum_stats_calculator,
                                        distance_calculator)
        accepted = distance_result.distance <= eps
        distance_result.accepted = accepted
        return distance_result


class SimpleModel(Model):
    """
    A model which is initialized with a function which generates the samples.
    For most cases this class is to be used.
    Note that you can also pass a plain function to the ABCSMC class.
    This function get automatically converted to a SimpleModel.

    Parameters
    ----------

    sample_function: Callable[[Parameter], Any]
        Returns the sample to be passed to the summary statistics method.
        This function as a single argument which is a Parameter.

    name: str. optional
        The name of the model. If not provided, the names if inferred from
        the function name of `sample_function`.
    """
    def __init__(self, sample_function: Callable[[Parameter], Any], name=None):
        if name is None:
            name = sample_function.__name__
        super().__init__(name)
        self.sample_function = sample_function

    def sample(self, pars):
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
    def integrated_simulate(self, pars, eps: float) -> ModelResult:
        """
        Method which integrates simulation and acceptance/rejection
        in a single method.

        Parameters
        ----------
        pars: dict
            Parameters at which to evaluate the model

        eps: float
            Current acceptance threshold

        Returns
        -------

        model_result: ModelResult
            In case the parameter evaluation is rejected, this method
            should simply return ``ModelResult(accepted=False)``.
            If the parameter was accepted, this method should return either
            ``ModelResult(accepted=True, distance=distance)`` or
            ``ModelResult(accepted=True, distance=distance, \
sum_stats=sum_stats)``
            in which ``distance`` denotes the achieved
            distance and ``sum_stats`` the summary statistics (e.g. simulated
            data) of the run. Note that providing the summary statistics
            is optional. If they are provided, then they are also logged in
            the database.
        """
        raise NotImplementedError()

    def summary_statistics(self, pars, sum_stats_calculator):
        return ModelResult()

    def accept(self, pars, sum_stats_calculator, distance_calculator, eps):
        return self.integrated_simulate(pars, eps)
