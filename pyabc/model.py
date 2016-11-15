"""
Models
======

Models afor ABCSMC
"""


class ModelResult:
    """
    Result of a model evaluation.
    Allows to flexibly return everything from summary_statistics to accepted/rejected.
    """
    def __init__(self, sum_stats=None, distance=None, accepted=None):
        self.sum_stats = sum_stats if sum_stats is not None else {}
        self.distance = distance
        self.accepted = accepted


class Model:
    """
    General ABC Model. This is to be subclassed.

    The individual steps

      * sample
      * summary_statistics
      * distance
      * accept

    are to be overwritten.

    Every model has to have a working summary_statistics implementation
    and a working accept implementation. The summary_statistics method
    is necessary to initialize distance functions are epsilons.
    """
    def __init__(self, name="model"):
        self.name = name

    def __repr__(self):
        return "<{} {}>".format(self.__class__.__name__, self.name)

    def sample(self, pars):
        raise NotImplementedError()

    def summary_statistics(self, pars, sum_stats_calculator) -> ModelResult:
        raw_data = self.sample(pars)
        sum_stats = sum_stats_calculator(raw_data)
        return ModelResult(sum_stats=sum_stats)

    def distance(self, pars, sum_stats_calculator, distance_calculator) -> ModelResult:
        sum_stats_result = self.summary_statistics(pars, sum_stats_calculator)
        distance = distance_calculator(sum_stats_result.sum_stats)
        sum_stats_result.distance = distance
        return sum_stats_result

    def accept(self, pars, sum_stats_calculator, distance_calculator, eps) -> ModelResult:
        distance_result = self.distance(pars, sum_stats_calculator, distance_calculator)
        accepted = distance_result.distance <= eps
        distance_result.accepted = accepted
        return distance_result


class SimpleModel(Model):
    """
    A model which is to be initialized with a function which generates the sample.

    Parameters
    ----------

    sample_function: Callable[[Parameter], Any]
        Returns the sample to be passed to the summary statistics method
    """
    def __init__(self, sample_function, name=None):
        if name is None:
            name = sample_function.__name__
        super().__init__(name)
        self.sample_function = sample_function

    def sample(self, pars):
        return self.sample_function(pars)
