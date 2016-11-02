from collections import UserDict
import numbers
from typing import Any, TypeVar

class Result:
    def __init__(self, data):
        self.data = data

    def to_summary_statistics(self, summary_statistics) -> "SummaryStatistics":
        return self

    def to_distance(self, distance) -> "Distance":
        return self

    def accepted(self, eps) -> bool:
        return self


def result_factory(res) -> Result:
    if isinstance(res, Result):
        return res
    if isinstance(res, numbers.Number):
        return Distance(res)
    if isinstance(res, dict):
        return SummaryStatistics(res)
    if isinstance(res, bool):
        return Accepted() if res else Rejected()
    return RawOutput(res)


class RawOutput(Result):
    def to_summary_statistics(self, summary_statistics) -> "SummaryStatistics":
        return SummaryStatistics(summary_statistics(self.data))


class SummaryStatistics(Result, UserDict):
    def to_distance(self, distance) -> "Distance":
        return Distance(distance(self.data))


class Distance(Result):
    def accepted(self, eps) -> bool:
        return self.data <= eps

    def __float__(self):
        return float(self.data)


class Rejected(Result):
    def accepted(self, eps):
        return False


class Accepted(Result):
    def accepted(self, eps):
        return True


class Model:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return "<{} {}>".format(self.__class__.__name__, self.name)

    def sample(self, pars) -> Any:
        raise NotImplementedError()

    def summary_statistics(self, pars, sum_stats) -> dict:
        return sum_stats(self.sample(pars))

    def distance(self, pars, distance, sum_stats) -> float:
        return distance(self.summary_statistics(pars, sum_stats))

    def accept(self, pars, eps, distance, sum_stats) -> bool:
        return self.distance(pars, distance, sum_stats) <= eps


class SimpleModel(Model):
    def __init__(self, name, sample_function):
        super().__init__(name)
        self.sample_function = sample_function

    def sample(self, pars):
        return self.sample_function(pars)
