from collections import UserDict
import numbers


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
