from abc import ABCMeta
import numpy as np
import pandas as pd
import functools


def wrap_fit(f):
    @functools.wraps(f)
    def fit(self, X, w):
        self.X = X
        self.w = w
        if len(X.columns) == 0:
            self.no_parameters = True
            return
        self.no_parameters = False
        if w.size > 0:
            if not np.isclose(w.sum(), 1):
                w /= w.sum()
        f(self, X, w)
    return fit


def wrap_pdf(f):
    @functools.wraps(f)
    def pdf(self, x):
        if self.no_parameters:
            return 1
        return f(self, x)
    return pdf


def wrap_rvs_single(f):
    @functools.wraps(f)
    def rvs_single(self):
        if self.no_parameters:
            return pd.Series()
        return f(self)
    return rvs_single


class TransitionMeta(ABCMeta):
    """
    This metaclass handles the special case of no parameters.
    Transition classes do not have to check for it anymore
    """
    def __init__(cls, name, bases, attrs):
        ABCMeta.__init__(cls, name, bases, attrs)
        cls.fit = wrap_fit(cls.fit)
        cls.pdf = wrap_pdf(cls.pdf)
        cls.rvs_single = wrap_rvs_single(cls.rvs_single)
