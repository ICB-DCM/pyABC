import numpy as np
from scipy.optimize import curve_fit


def _power_law(x, a, b):
    return a * x ** (-b)


def _finverse(y, a, b):
    return (a / y) ** (1 / b)


def fitpowerlaw(x, y):
    x = np.array(x)
    y = np.array(y)
    popt, _ = curve_fit(_power_law, x, y, p0=[0.5, 1 / 5])
    return popt, lambda x: _power_law(x, *popt), lambda y: _finverse(y, *popt)
