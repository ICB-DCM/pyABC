import matplotlib.ticker
import scipy as sp


class ExpFormatter(matplotlib.ticker.StrMethodFormatter):
    def __init__(self):
        super().__init__("{x:.0e}")

    def __call__(self, x, pos=None):
        if x == 0:
            return "0"
        exponent = int(sp.log10(x))
        factor = x / 10**exponent
        if int(factor) == float(factor):
            factor = int(factor)
        else:
            factor = float(factor)

        exponent = int(exponent)

        return "$\mathdefault{" + "{} · 10^{{{}}}".format(factor, exponent) + "}$"