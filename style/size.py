_golden_ratio = 1.61803398875
_quadratic = 1


def _format(scale, ratio):
    return ratio * scale, scale


m = _format(1, _golden_ratio)
s = _format(.75, _golden_ratio)
xs = _format(.5, _golden_ratio)

m = _format(1, _quadratic)
sq = _format(.75, _quadratic)
xsq = _format(.5, _quadratic)
