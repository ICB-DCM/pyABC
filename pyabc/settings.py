"""Global settings"""

import matplotlib as mpl
from matplotlib import rcParams


def set_figure_params(
    style: str = 'pyabc',
    color_map: str = 'plasma',
):
    """Set global figure parameters for a consistent, beautified design.

    Parameters
    ----------
    style:
        Style set to use. Possible: 'pyabc', 'default'
        (the matplotlib defaults).
    color_map:
        Color

    **Note:** This affects global matplotlib parameters.
    """
    style = style.lower()
    if style == 'pyabc':
        set_figure_params_pyabc(
            color_map=color_map)
    elif style == 'default':
        set_figure_params_default()
    else:
        raise ValueError(f"Style not recognized: {style}")


def set_figure_params_pyabc(color_map: str):
    """Set layout parameters for style 'pyabc'"""
    rcParams['axes.spines.right'] = False
    rcParams['axes.spines.top'] = False
    if color_map is not None:
        rcParams['image.cmap'] = color_map


def set_figure_params_default():
    """Set layout parameters for the default matplotlib style"""
    mpl.rcdefaults()
