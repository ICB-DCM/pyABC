"""Global settings"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rcParams
from typing import List


def set_figure_params(
    theme: str = 'pyabc',
    style: str = None,
    color_map: str = 'plasma',
    color_cycle: List[str] = None,
) -> None:
    """Set global figure parameters for a consistent, beautified design.

    Parameters
    ----------
    theme:
        Overall theme. Possible: 'pyabc', 'default'
        (the matplotlib defaults, ignoring all other inputs).
    style:
        Matplotlib overall style set. Possible values e.g. 'dark_background',
        'ggplot', 'seaborn'.
        Note that the other parameters are applied on top, so should be None
        if that is not desirable.
        See e.g.:
        https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html  # noqa
    color_map:
        Colormap ('image.cmap').
    color_cycle:
        Color cycle ('axes.prop_cycle').

    **Note:** This modifies global matplotlib parameters.
    """
    theme = theme.lower()
    if theme == 'pyabc':
        _set_figure_params_pyabc(
            style=style,
            color_map=color_map,
            color_cycle=color_cycle,
        )
    elif theme == 'default':
        _set_figure_params_default()
    else:
        raise ValueError(f"Theme not recognized: {theme}")


def _set_figure_params_pyabc(
    style: str,
    color_map: str,
    color_cycle: List[str],
) -> None:
    """Set layout parameters for style 'pyabc'"""
    # overall style
    if style is not None:
        plt.style.use(style)

    # spines
    rcParams['axes.spines.right'] = False
    rcParams['axes.spines.top'] = False

    # colors
    if color_map is not None:
        rcParams['image.cmap'] = color_map
    if color_cycle is not None:
        rcParams['axes.prop_cycle'] = color_cycle


def _set_figure_params_default() -> None:
    """Set layout parameters for the default matplotlib style"""
    mpl.rcdefaults()
