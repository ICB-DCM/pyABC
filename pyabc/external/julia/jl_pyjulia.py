"""Interface to Julia via PyJulia."""

from typing import Any, Callable, Dict, Union

import numpy as np
import pandas as pd

try:
    from julia import Main
except ImportError:
    pass


def _dict_vars2py(
    dct: Dict[str, Any]
) -> Dict[str, Union[np.ndarray, pd.Series, pd.DataFrame]]:
    """Convert non-pandas dictionary entries to numpy arrays.

    Parameters
    ----------
    dct: Model simulations as returned from the PyJulia call.

    Returns
    -------
    dct: The same dictionary with modified values.
    """
    for key, val in dct.items():
        if not isinstance(val, (pd.DataFrame, pd.Series)):
            dct[key] = np.asarray(val)
    return dct


def _read_source(module_name: str, source_file: str) -> None:
    """Read source if module not attached to julia Main yet.

    Parameters
    ----------
    module_name: Julia module name.^^
    source_file: Qualified Julia source file.
    """
    if not hasattr(Main, module_name):
        Main.include(source_file)


class _JlWrap(object):
    """Wrapper around Julia object.

    This in particular makes the objects pickleable, by reconstruction from
    source.
    """

    def __init__(self, module_name: str, source_file: str, function_name: str):
        self.module_name: str = module_name
        self.source_file: str = source_file
        self.function_name: str = function_name

        _read_source(source_file=source_file, module_name=module_name)
        self.callable = self._get_callable()

    def _get_callable(self):
        """Get callable from Julia module."""
        return getattr(getattr(Main, self.module_name), self.function_name)

    def __getstate__(self):
        return {
            "module_name": self.module_name,
            "source_file": self.source_file,
            "function_name": self.function_name,
        }

    def __setstate__(self, d):
        for key, val in d.items():
            setattr(self, key, val)
        _read_source(self.module_name, self.source_file)
        self.callable = self._get_callable()

    @property
    def __name__(self):
        return (
            f"{self.__class__.__name__}_{self.function_name}_"
            f"{self.module_name}_{self.source_file}"
        )


class _JlModel(_JlWrap):
    """Wrapper around a Julia model."""

    def __call__(self, par):
        ret = self.callable(par)
        ret = _dict_vars2py(ret)
        return ret


class _JlDistance(_JlWrap):
    """Wrapper around a Julia distance."""

    def __call__(self, y, y0):
        return float(self.callable(y, y0))


class Julia:
    """Interface to Julia via PyJulia.

    This class provides `model`, `distance`, and `observable` wrappers
    around Julia objects.
    It expects the corresponding Julia objects to be defined in a
    `source_file` within a `module_name`.

    The Julia model is expected to return a dictionary object compatible
    with pyABC's model simulation type, in particular values convertible
    to numpy.ndarray, pandas.Series, or pandas.DataFrame.

    The Julia distance takes two model simulations and returns a single
    floating value.

    We use the PyJulia package to access Julia from inside Python.
    It can be installed via `pip install pyabc[julia]`, however requires
    additional Julia dependencies to be installed via:

    >>> python -c "import julia; julia.install()"

    For further information, see
    https://pyjulia.readthedocs.io/en/latest/installation.html.

    There are some known problems, e.g. with statically linked Python
    interpreters, see
    https://pyjulia.readthedocs.io/en/latest/troubleshooting.html
    for details.
    Possible solutions are to pass ``compiled_modules=False`` to the Julia
    constructor early in your code:

    >>> from julia.api import Julia
    >>> jl = Julia(compiled_modules=False)

    This however slows down loading and using Julia packages, especially for
    large ones.
    An alternative is to use the ``python-jl`` command shipped with PyJulia:

    >>> python-jl MY_SCRIPT.py

    This basically launches a Python interpreter inside Julia.
    When using Jupyter notebooks, this wrapper can be installed as an
    additional kernel via:

    >>> python -m ipykernel install --name python-jl [--prefix=/path/to/python/env]

    And changing the first argument in
    ``/path/to/python/env/share/jupyter/kernels/python-jl/kernel.json``
    to ``python-jl``.

    Model simulations are eagerly converted to Python objects
    (specifically, `numpy.ndarray` and `pandas.DataFrame`).
    This can introduce overhead and could be avoided by an alternative
    lazy implementation.
    """

    def __init__(self, module_name: str, source_file: str = None):
        if Main is None:
            raise ImportError(
                "Install PyJulia, e.g. via `pip install pyabc[julia]`, "
                "and see the class documentation",
            )
        self.module_name: str = module_name
        if source_file is None:
            source_file = module_name + ".jl"
        self.source_file: str = source_file
        _read_source(self.module_name, self.source_file)

    def model(self, name: str = "model"):
        """Get a wrapped Julia model callable.

        Parameters
        ----------
        name: Name of the model within the Julia module.
        """
        return _JlModel(
            module_name=self.module_name,
            source_file=self.source_file,
            function_name=name,
        )

    def observation(self, name: str = "observation"):
        """Get the observed data from Julia.

        Parameters
        ----------
        name: Identifier of observed data variable within the Julia module.
        """
        observation = getattr(getattr(Main, self.module_name), name)
        observation = _dict_vars2py(observation)
        return observation

    def distance(self, name: str = "distance") -> Callable:
        """Get the distance function from Julia.

        Parameters
        ----------
        name: Name of the distance function within the Julia module.
        """
        return _JlDistance(
            module_name=self.module_name,
            source_file=self.source_file,
            function_name=name,
        )

    def display_source_ipython(self):
        """Display source code as syntax highlighted HTML within IPython."""
        import IPython.display as display
        from pygments import highlight
        from pygments.formatters import HtmlFormatter
        from pygments.lexers import JuliaLexer

        with open(self.source_file) as f:
            code = f.read()

        formatter = HtmlFormatter()
        return display.HTML(
            '<style type="text/css">{}</style>{}'.format(
                formatter.get_style_defs('.highlight'),
                highlight(code, JuliaLexer(), formatter),
            )
        )
