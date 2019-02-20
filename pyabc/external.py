"""
Interface to external simulators
================================

Currently, the R language is supported.

.. note::

    The rpy2 package needs to be installed to interface with the R language.
    Installation of rpy2 is optional if R support is not required.
    See also :ref:`installation of optional dependencies <install-optional>`.
"""

try:
    from rpy2.robjects import r
except ImportError:  # in Python 3.6 ModuleNotFoundError can be used
    raise Exception("Install rpy2 to enable support for the R language")
from .random_variables import Parameter
import numpy as np
import pandas as pd
import warnings


__all__ = ["R"]


def dict_to_named_list(dct):
    if (isinstance(dct, dict)
            or isinstance(dct, Parameter)
            or isinstance(dct, pd.core.series.Series)):
        return r.list(**{key: val for key, val in dct.items()})
    return dct


# The way unpickling is done is not optimal
# The objects reloaded when reading the source again will be different
# from the unpickled objects.
# This might cause problems if the R code relies on side effects/
#
# On the other hand, this might be the intended behavior, if for example
# one of the objects is modified by hand within Python.


class R:
    """
    Interface to R.

    Parameters
    ----------

    source_file: str

        Path to the file which contains the definitions for
        the model, the summary statistics and the distance function as
        well as the observed data.
    """
    def __init__(self, source_file: str):
        warnings.warn("The support of the R language for ABC-SMC is "
                      "considered experimental. The API might change in the "
                      "future.")
        self.source_file = source_file
        self._read_source()

    def __getstate__(self):
        return self.source_file

    def __setstate__(self, state):
        self.source_file = state
        self._read_source()

    def _read_source(self):
        r.source(self.source_file)

    def display_source_ipython(self):
        """
        Convenience method to print the loaded source file
        as syntax highlighted HTML within IPython.
        """
        from pygments import highlight
        from pygments.lexers import SLexer

        from pygments.formatters import HtmlFormatter
        import IPython.display as display

        with open(self.source_file) as f:
            code = f.read()

        formatter = HtmlFormatter()
        return display.HTML('<style type="text/css">{}</style>{}'.format(
            formatter.get_style_defs('.highlight'),
            highlight(code, SLexer(), formatter)))

    def model(self, function_name: str):
        """
        The R-model.

        Parameters
        ----------
        function_name: str
            Name of the function in the R script which defines the model.

        Returns
        -------

        model: callable
            The model.

        """
        model = r[function_name]

        def model_py(par):
            return model(dict_to_named_list(par))

        model_py.__name__ = function_name
        # set reference to this class to ensure the source file is
        # read again when unpickling
        model_py._R = self
        return model_py

    def distance(self, function_name: str):
        """
        The R-distance function.

        Parameters
        ----------
        function_name: str
            Name of the function in the R script which defines the distance
            function.

        Returns
        -------

        distance: callable
            The distance function.

        """
        distance = r[function_name]

        def distance_py(*args):
            args = tuple(dict_to_named_list(d) for d in args)
            return float(np.array(distance(*args)))

        distance_py.__name__ = function_name
        # set reference to this class to ensure the source file is
        # read again when unpickling
        distance_py._R = self
        return distance_py

    def summary_statistics(self, function_name: str):
        """
        The R-summary statistics.

        Parameters
        ----------
        function_name: str
          Name of the function in the R script which defines the summary
          statistics function.

        Returns
        -------

        summary_statistics: callable
            The summary statistics function.
        """
        summary_statistics = r[function_name]
        summary_statistics.__name__ = function_name
        # set reference to this class to ensure the source file is
        # read again when unpickling
        summary_statistics._R = self
        return summary_statistics

    def observation(self, name: str):
        """
        The summary statistics of the observed data as defined in R.

        Parameters
        ----------

        name: str
            Name of the named list defined in the R script which holds
            the observed data.

        Returns
        -------

        observation: r named list
            A dictionary like object which holds the summary statistics
            of the observed data.
        """
        obs = r[name]
        # set reference to this class to ensure the source file is
        # read again when unpickling
        obs._r = self
        return obs
