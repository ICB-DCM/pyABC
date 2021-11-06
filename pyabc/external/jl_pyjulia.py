import numpy as np

try:
    from julia import Main
except ImportError:
    Main = None


def _dict_lists2arrays(dct):
    for key, val in dct.items():
        dct[key] = np.asarray(val)
    return dct


class Julia:
    """Interfact to Julia."""

    def __init__(self, module_name: str, source_file: str = None):
        if Main is None:
            raise ImportError(
                "Install pyjulia, e.g. via `pip install pyabc[julia]`",
            )
        self.module_name: str = module_name
        if source_file is None:
            source_file = module_name + ".jl"
        self.source_file: str = source_file
        self._read_source()

    def _read_source(self):
        Main.include(self.source_file)

    def model(self, function_name: str = "model"):
        model = getattr(getattr(Main, self.module_name), function_name)

        def model_py(par):
            ret = model(par)
            ret = _dict_lists2arrays(ret)
            return ret

        return model_py

    def observation(self, name: str = "observation"):
        observation = getattr(getattr(Main, self.module_name), name)

        observation = _dict_lists2arrays(observation)

        return observation

    def distance(self, function_name: str = "distance"):
        distance = getattr(getattr(Main, self.module_name), function_name)

        def distance_py(y, y0):
            return float(distance(y, y0))

        return distance_py

    def display_source_ipython(self):
        """
        Convenience method to print the loaded source file
        as syntax highlighted HTML within IPython.
        """
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
