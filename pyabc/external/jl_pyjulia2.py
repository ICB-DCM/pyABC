import numpy as np

try:
    from julia import Main
except ImportError:
    Main = None


def _dict_lists2arrays(dct):
    for key, val in dct.items():
        dct[key] = np.asarray(val)
    return dct


def _read_source(module_name, source_file):
    if not hasattr(Main, module_name):
        Main.include(source_file)


class ModelPy(object):
    def __init__(self, module_name, source_file, function_name):
        self.module_name = module_name
        self.source_file = source_file
        self.function_name = function_name

        _read_source(self.module_name, self.source_file)
        self.__name__ = "Model"

        self.model = getattr(
            getattr(Main, self.module_name), self.function_name
        )

    def __call__(self, par):
        ret = self.model(par)
        ret = _dict_lists2arrays(ret)
        return ret

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

        self.model = getattr(
            getattr(Main, self.module_name), self.function_name
        )


class DistancePy(object):
    def __init__(self, module_name, source_file, function_name):
        self.module_name = module_name
        self.source_file = source_file
        self.function_name = function_name
        self.__name__ = "Distance"

        _read_source(self.module_name, self.source_file)

        self.distance = getattr(
            getattr(Main, self.module_name), self.function_name
        )

    def __call__(self, y, y0):
        return float(self.distance(y, y0))

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

        self.distance = getattr(
            getattr(Main, self.module_name), self.function_name
        )


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
        _read_source(self.module_name, self.source_file)

    def model(self, function_name: str = "model"):

        return ModelPy(
            self.module_name,
            self.source_file,
            function_name,
        )

    def observation(self, name: str = "observation"):
        observation = getattr(getattr(Main, self.module_name), name)

        observation = _dict_lists2arrays(observation)

        return observation

    def distance(self, function_name: str = "distance"):
        return DistancePy(
            self.module_name,
            self.source_file,
            function_name,
        )

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
