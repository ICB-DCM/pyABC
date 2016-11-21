from . import plotstyle
from . import color
import inspect
import matplotlib.pyplot as plt
from collections import namedtuple
import os


class dummy:
    input = ["magic input dummy"]
    output = ["magicoutdummy.pdf"]


SnakemakeDummy = namedtuple("SnakemakeDummy", "input output")


def tmp_file(file):
    return os.path.join(os.path.expanduser("~/tmp"), file)


snakemake = None


def make(*, input="", output=""):
    """
    Inspect the stack to get the snakemake object fomr the outer frame
    """
    global snakemake
    stack = inspect.stack()
    for s in stack:
        inside = "snakemake" in s.frame.f_locals
        if inside:
            snakemake = s.frame.f_locals["snakemake"]
    snakemake = SnakemakeDummy(input=[tmp_file(input)], output=[tmp_file(output)])
    return snakemake


def auto_input(snakemake):
    return snakemake.input[0]

this_input = auto_input(snakemake)


def auto_store(snakemake, result):
    if isinstance(result, plt.Figure):
        result.savefig(snakemake.output[0], bbox_inches="tight")
        return
    with open(snakemake.output[0], "w") as f:
        f.write(result)


def magicrun(f):
    def autoload_and_store():
        print("RUN")
        input = this_input
        result = f(input)
        auto_store(snakemake, result)
        return result
    return autoload_and_store


def magicin(default=None):
    if snakemake is dummy:
        return default
    return auto_input(snakemake)


def magicout(result):
    auto_store(snakemake, result)
    return result
