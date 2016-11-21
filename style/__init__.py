import inspect
from collections import namedtuple
import os
from snakemake.script import Snakemake
from . import name
from . import labelpad


class dummy:
    input = ["magic input dummy"]
    output = ["magicoutdummy.pdf"]


SnakemakeDummy = namedtuple("SnakemakeDummy", "input output wildcards")


def tmp_file(file):
    return os.path.join(os.path.expanduser("~/tmp"), file)


class holder:
    snakemake = None


def make(*, input="", output="", wildcards=None):
    """
    Inspect the stack to get the snakemake object fomr the outer frame
    """
    stack = inspect.stack()
    for s in stack:
        inside = "snakemake" in s.frame.f_locals
        if inside:
            snm = s.frame.f_locals["snakemake"]
            break
    else:
        snm = SnakemakeDummy(input=[tmp_file(input)], output=[tmp_file(output)], wildcards=wildcards)
    holder.snakemake = snm
    return snm

if holder.snakemake is None:
    holder.snakemake = make()


def snakemakerun():
    return isinstance(holder.snakemake, Snakemake)

if snakemakerun():
    import matplotlib as mpl
    mpl.use("Agg", force=True)
    mpl.interactive(False)


def auto_input():
    return holder.snakemake.input

this_input = auto_input()


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
        auto_store(holder.snakemake, result)
        return result
    return autoload_and_store


def magicin(default=None):
    if holder.snakemake is dummy:
        return default
    return auto_input(holder.snakemake)


def magicout(result):
    auto_store(holder.snakemake, result)
    return result


# it is important that matplotlib is imported only here
# to set the backend to agg before that if the run
# is within snakemake
from . import plotstyle
from . import color
import matplotlib.pyplot as plt
from .plot import set_ticksmiddle_minor, remove_middle_tick_labels, plot, middle_ticks_minor