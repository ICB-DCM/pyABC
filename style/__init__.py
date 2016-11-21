from . import plotstyle
import inspect


class dummy:
    input = ["dummy in"]
    output = ["dummy out"]


def make():
    """
    Inspect the stack to get the snakemake object fomr the outer frame
    """
    stack = inspect.stack()
    for s in stack:
        inside = "snakemake" in s.frame.f_locals
        if inside:
            return s.frame.f_locals["snakemake"]
    return dummy


snakemake = make()


def auto_input(snakemake):
    return snakemake.input[0]

this_input = auto_input(snakemake)


def auto_store(snakemake, result):
    with open(snakemake.output[0], "w") as f:
        f.write(result)


def magicrun(f):
    def autoload_and_store():
        print("RUN")
        input = this_input
        result = f(input)
        auto_store(snakemake, result)
    return autoload_and_store