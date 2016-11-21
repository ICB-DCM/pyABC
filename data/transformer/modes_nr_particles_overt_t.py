import matplotlib.pyplot as plt
from style import make, magicrun


@magicrun
def transform(input):
    return "transformcall" + input

transform()