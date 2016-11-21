import matplotlib.pyplot as plt
from style import magicrun, magicin, magicout


@magicrun
def transform(input):
    fig, ax = plt.subplots()
    ax.plot([1, 4, 2])
    ax.set_xlabel(input)
    return fig

#transform()

fig, ax = plt.subplots()
ax.plot([1, 4, 2])
ax.set_xlabel(magicin("default value"))
magicout(fig)