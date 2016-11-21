import matplotlib.pyplot as plt
from style import magicrun, magicin, magicout


import pdb
pdb.set_trace()

fig, ax = plt.subplots()
ax.plot([1, 4, 2])
ax.set_xlabel(magicin("default value"))

r = magicout(fig)