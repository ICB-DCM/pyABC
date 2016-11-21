import style
from style import make
import matplotlib.pyplot as plt
from pyabc import History
import seaborn as sns
import scipy as sp

sm = make(output="test.pdf", input="/home/emmanuel/abc/data/raw/toy:modes=1.db",
          wildcards=["1"])

history = History("sqlite:///" + sm.input[0], 23, ["sdf"])
history.id = 1


nr_particles = sp.array([len(history.weighted_parameters_dataframe(t, 0)[0])
                for t in range(1, history.max_t)])


weights = [history.weighted_parameters_dataframe(t, 0)[1] for t in range(1, history.max_t)]

                
#%%
fig, ax = fig, ax = plt.subplots()
ax.scatter(sp.arange(nr_particles.size) + 1, nr_particles,
           color=sns.utils.get_color_cycle()[0])
#ax.scatter(sp.arange(nr_particles.size) + 1, eff_sample_size,
#           color=sns.utils.get_color_cycle()[1])

ax.set_xlabel(style.name.nr_populations)
ax.set_ylabel(style.name.nr_particles)
style.middle_ticks_minor(ax)
sns.despine(fig)
fig.save(sm.output[0])
