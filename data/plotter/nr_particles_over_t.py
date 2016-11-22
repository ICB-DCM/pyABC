import style
from style import make
import matplotlib.pyplot as plt
from pyabc import History
import seaborn as sns
import scipy as sp
from scipy.stats import variation

sm = make(output="test.pdf", input="/home/emmanuel/abc/data/raw/toy:modes=1.db",
          wildcards=["1"])

history = History("sqlite:///" + sm.input[0], 23, ["sdf"])
history.id = 1


nr_particles = sp.array([len(history.weighted_parameters_dataframe(t, 0)[0])
                for t in range(1, history.max_t)])


weights = [history.weighted_parameters_dataframe(t, 0)[1] for t in range(1, history.max_t)]

effective_samples_size = sp.array([w.size/ (1+variation(w)) for w in weights])
                
#%%
fig, ax = fig, ax = plt.subplots()
ax.scatter(sp.arange(nr_particles.size) + 1, nr_particles,
           color=sns.utils.get_color_cycle()[0], label=style.name.nr_particles)
ax.scatter(sp.arange(nr_particles.size) + 1, effective_samples_size,
           color=sns.utils.get_color_cycle()[1], label=style.name.effective_samples_size)

ax.set_xlabel(style.name.nr_populations)
ax.set_ylabel(style.name.n, labelpad=style.labelpad.reduced)
ax.legend(loc="center left", bbox_to_anchor=(1, .5))
style.middle_ticks_minor(ax)
sns.despine(fig)
fig.save(sm.output[0])
