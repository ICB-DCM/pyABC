import style
from style import make
import matplotlib.pyplot as plt
from pyabc import History
import seaborn as sns
import scipy as sp
from scipy.stats import variation

sm = make(output="test.pdf",
          input="/home/emmanuel/abc/data/raw/toy:modes=1.db",
          wildcards=["1"])

history = History("sqlite:///" + sm.input[0], 23, ["sdf"])
history.id = 1


nr_particles = sp.array([len(history.weighted_parameters_dataframe(t, 0)[0])
                for t in range(1, history.max_t)])

populations = history.get_all_populations()
epsilons = populations.epsilon[1:]  # the very fist ist just 0 by definition

weights = [history.weighted_parameters_dataframe(t, 0)[1]
           for t in range(1, history.max_t)]

effective_samples_size = sp.array([w.size/ (1+variation(w)) for w in weights])
                
#%%
fig, (ax1, ax2) = plt.subplots(2, sharex=True)
fig.set_size_inches((style.size.m[0], style.size.m[1]*1.5))

### epsilon
ax1.scatter(sp.arange(epsilons.size) + 1, epsilons,
            color=sns.utils.get_color_cycle()[0], clip_on=False,
            label=style.name.epsilon)
style.middle_ticks_minor(ax1)
ax1.set_ylim(0, epsilons.max())
ax1.set_yticks([0, epsilons.max()])
#ax1.legend(loc="center left", bbox_to_anchor=(1, .5))
ax1.set_ylabel(style.name.epsilon, labelpad=style.labelpad.reduced)

### nr particles
ax2.scatter(sp.arange(nr_particles.size) + 1, nr_particles,
           color=sns.utils.get_color_cycle()[1], label=style.name.nr_particles)
ax2.scatter(sp.arange(nr_particles.size) + 1, effective_samples_size,
           color=sns.utils.get_color_cycle()[2],
           label=style.name.effective_samples_size)

ax2.set_xlabel(style.name.nr_populations)
ax2.set_ylabel(style.name.n, labelpad=style.labelpad.reduced)
ax2.legend(loc="center left", bbox_to_anchor=(1, .5))
style.middle_ticks_minor(ax2)

sns.despine(fig)
fig.save(sm.output[0])
