from pyabc import (ABCSMC, Distribution, RV,
                   PNormDistance, AdaptivePNormDistance)
from pyabc.visualization import plot_kde_1d
import scipy
import tempfile
import os
import matplotlib.pyplot as pyplot


def model(p):
    return {'ss1': p['theta'] + 1 + 0.1*scipy.randn(),
            'ss2': 2 + 10*scipy.randn()}


# ss1 is informative, ss2 is uninformative about theta
prior = Distribution(theta=RV('uniform', 0, 10))
distance = AdaptivePNormDistance(p=2,
                                 use_all_w=True, adaptive=True,
                                 scale_type=
                                 AdaptivePNormDistance.SCALE_TYPE_MAD)
distance = PNormDistance(p=2, w=None, use_all_w=False)
abc = ABCSMC(model, prior, distance)
db_path = ("sqlite:///" + os.path.join(tempfile.gettempdir(), "test.db"))
observation1 = 4
observation2 = 2

# desirable: optimal theta=3
abc.new(db_path, {'ss1': observation1, 'ss2': observation2})

# run
history = abc.run(minimum_epsilon=.1, max_nr_populations=8)

# df, w = history.get_distribution(0,history.max_t)
# df["CDF"] = w
# for parameter in [col for col in df if col != "CDF"]:
#     plot_df = df[["CDF", parameter]].sort_values(parameter)
#     plot_df_cumsum = plot_df.cumsum()
#     plot_df_cumsum[parameter] = plot_df[parameter]

# output
history.get_weighted_sum_stats(None)
fig, ax = pyplot.subplots()
for t in range(history.max_t + 1):
    df, w = history.get_distribution(m=0, t=t)
    plot_kde_1d(df, w,
                xmin=0, xmax=10,
                x='theta', ax=ax,
                label="PDF t={}".format(t))
ax.axvline(observation1-1, color="k", linestyle="dashed")
ax.legend()
pyplot.show()

print("done test1")