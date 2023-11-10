"""
.. _api_visualization:

Visualization
=============

Visualize results of ABCSMC runs.
"""

from .contour import (
    plot_contour_2d,
    plot_contour_2d_lowlevel,
    plot_contour_matrix,
    plot_contour_matrix_lowlevel,
)
from .credible import (
    plot_credible_intervals,
    plot_credible_intervals_for_time,
    plot_credible_intervals_plotly,
)
from .data import plot_data_callback, plot_data_default
from .distance import plot_distance_weights
from .effective_sample_size import (
    plot_effective_sample_sizes,
    plot_effective_sample_sizes_plotly,
)
from .epsilon import plot_epsilons, plot_epsilons_plotly
from .histogram import (
    plot_histogram_1d,
    plot_histogram_1d_lowlevel,
    plot_histogram_2d,
    plot_histogram_2d_lowlevel,
    plot_histogram_matrix,
    plot_histogram_matrix_lowlevel,
)
from .kde import (
    plot_kde_1d,
    plot_kde_1d_highlevel,
    plot_kde_1d_highlevel_plotly,
    plot_kde_1d_plotly,
    plot_kde_2d,
    plot_kde_2d_highlevel,
    plot_kde_2d_highlevel_plotly,
    plot_kde_2d_plotly,
    plot_kde_matrix,
    plot_kde_matrix_highlevel,
    plot_kde_matrix_highlevel_plotly,
    plot_kde_matrix_plotly,
)
from .model_probabilities import (
    plot_model_probabilities,
    plot_model_probabilities_plotly,
)
from .sample import (
    plot_acceptance_rates_trajectory,
    plot_acceptance_rates_trajectory_plotly,
    plot_lookahead_acceptance_rates,
    plot_lookahead_evaluations,
    plot_lookahead_final_acceptance_fractions,
    plot_sample_numbers,
    plot_sample_numbers_plotly,
    plot_sample_numbers_trajectory,
    plot_sample_numbers_trajectory_plotly,
    plot_total_sample_numbers,
    plot_total_sample_numbers_plotly,
)
from .sankey import plot_sensitivity_sankey
from .walltime import (
    plot_eps_walltime,
    plot_eps_walltime_lowlevel,
    plot_eps_walltime_lowlevel_plotly,
    plot_eps_walltime_plotly,
    plot_total_walltime,
    plot_total_walltime_plotly,
    plot_walltime,
    plot_walltime_lowlevel,
    plot_walltime_lowlevel_plotly,
    plot_walltime_plotly,
)
