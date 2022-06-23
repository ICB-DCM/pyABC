"""
Visualization
=============

Visualize results of ABCSMC runs.
"""

from .credible import plot_credible_intervals, plot_credible_intervals_for_time
from .data import plot_data_callback, plot_data_default
from .distance import plot_distance_weights
from .effective_sample_size import plot_effective_sample_sizes
from .epsilon import plot_epsilons
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
    plot_kde_2d,
    plot_kde_2d_highlevel,
    plot_kde_matrix,
    plot_kde_matrix_highlevel,
)
from .model_probabilities import plot_model_probabilities
from .sample import (
    plot_acceptance_rates_trajectory,
    plot_lookahead_acceptance_rates,
    plot_lookahead_evaluations,
    plot_lookahead_final_acceptance_fractions,
    plot_sample_numbers,
    plot_sample_numbers_trajectory,
    plot_total_sample_numbers,
)
from .sankey import plot_sensitivity_sankey
from .walltime import (
    plot_eps_walltime,
    plot_eps_walltime_lowlevel,
    plot_total_walltime,
    plot_walltime,
    plot_walltime_lowlevel,
)
