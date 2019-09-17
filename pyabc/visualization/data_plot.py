import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging


logger = logging.getLogger("Data_plot")


def plot_data(obs_data: dict, sim_data: dict, key=None):
    """
    Plot summary statistic data.

    Parameters
    ----------

    obs_data: dict
        A dictionary for the summary statistic of the observed data,
        where keys represent the summary statistic name and values represent
        the data itself. The values can be represented as pandas dataframe,
        1d numpy array, or 2d numpy array.
    sim_data: dict
        A dictionary for the summary statistic of the simulated data,
        where keys represent the summary statistic name and values represent
        the data itself. The values can be represented as pandas dataframe,
        1d numpy array, or 2d numpy array.
    key: Union[str, int], optional
        A specified specific summary statistic data, if None,
        then all summary statistics values are used.

    Returns
    -------

    ax: Axis of the generated plot.
    """
    fig = plt.figure()
    plot_index = 1
    # check if user specify a specific key to be printed
    if key is not None:
        obs_data = {key: obs_data[key]}
        sim_data = {key: sim_data[key]}
    if len(obs_data) <= 3:
        plot_row_size = 1
        plot_col_size = len(obs_data)
    elif len(obs_data) <= 6:
        plot_row_size = 2
        plot_col_size = 2
    elif len(obs_data) <= 9:
        plot_row_size = 3
        plot_col_size = 3
    elif len(obs_data) <= 16:
        plot_row_size = 4
        plot_col_size = 4
    else:
        logger.debug("Data length should be equal or less than 16."
                     " Found = {}".format(len(obs_data)))
        return
    # check if the data types are pandas dataframe
    for (obs_key, obs), (_, sim) \
            in zip(obs_data.items(), sim_data.items()):
        plt.subplot(plot_row_size, plot_col_size, plot_index)
        plot_index = plot_index + 1
        if isinstance(obs, pd.DataFrame)\
                and isinstance(sim, pd.DataFrame):
            obs_value = obs.values.item()
            sim_value = sim.values.item()
            plt.plot(sim_value,
                     color="C0", label='Simulation')
            plt.plot(obs_value,
                     color="C1", label='Data')
        # check if the data types are 2d array
        elif isinstance(obs, np.ndarray) \
                and isinstance(sim, np.ndarray) \
                and (obs.ndim == 2) and (sim.ndim == 2):
            obs_value = obs
            sim_value = sim
            plt.scatter(sim_value[0],
                        sim_value[1], color="C0", label='Simulation')
            plt.scatter(obs_value[0],
                        obs_value[1], color="C1", label='Data')
        # check if the data types are 1d numpy array
        elif isinstance(obs, np.ndarray) \
                and isinstance(sim, np.ndarray):
            obs_value = obs
            sim_value = sim
            plt.plot(sim_value,
                     color="C0", label='Simulation')
            plt.plot(obs_value,
                     color="C1", label='Data')
        else:
            logger.debug('The selected data type is '
                         'not yet supported. Try to use '
                         'Pandas.Dataframe, 1d numpy.array, '
                         'or 2d numpy.array. Found = {}'.format(type(obs)))
            return
        plt.xlabel('Time $t$')
        plt.ylabel('Measurement $Y$')
        plt.title(obs_key)
        plt.legend()
        fig.suptitle("Observed vs. simulated data", fontsize=16)
