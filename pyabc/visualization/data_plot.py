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
    # check if user specified a specific key to be printed
    if key is not None:
        obs_data = {key: obs_data[key]}
        sim_data = {key: sim_data[key]}

    # get number of rows and columns
    if len(obs_data) <= 3:
        nrows = 1
        ncols = len(obs_data)
    elif len(obs_data) <= 6:
        nrows = 2
        ncols = 2
    elif len(obs_data) <= 9:
        nrows = 3
        ncols = 3
    elif len(obs_data) <= 16:
        nrows = 4
        ncols = 4
    else:
        logger.error("Data length should be equal or less than 16. "
                     "Found = {}.".format(len(obs_data)))
        return

    # initialize figure
    fig, arr_ax = plt.subplots(nrows, ncols)

    # iterate over keys
    for plot_index, ((obs_key, obs), (_, sim)) \
            in enumerate(zip(obs_data.items(), sim_data.items())):
        ax = arr_ax.flatten()[plot_index]

        # data frame
        if isinstance(obs, pd.DataFrame):
            obs_value = obs.values.item()
            sim_value = sim.values.item()
            ax.plot(sim_value,  '-x', color="C0", label='Simulation')
            ax.plot(obs_value,  '-x', color="C1", label='Data')
        # 2d array
        elif isinstance(obs, np.ndarray) and (obs.ndim == 2):
            obs_value = obs
            sim_value = sim
            ax.scatter(sim_value[0], sim_value[1],
                       color="C0", label='Simulation')
            ax.scatter(obs_value[0], obs_value[1],
                       color="C1", label='Data')
        # 1d array
        elif isinstance(obs, np.ndarray):
            obs_value = obs
            sim_value = sim
            ax.plot(sim_value, '-x', color="C0", label='Simulation')
            ax.plot(obs_value, '-x', color="C1", label='Data')
        else:
            logger.info("The selected data type is "
                        "not yet supported. Try to use "
                        "Pandas.Dataframe, 1d numpy.array, "
                        "or 2d numpy.array. Found = {}.".format(type(obs)))

        # finalize axes
        ax.set_title(obs_key)
        ax.legend()

    # finalize plot
    fig.tight_layout()

    return arr_ax
