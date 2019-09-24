import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import List, Union


logger = logging.getLogger("Data_plot")


def plot_data(obs_data: dict,
              sim_data: dict,
              keys: Union[List[str], str] = None):
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
    key: Union[List[str], str], optional
        Specific summary statistic keys to be used. If None,
        then all entries are used.

    Returns
    -------

    arr_ax: Axes of the generated plot.
    """
    # check if user specified a specific key to be printed
    if keys is None:
        keys = list(obs_data.keys())
    if not isinstance(keys, list):
        keys = [keys]
    obs_data = {key: obs_data[key] for key in keys}
    sim_data = {key: sim_data[key] for key in keys}

    # get number of rows and columns
    ndata = len(obs_data)
    ncols = int(np.ceil(np.sqrt(ndata)))
    nrows = ncols
    while ncols * (nrows - 1) >= ndata:
        nrows -= 1

    # initialize figure
    fig, arr_ax = plt.subplots(nrows, ncols)

    # iterate over keys
    for plot_index, ((obs_key, obs), (_, sim)) \
            in enumerate(zip(obs_data.items(), sim_data.items())):
        if nrows == ncols == 1:
            ax = arr_ax
        else:
            ax = arr_ax.flatten()[plot_index]

        # data frame
        if isinstance(obs, pd.DataFrame):
            if len(obs.columns) == 1:
                # 1d: plot
                ax.plot(sim.values.flatten(), '-x', label="Simulation")
                ax.plot(obs.values.flatten(), '-x', label="Data")
                ax.set_xlabel("Index")
                ax.set_ylabel(obs.columns[0])
            else:
                # nd: scatter
                for j, key in enumerate(obs.columns):
                    ax.scatter(obs[key].values, sim[key].values, label=key)
                ax.set_xlabel("Data")
                ax.set_ylabel("Simulation")
        elif isinstance(obs, np.ndarray) and obs.ndim == 1:
            # 1d: plot
            obs_value = obs
            sim_value = sim
            ax.plot(sim_value, '-x', color="C0", label='Simulation')
            ax.plot(obs_value, '-x', color="C1", label='Data')
            ax.set_xlabel("Index")
            ax.set_ylabel(str(obs_key))
        elif isinstance(obs, np.ndarray):
            # nd: scatter
            for j, (obs_val, sim_val) in enumerate(zip(obs, sim)):
                ax.scatter(obs_val, sim_val, label=f"Coordinate {j}")
            ax.set_xlabel("Data")
            ax.set_ylabel("Simulation")
        else:
            logger.info(f"Data type {type(obs)} for key {obs_key} is "
                        f"not supported.")
            # remove not needed axis
            ax.axis('off')

        # finalize axes
        ax.set_title(str(obs_key))
        ax.legend()

    # remove not needed axes
    for plot_index in range(ndata, ncols * nrows):
        ax = arr_ax.flatten()[plot_index]
        ax.axis('off')

    # finalize plot
    fig.tight_layout()

    return arr_ax
