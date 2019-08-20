import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_data(time, obs_data, sim_data, plt_title=""):

    # check if the data types are pandas dataframe
    if isinstance(obs_data, pd.DataFrame)\
            and isinstance(sim_data, pd.DataFrame):
        obs_value = obs_data.values.item()
        sim_value = sim_data.values.item()
        plt.scatter(time, sim_value,
                    color="C0", label='Simulation')
        plt.scatter(time, obs_value,
                    color="C1", label='Data')
        plt.xlabel('Time $t$')
        plt.ylabel('Measurement $Y$')
        plt.title('Observed vs. simulated data')
        plt.legend()
        # check if the data types are 2d array
    elif len(obs_data) == 2 and len(sim_data) == 2 \
            and isinstance(sim_data, list)\
            and isinstance(obs_data, list):
        plt.scatter(sim_data[0], sim_data[1],
                    color="C0", label='Simulation')
        plt.scatter(obs_data[0], obs_data[1],
                    color="C1", label='Data')
        plt.xlabel('Time $t$')
        plt.ylabel('Measurement $Y$')
        plt.title('Observed vs. simulated data')
        plt.legend()
    # check if the data types are numpy array
    elif isinstance(obs_data, np.ndarray) \
            and isinstance(sim_data, np.ndarray):
        plt.scatter(time, sim_data,
                    color="C0", label='Simulation')
        plt.scatter(time, obs_data,
                    color="C1", label='Data')
        plt.xlabel('Time $t$')
        plt.ylabel('Measurement $Y$')
        plt.title('Observed vs. simulated data')
        plt.legend()
    else:
        raise SyntaxError('The selected data type is '
                          'not yet supported. Try to use '
                          'Pandas.Dataframe, numpy.array, or 2d arrays')
    plt.show()
