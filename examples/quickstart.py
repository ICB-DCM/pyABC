import os
import tempfile

import scipy.stats as st

from pyabc import (ABCSMC, RV, Distribution,
                   PercentileDistanceFunction, SimpleModel,
                   ConstantPopulationStrategy)

# Define a gaussian model
sigma = .5

def model(args):
    return {"y": st.norm(args['x'], sigma).rvs()}


# We define two models, but they are identical so far
models = [model, model]
models = list(map(SimpleModel, models))


# However, our models' priors are not the same. Their mean differs.
mu_x_1, mu_x_2 = 0, 1
parameter_priors = [
    Distribution(x=RV("norm", mu_x_1, sigma)),
    Distribution(x=RV("norm", mu_x_2, sigma))
]


# We plug all the ABC setup together
population_strategy = ConstantPopulationStrategy(100, 2)
abc = ABCSMC(models, parameter_priors,
             PercentileDistanceFunction(measures_to_use=["y"]),
             population_strategy)

# Finally we add meta data such as model names
# and define where to store the results
db_path = (os.path.expanduser("sqlite:///" +
           os.path.join(tempfile.gettempdir(), "test.db")))
# y_observed is the important piece here: our actual observation.
y_observed = 1
abc.set_data({"y": y_observed}, db_path)

# We run the ABC with 3 populations max
minimum_epsilon = .05
history = abc.run(minimum_epsilon)

# Evaluate the model probabililties
mp = history.get_model_probabilities(history.max_t)

print(mp)
