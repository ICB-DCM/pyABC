import scipy.stats as st
from pyabc import (ABCSMC, RV, ModelPerturbationKernel, Distribution,
                   MedianEpsilon,
                   PercentileDistanceFunction, SimpleModel,
                   MultivariateNormalTransition, ConstantPopulationStrategy)
from parallel import MulticoreSampler
import tempfile
import os


# Define a gaussian model
sigma = .5

def model(args):
    return {"y": st.norm(args['x'], sigma).rvs()}


# We define two models, but they are identical so far
models = [model, model]
models = list(map(SimpleModel, models))

# The prior over the model classes is uniform
model_prior = RV("randint", 0, 2)

# However, our models' priors are not the same. Their mean differs.
mu_x_1, mu_x_2 = 0, 1
parameter_given_model_prior_distribution = [Distribution(x=RV("norm", mu_x_1, sigma)),
                                            Distribution(x=RV("norm", mu_x_2, sigma))]

# Particles are perturbed in a Gaussian fashion
parameter_perturbation_kernels = [MultivariateNormalTransition() for _ in range(2)]

# We plug all the ABC setup together
nr_populations = 3
population_size = ConstantPopulationStrategy(100, 2)
abc = ABCSMC(models, model_prior, ModelPerturbationKernel(2, probability_to_stay=.7),
             parameter_given_model_prior_distribution, parameter_perturbation_kernels,
             PercentileDistanceFunction(measures_to_use=["y"]), MedianEpsilon(.2),
             population_size,
             sampler=MulticoreSampler())

# Finally we add meta data such as model names and define where to store the results
options = {'db_path': os.path.expanduser("sqlite:///" + os.path.join(tempfile.gettempdir(), "test.db"))}
# y_observed is the important piece here: our actual observation.
y_observed = 1
abc.set_data({"y": y_observed}, 0, {}, options)

# We run the ABC with 3 populations max
minimum_epsilon = .05
history = abc.run(minimum_epsilon)

# Evaluate the model probabililties
mp = history.get_model_probabilities(history.max_t)

print(mp)
