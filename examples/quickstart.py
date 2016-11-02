from pyabc import ABCSMC, RV, Distribution, Kernel, ModelPerturbationKernel, PercentileDistanceFunction, MedianEpsilon
import scipy.stats as st



# Define a gaussian model
sigma = .5


# "y" is the result key
def model(args):
    return {"y": st.norm(args['x'], sigma).rvs()}


# We define two models, but they are identical so far
models = [model, model]

# The prior over the model classes is uniform
model_prior = RV("randint", 0, 2)

# However, our models' priors are not the same. Their mean differs.
mu_x_1, mu_x_2 = 0, 1
parameter_given_model_prior_distribution = [
    Distribution(x=RV("norm", mu_x_1, sigma)),
    Distribution(x=RV("norm", mu_x_2, sigma))
]

# Particles are perturbed in a Gaussian fashion
parameter_perturbation_kernels = [
    lambda t, stat: Kernel(stat['cov']) for _ in range(2)
    ]

# We plug all the ABC setup together.
# We use "y" in the distance function as this
# was the result key defined for the model
nr_particles = 400
abc = ABCSMC(models, model_prior,
             ModelPerturbationKernel(2, probability_to_stay=.7),
             parameter_given_model_prior_distribution,
             parameter_perturbation_kernels,
             PercentileDistanceFunction(measures_to_use=["y"]),
             MedianEpsilon(.2), nr_particles,
             max_nr_allowed_sample_attempts_per_particle=2000)

# Finally we add meta data such as model
# names and define where to store the results
model_names = ["m1", "m2"]
options = {'db_path': "sqlite:////tmp/abc.db"}
# y_observed is the important piece here: our actual observation.
y_observed = 1
abc.set_data({"y": y_observed}, 0, {}, options, model_names)

# We run the ABC with 3 populations max
minimum_epsilon = .05
nr_populations = 3
nr_samples_per_particles = [1] * nr_populations
history = abc.run(nr_samples_per_particles, minimum_epsilon)

# Evaluate the model probabililties
p1, p2 = history.get_model_probabilities(-1)

# Model 2 should have a higher probability.
print(p1, p2)