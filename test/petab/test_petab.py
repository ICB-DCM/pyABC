import amici.petab_import
import petab
import pyabc.petab

import git
import os
import numpy as np


def test_import():
    # download archive
    benchmark_dir = "doc/examples/tmp/benchmark-models-petab"
    if not os.path.exists(benchmark_dir):
        git.Repo.clone_from(
            "https://github.com/benchmarking-initiative"
            "/benchmark-models-petab.git",
            benchmark_dir, depth=1)
    g = git.Git(benchmark_dir)

    # update repo if online
    try:
        g.pull()
    except git.exc.GitCommandError:
        pass

    # create problem
    petab_problem = petab.Problem.from_yaml(os.path.join(
        benchmark_dir, "Benchmark-Models",
        "Boehm_JProteomeRes2014", "Boehm_JProteomeRes2014.yaml"))

    # compile amici
    model = amici.petab_import.import_petab_problem(petab_problem)
    solver = model.getSolver()

    # import to pyabc
    importer = pyabc.petab.AmiciPetabImporter(petab_problem, model, solver)

    # extract required objects
    prior = importer.create_prior()
    model = importer.create_model()
    kernel = importer.create_kernel()

    # call model
    assert np.isclose(
        model(petab_problem.x_nominal_free_scaled)['llh'], -138.221996)

    # mini analysis
    temperature = pyabc.Temperature(
        enforce_exact_final_temperature=False,
        schemes=[pyabc.AcceptanceRateScheme()])
    acceptor = pyabc.StochasticAcceptor()

    abc = pyabc.ABCSMC(model, prior, kernel, eps=temperature,
                       acceptor=acceptor, population_size=10)
    abc.new(pyabc.storage.create_sqlite_db_id(), None)
    abc.run(max_nr_populations=1)
