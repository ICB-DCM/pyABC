import datetime
import os
import sys
from typing import List, Tuple

import git
import numpy as np
import pandas as pd
import scipy as sp
from sqlalchemy import func

from .. import weighted_statistics
from ..parameters import ValidParticle
from .db_model import with_session, ABCSMC, Population, Model, Particle, Parameter, Sample, SummaryStatistic, Base


class History:
    """
    History for ABCSMC.

    This class records the evolution of the populations and stores the ABCSMC results.

    Parameters
    ----------
    db_path: stt
        SQLAlchemy database identifier.

    nr_models: int
        Nr of models.

    model_names: List[str]
        List of model names.

    min_nr_particles_per_population: int
        Minimum nr of particles per population.

    debug: bool
        Whether to print additional debug output.




    .. warning::

        Most likely you will never have to instantiate the class yourself.
        An instance of this class is returned by the ``ABCSMC.run`` method.
        It can then be used for querying. However, most likely even that won't be
        used as querying is usually done on the stored database usind the abc_loader.

    """
    def __init__(self, db_path: str, nr_models: int, model_names: List[str], debug=False):
        self.store = [] # type: List[List[List[ValidParticle]]]
        self.model_probabilities = []
        self.nr_models = nr_models
        self.nr_simulations = []
        "Only counts the simulations which appear in particles. If a simulation terminated prematurely it is not counted."
        self.db_path = db_path
        self.model_names = list(model_names)
        self._session = None
        self._engine = None
        self.debug = debug

    def weighted_particles_dataframe(self, t, m):
        population = self.store[t][m]
        weights = sp.array([particle.weight for particle in population])
        parameters = pd.DataFrame([dict(particle.parameter) for particle in population])
        return parameters, weights

    @with_session
    def store_initial_data(self, ground_truth_model: int, options,
                           observed_summary_statistics: dict,
                           ground_truth_parameter: dict,
                           distance_function_json_str: str,
                           eps_function_json_str: str):
        """
        Store the initial configuration data.

        Parameters
        ----------
        ground_truth_model: int
            Nr of the ground truth model.

        observed_summary_statistics: dict
            the measured summary statistics

        ground_truth_parameter: dict
            the ground truth parameters

        distance_function_json_str: str
            the distance function represented as json string

        eps_function_json_str: str
            the epsilon represented as json string
        """
        # store ground truth to db
        try:
            git_hash = git.Repo(os.environ['PYTHONPATH']).head.commit.hexsha
        except (git.exc.NoSuchPathError, KeyError) as e:
            git_hash = str(e)
        abc_smc_simulation = ABCSMC(json_parameters=str(options),
                                    start_time=datetime.datetime.now(),
                                    git_hash=git_hash,
                                    distance_function=distance_function_json_str,
                                    epsilon_function=eps_function_json_str)
        population = Population(t=-1, nr_samples=0, epsilon=0)
        abc_smc_simulation.populations.append(population)

        model = Model(m=ground_truth_model, p_model=1, name=self.model_names[ground_truth_model])
        population.models.append(model)

        gt_part = Particle(w=1)
        model.particles.append(gt_part)

        for key, value in ground_truth_parameter.items():
                gt_part.parameters.append(Parameter(name=key, value=value))
        sample = Sample(distance=0)
        gt_part.samples = [sample]
        sample.summary_statistics = [SummaryStatistic(name=key, value=value)
                                     for key, value in observed_summary_statistics.items()]
        self._session.add(abc_smc_simulation)
        self._session.commit()
        self.id = abc_smc_simulation.id
        if self.debug:
            print("Hist start:", abc_smc_simulation)

    @property
    @with_session
    def total_nr_simulations(self):
        "Total nr of simulations/samples."
        nr_sim = self._session.query(func.sum(Population.nr_samples)).join(ABCSMC).filter(ABCSMC.id == self.id).one()[0]
        return nr_sim

    def _make_session(self):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        engine = create_engine(self.db_path, connect_args={'timeout': 120})
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        session = Session()
        self._session = session
        self._engine = engine
        return session

    def _close_session(self):
        self._session.close()
        self._engine.dispose()
        self._session = None
        self._engine = None

    @with_session
    def done(self):
        "Close database sessions and store end time of population."
        abc_smc_simulation = (self._session.query(ABCSMC)
                              .filter(ABCSMC.id == self.id)
                              .one())
        abc_smc_simulation.end_time = datetime.datetime.now()
        self._session.commit()
        if self.debug:
            print("Hist done:", abc_smc_simulation)

    @with_session
    def _save_to_population_db(self, t, current_epsilon, nr_simulations):
        # sqlalchemy experimental stuff and highly inefficient implementation here
        # but that is ok for testing purposes for the moment
        # prepare
        abc_smc_simulation = (self._session.query(ABCSMC)
                              .filter(ABCSMC.id == self.id)
                              .one())

        # store the population
        population = Population(t=t, nr_samples=nr_simulations, epsilon=current_epsilon)
        abc_smc_simulation.populations.append(population)
        for m in range(len(self.store[t])):
            model = Model(m=m, p_model=self.model_probabilities[t][m], name=self.model_names[m])
            population.models.append(model)
            for store_item in self.store[t][m]:
                weight = store_item['weight']
                distance_list = store_item['distance_list']
                parameter = store_item['parameter']
                summary_statistics_list = store_item['summary_statistics_list']
                particle = Particle(w=weight)
                model.particles.append(particle)
                for key, value in parameter.items():
                    if isinstance(value, dict):
                        for key_dict, value_dict in value.items():
                            particle.parameters.append(Parameter(name=key + "_" + key_dict, value=value_dict))
                    else:
                        particle.parameters.append(Parameter(name=key, value=value))
                for distance, summ_stat in zip(distance_list, summary_statistics_list):
                    sample = Sample(distance=distance)
                    particle.samples.append(sample)
                    for name, value in summ_stat.items():
                        sample.summary_statistics.append(SummaryStatistic(name=name, value=value))

        self._session.commit()
        if self.debug:
            print("Hist append:", population)

    def _append(self, t, valid_particle: ValidParticle):
        self.store[t][valid_particle.m].append(valid_particle)  # summary statistics are only recorded for analysis purposes

    def _extend_store(self, t):
        while len(self.store) < t+1:
            self.store.append([[] for _ in range(self.nr_models)])
            self.model_probabilities.append(None)
            self.nr_simulations.append(0)

    def append_population(self, t: int, current_epsilon: float, particle_population: list, nr_simulations: int):
        """
        Append population to database.

        Parameters
        ----------
        t: int
            Population number.

        current_epsilon: float
            Current epsilon value.

        particle_population: list
            List of sampled particles

        Returns
        -------

        enough_particles: bool
            Whether enough particles were found in the population.

        """
        particle_population = list(particle_population)
        self._extend_store(t)
        for particle in particle_population:
            if particle:  # particle might be none or empty if no particle was found within the allowed nr of sample attempts
                self._append(t, particle)
            else:
                print("ABC History warning: Empty particle.", file=sys.stderr)
        self._normalize(t)
        self._save_to_population_db(t, current_epsilon, nr_simulations)
        self.nr_simulations[t] = nr_simulations



    def _normalize(self, t):
        """
          * Normalize particle weights according to nr of particles in a model
          * Caclculate marginal model probabilities
        """
        population = self.store[t]

        model_total_weights = [sum(particle.weight for particle in model)
                               for model in self.store[t]]

        # normalize within each model
        for model_total_weight, model in zip(model_total_weights, population):
            for particle in model:
                particle.weight /= model_total_weight

        population_total_weight = sum(model_total_weights)
        model_probabilities = [w / population_total_weight for w in model_total_weights]

        # only update model probabilities if not previously calculated
        # otherwise all probabilities will be equal after a second
        # normalization
        if self.model_probabilities[t] is None:
            self.model_probabilities[t] = model_probabilities

    def get_distribution(self, t: int, m: int, parameter: str) -> (np.ndarray, np.ndarray):
        """
        Returns parameter values and weights.

        Parameters
        ----------


        t: int
            Population number

        m: int
            Model number

        parameter: str

        Returns
        -------
        (points, weights): Tuple[np.ndarray]
            The points and their weights.
        """
        points = self.store[t][m]
        if len(points) > 0:
            par, w = zip(*[(p['parameter'][parameter], p['weight']) for p in points])
            return sp.asarray(par), sp.asarray(w)
        else:
            return sp.asarray([]), sp.asarray([])

    def get_results_distribution(self, m: int, parameter: str) -> Tuple[np.ndarray]:
        """
        Returns parameter values and weights of the last population.

        Parameters
        ----------

        m: int
            Model number

        parameter: str
            Parameter name.

        Returns
        -------

        results: Tuple[np.ndarray]
            results = (points, weights) with the points and the weights of the last population.
        """
        return self.get_distribution(-1, m, parameter)

    @with_session
    def get_model_probabilities(self, t=None) -> np.ndarray:
        """
        Model probabilities.

        Parameters
        ----------
        t: int or None
            Population. Defaults to None, i.e. the last population.

        Returns
        -------
        probabilities: np.ndarray
            Model probabilities
        """
        if t is None:
            t = (self._session
                 .query(func.max(Population.t))
                 .filter(Population.abc_smc_id == self.id)
                 .one()
                 [0])

        p_models = (self._session
                    .query(Model.p_model)
                    .join(Population)
                    .join(ABCSMC)
                    .filter(ABCSMC.id == self.id)
                    .filter(Population.t == t)
                    .order_by(Model.m)
                    .all())

        p_models_arr = sp.array([r[0] for r in p_models], dtype=float)
        return p_models_arr

    def nr_of_models_alive(self, t=-1) -> int:
        """
        Number of models still alive.

        Parameters
        ----------
        t: int
            Population number

        Returns
        -------
        nr_alive: int
            Number of models still alive.
        """
        model_probs = self.get_model_probabilities(t)
        return int((model_probs > 0).sum())

    def get_complete_population_median(self, t: int) -> float:
        """
        Median of a population's distances to the measured sample

        Parameters
        ----------
        t: int
            Population number

        Returns
        -------

        median: float
            The median of the distances.
        """
        models = self.store[t]
        model_probabilities = self.model_probabilities[t]
        distances = sp.asarray([dist
                                for model in models
                                for point in model
                                for dist in point['distance_list']])
        weights = sp.asarray([point['weight'] * model_weight / len(point['distance_list'])
                              for model, model_weight in zip(models, model_probabilities)
                              for point in model
                              for _ in point['distance_list']])
        median = weighted_statistics.weighted_median(distances, weights)
        return median

    @property
    def t(self):
        """
        Current population.
        """
        return len(self.store)