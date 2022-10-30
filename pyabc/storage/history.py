import datetime
import json
import logging
import os
import tempfile
from functools import wraps
from typing import List, Union

import numpy as np
import pandas as pd
import sqlalchemy as sa
from sqlalchemy.orm import subqueryload

from ..parameters import Parameter as PyParameter
from ..population import Particle as PyParticle
from ..population import Population as PyPopulation
from .db_model import (
    ABCSMC,
    Base,
    Model,
    Parameter,
    Particle,
    Population,
    Sample,
    SummaryStatistic,
    Version,
)
from .version import __db_version__

logger = logging.getLogger("ABC.History")

SQLITE_STR = "sqlite:///"


def with_session(f):
    @wraps(f)
    def f_wrapper(self: "History", *args, **kwargs):
        logger.debug(f"Database access through {f.__name__}")
        no_session = self._session is None and self._engine is None
        if no_session:
            self._make_session()
        res = f(self, *args, **kwargs)
        if no_session:
            self._close_session()
        return res

    return f_wrapper


def internal_docstring_warning(f):
    first_line = f.__doc__.split("\n")[1]
    indent_level = len(first_line) - len(first_line.lstrip())
    indent = " " * indent_level
    warning = (
        "\n\n"
        + indent
        + "**Note.** This function is called by the :class:`pyabc.ABCSMC` "
        "class internally. "
        "You should most likely not find it necessary to call "
        "this method under normal circumstances."
    )

    f.__doc__ += warning
    return f


def git_hash():
    try:
        import git
    except ImportError:
        return "Install pyABC's optional git dependency for git support"
    try:
        hash_ = git.Repo(os.getcwd()).head.commit.hexsha
    except (
        git.exc.NoSuchPathError,
        KeyError,
        ValueError,
        git.exc.InvalidGitRepositoryError,
    ) as e:
        hash_ = str(e)
    return hash_


def create_sqlite_db_id(dir_: str = None, file_: str = "pyabc_test.db"):
    """
    Convenience function to create an sqlite database identifier which
    can be understood by sqlalchemy.

    Parameters
    ----------
    dir_:
        The base folder name. Optional, defaults to the system's
        temporary directory, i.e. "/tmp/" on Linux. While this makes
        sense for testing purposes, for productive use a non-temporary
        location should be used.
    file_:
        The database file name. Optional, defaults to "pyabc_test.db".
    """
    if dir_ is None:
        dir_ = tempfile.gettempdir()
    return "sqlite:///" + os.path.join(dir_, file_)


def database_exists(db: str) -> bool:
    """Does the database file exist already?"""
    return (
        db != "sqlite://"
        and os.path.exists(db[len(SQLITE_STR) :])
        and os.path.getsize(db[len(SQLITE_STR) :]) > 0
    )


class History:
    """
    History for ABCSMC.

    This class records the evolution of the populations
    and stores the ABCSMC results.

    Attributes
    ----------
    db: str
        SQLalchemy database identifier. For a relative path use the
        template "sqlite:///file.db", for an absolute path
        "sqlite:////path/to/file.db", and for an in-memory database
        "sqlite://".
    stores_sum_stats: bool, optional (default = True)
        Whether to store summary statistics to the database. Note: this
        is True by default, and should be set to False only for testing
        purposes (i.e. to speed up the writing to the file system),
        as it can not be guaranteed that all methods of pyabc work
        correctly if the summary statistics are not stored.
    id: int
        The id of the ABCSMC analysis that is currently in use.
        If there are analyses in the database already, this defaults
        to the latest id. Manually set if another run is wanted.
    """

    # session timeout
    DB_TIMEOUT = 120
    # time before first population time
    PRE_TIME = -1

    def __init__(
        self,
        db: str,
        stores_sum_stats: bool = True,
        _id: int = None,
        create: bool = True,
    ):
        """Initialize history object.

        Parameters
        ----------
        create:
            If False, an error is thrown if the database does not exist.
        """
        db_exists = database_exists(db)
        if not create and not db_exists:
            raise ValueError(f"Database file {db} does not exist.")

        self.db = db
        self.stores_sum_stats = stores_sum_stats

        # to be filled using the session wrappers
        self._session = None
        self._engine = None

        # set version if new file
        if not db_exists:
            self._set_version()

        # check that the database version is up-to-date
        self._check_version()

        # find id in database
        if _id is None:
            _id = self._find_latest_id()
        self._id = _id

    def db_file(self):
        f = self.db.split(":")[-1][3:]
        return f

    @property
    def in_memory(self):
        return (
            self._engine is not None and str(self._engine.url) == "sqlite://"
        )

    @property
    def db_size(self) -> Union[int, str]:
        """
        Size of the database.

        Returns
        -------
        db_size: int, str
            Size of the SQLite database in MB.
            Currently this only works for SQLite databases.

            Returns an error string if the DB size cannot be calculated.
        """
        try:
            return os.path.getsize(self.db_file()) / 10**6
        except FileNotFoundError:
            return "Cannot calculate size"

    @with_session
    def all_runs(self):
        """
        Get all ABCSMC runs which are stored in the database.
        """
        runs = self._session.query(ABCSMC).all()
        return runs

    @with_session
    def _check_version(self):
        """Check that the database version is up-to-date.

        Raises
        ------
        ValueError if the version is not up-to-date.
        """
        # extract version table
        version = self._session.query(Version)
        if len(version.all()) == 0:
            # the first version has no version table
            version = '0'
        else:
            version = str(version.one().version_num)

        # compare to current version
        if version != __db_version__:
            raise AssertionError(
                f"Database has version {version}, latest format version is "
                f"{__db_version__}. Thus, not all queries may work correctly. "
                "Consider migrating the database to the latest version via "
                "`abc-migrate`. Check `abc-migrate --help` and the "
                "documentation for further information."
            )

    @with_session
    def _find_latest_id(self):
        """
        If there are analysis objects saved in the database already,
        the id of the last successful run is returned.
        This is because that is usually the run the user will be
        interested in.
        """
        abcs = self._session.query(ABCSMC).all()
        if len(abcs) > 0:
            for abc in reversed(abcs):
                if len(abc.populations):
                    return abc.id
        return None

    @property
    @with_session
    def id(self):
        return self._id

    @id.setter
    @with_session
    def id(self, val):
        """
        Set id to `val`. If `val` is None, self._find_latest_id()
        is employed to try to find one.
        """
        if val is None:
            val = self._find_latest_id()
        elif val not in [obj.id for obj in self._session.query(ABCSMC).all()]:
            raise ValueError(f"Specified id {val} does not exist in database.")
        self._id = val

    @with_session
    def alive_models(self, t: int = None) -> List:
        """
        Get the models which are still alive at time `t`.

        Parameters
        ----------
        t: int, optional (default = self.max_t)
            Population index.

        Returns
        -------
        alive: List
            A list which contains the indices of those
            models which are still alive.

        """
        if t is None:
            t = self.max_t
        else:
            t = int(t)

        alive = (
            self._session.query(Model.m)
            .join(Population)
            .join(ABCSMC)
            .filter(ABCSMC.id == self.id)
            .filter(Population.t == t)
        ).all()

        return sorted(a[0] for a in alive)

    @with_session
    def get_distribution(
        self, m: int = 0, t: int = None
    ) -> (pd.DataFrame, np.ndarray):
        """
        Returns the weighted population sample for model m and timepoint t
        as a tuple.

        Parameters
        ----------
        m: int, optional (default = 0)
            Model index.
        t: int, optional (default = self.max_t)
            Population index.
            If t is not specified, then the last population is returned.

        Returns
        -------
        df, w: pandas.DataFrame, np.ndarray
            * df: a DataFrame of parameters
            * w: are the weights associated with each parameter
        """
        m = int(m)
        if t is None:
            t = self.max_t
        else:
            t = int(t)

        query = (
            self._session.query(
                Particle.id, Parameter.name, Parameter.value, Particle.w
            )
            .filter(Particle.id == Parameter.particle_id)
            .join(Model)
            .join(Population)
            .filter(Model.m == m)
            .filter(Population.t == t)
            .join(ABCSMC)
            .filter(ABCSMC.id == self.id)
        )
        df = pd.read_sql_query(query.statement, self._engine)
        pars = df.pivot(
            index="id", columns="name", values="value"
        ).sort_index()
        w = df[["id", "w"]].drop_duplicates().set_index("id").sort_index()
        w_arr = w.w.values
        if w_arr.size > 0 and not np.isclose(w_arr.sum(), 1):
            raise AssertionError(
                "Weight not close to 1, w.sum()={}".format(w_arr.sum())
            )
        return pars, w_arr

    @with_session
    def model_names(self, t: int = PRE_TIME):
        """
        Get the names of alive models for population `t`.

        Parameters
        ----------
        t: int, optional (default = -1)
            Population index.

        Returns
        -------
        model_names: List[str]
            List of model names.
        """
        res = (
            self._session.query(Model.name)
            .join(Population)
            .join(ABCSMC)
            .filter(ABCSMC.id == self.id)
            .filter(Population.t == t)
            .filter(Model.name.isnot(None))
            .order_by(Model.m)
            .distinct()
            .all()
        )
        return [r[0] for r in res]

    @with_session
    def get_abc(self):
        return self._session.query(ABCSMC).filter(ABCSMC.id == self.id).one()

    @with_session
    def get_all_populations(self):
        """
        Returns a pandas DataFrame with columns

        * `t`: Population number
        * `population_end_time`: The end time of the population
        * `samples`: The number of sample attempts performed
           for a population
        * `epsilon`: The acceptance threshold for the population.

        Returns
        -------
        all_populations: pd.DataFrame
            DataFrame with population info
        """
        query = self._session.query(
            Population.t,
            Population.population_end_time,
            Population.nr_samples,
            Population.epsilon,
        ).filter(Population.abc_smc_id == self.id)
        df = pd.read_sql_query(query.statement, self._engine)
        particles = self.get_nr_particles_per_population()
        particles.index += 1
        df["particles"] = particles
        df = df.rename(columns={"nr_samples": "samples"})
        return df

    @with_session
    @internal_docstring_warning
    def store_initial_data(
        self,
        ground_truth_model: int,
        options: dict,
        observed_summary_statistics: dict,
        ground_truth_parameter: dict,
        model_names: List[str],
        distance_function_json_str: str,
        eps_function_json_str: str,
        population_strategy_json_str: str,
        start_time: datetime.datetime = None,
    ) -> None:
        """
        Store the initial configuration data.

        Parameters
        ----------
        ground_truth_model:
            Index of the ground truth model.
        options:
            Of ABC metadata.
        observed_summary_statistics:
            The measured summary statistics.
        ground_truth_parameter:
            The ground truth parameters.
        model_names:
            A list of model names.
        distance_function_json_str:
            The distance function represented as json string.
        eps_function_json_str:
            The epsilon represented as json string.
        population_strategy_json_str:
            The population strategy represented as json string.
        start_time:
            Start time of the analysis.
        """
        if start_time is None:
            start_time = datetime.datetime.now()

        # create a new ABCSMC analysis object
        abcsmc = ABCSMC(
            json_parameters=str(options),
            start_time=start_time,
            git_hash=git_hash(),
            distance_function=distance_function_json_str,
            epsilon_function=eps_function_json_str,
            population_strategy=population_strategy_json_str,
        )

        # add to session
        self._session.add(abcsmc)
        self._session.commit()

        # set own id
        self.id = abcsmc.id

        # store pre population
        self.store_pre_population(
            ground_truth_model,
            observed_summary_statistics,
            ground_truth_parameter,
            model_names,
        )

        # log
        logger.info(f"Start {abcsmc.start_info()}")

    @with_session
    @internal_docstring_warning
    def store_pre_population(
        self,
        ground_truth_model: int,
        observed_summary_statistics: dict,
        ground_truth_parameter: dict,
        model_names: List[str],
    ):
        """
        Store a dummy pre-population containing some configuration data
        and in particular some ground truth values.

        For the parameters, see store_initial_data.

        """
        # extract analysis object
        abcsmc = self._session.query(ABCSMC).filter(ABCSMC.id == self.id).one()

        # store  ground truth to db

        # create and append dummy population
        population = Population(
            t=History.PRE_TIME, nr_samples=0, epsilon=np.inf
        )
        abcsmc.populations.append(population)

        # add (ground truth or dummy) model
        if ground_truth_model is not None:  # GT model given
            gt_model = Model(
                m=ground_truth_model,
                p_model=1,
                name=model_names[ground_truth_model],
            )
        else:
            gt_model = Model(m=None, p_model=1, name=None)
        population.models.append(gt_model)

        # add a particle
        gt_part = Particle(w=1)
        gt_model.particles.append(gt_part)

        # add ground truth parameter (or {})
        for key, value in ground_truth_parameter.items():
            gt_part.parameters.append(Parameter(name=key, value=value))

        # add a sample
        sample = Sample(distance=0)
        gt_part.samples = [sample]

        # add observed sum stats to sample
        sample.summary_statistics = [
            SummaryStatistic(name=key, value=value)
            for key, value in observed_summary_statistics.items()
        ]

        # add all models not added so far
        for m, name in enumerate(model_names):
            if m != ground_truth_model:
                population.models.append(Model(m=m, name=name, p_model=0))

        # commit changes
        self._session.commit()

    @with_session
    @internal_docstring_warning
    def update_after_calibration(
        self, nr_samples: int, end_time: datetime.datetime
    ):
        """Update after the calibration iteration.
        In particular set time and number of samples.
        Update the number of samples used in iteration `t`.

        Parameters
        ----------
        nr_samples:
            Number of samples reported.
        end_time:
            End time of the calibration iteration.
        """
        # extract population
        population = (
            self._session.query(Population)
            .join(ABCSMC)
            .filter(ABCSMC.id == self.id)
            .filter(Population.t == History.PRE_TIME)
            .one()
        )

        # update samples number
        population.nr_samples = nr_samples
        population.population_end_time = end_time

        # commit changes
        self._session.commit()

    @with_session
    def observed_sum_stat(self):
        """
        Get the observed summary statistics.

        Returns
        -------
        sum_stats_dct: dict
            The observed summary statistics.
        """
        sum_stats = (
            self._session.query(SummaryStatistic)
            .join(Sample)
            .join(Particle)
            .join(Model)
            .join(Population)
            .join(ABCSMC)
            .filter(ABCSMC.id == self.id)
            .filter(Population.t == History.PRE_TIME)
            .filter(Model.p_model == 1)
            .all()
        )
        sum_stats_dct = {ss.name: ss.value for ss in sum_stats}
        return sum_stats_dct

    @property
    @with_session
    def total_nr_simulations(self) -> int:
        """
        Number of sample attempts for the ABC run.

        Returns
        -------
        nr_sim: int
            Total nr of sample attempts for the ABC run.
        """
        nr_sim = (
            self._session.query(sa.func.sum(Population.nr_samples))
            .join(ABCSMC)
            .filter(ABCSMC.id == self.id)
            .one()[0]
        )
        return nr_sim

    def _make_session(self):
        # TODO: check if the session creation and closing is still necessary
        # I think I did this funny construction due to some pickling issues
        # but I'm not quite sure anymore
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        engine = create_engine(
            self.db, connect_args={'timeout': self.DB_TIMEOUT}
        )
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        session = Session()
        self._session = session
        self._engine = engine
        return session

    @with_session
    def _set_version(self):
        """Set database version.

        This should only be called if the database is new, in order not to
        wrongly assign a version to an unversioned (outdated) database.
        """
        # check if version exists already
        if len(self._session.query(Version).all()) > 0:
            raise AssertionError("Cannot set version as already exists.")

        # set version to latest
        self._session.add(Version(version_num=__db_version__))
        self._session.commit()

    def _close_session(self):
        # don't close in memory database
        if self.in_memory:
            return
        # only close connections to permanent databases
        self._session.close()
        self._engine.dispose()
        self._session = None
        self._engine = None

    def __getstate__(self):
        dct = self.__dict__.copy()
        if self.in_memory:
            dct["_engine"] = None
            dct["_session"] = None
        return dct

    @with_session
    @internal_docstring_warning
    def done(self, end_time: datetime.datetime = None):
        """
        Close database sessions and store end time of the analysis.

        Parameters
        ----------
        end_time:
            End time of the analysis.
        """
        if end_time is None:
            end_time = datetime.datetime.now()

        abcsmc = self._session.query(ABCSMC).filter(ABCSMC.id == self.id).one()
        abcsmc.end_time = end_time
        self._session.commit()
        logger.info(f"Done {abcsmc.end_info()}")

    @with_session
    def _save_to_population_db(
        self,
        t: int,
        current_epsilon: float,
        nr_simulations: int,
        particles_by_model: dict,
        model_probabilities: pd.DataFrame,
        model_names,
    ):
        # sqlalchemy experimental stuff and highly inefficient implementation
        # here but that is ok for testing purposes for the moment

        # extract analysis object
        abcsmc = self._session.query(ABCSMC).filter(ABCSMC.id == self.id).one()

        # store the population
        population = Population(
            t=t, nr_samples=nr_simulations, epsilon=current_epsilon
        )

        abcsmc.populations.append(population)

        # iterate over models
        for m, model_population in particles_by_model.items():
            # create new model
            model = Model(
                m=int(m),
                p_model=float(model_probabilities.loc[m, 'p']),
                name=str(model_names[m]),
            )
            # append model
            population.models.append(model)

            # TODO This normalization is different than in the in-memory
            #  population. It would be cleaner to update the db too.
            total_model_weight = sum(p.weight for p in model_population)

            # iterate over model population of particles
            for py_particle in model_population:
                # a store_item is a Particle
                py_parameter = py_particle.parameter

                # create new particle
                particle = Particle(
                    w=py_particle.weight / total_model_weight,
                    proposal_id=py_particle.proposal_id,
                )
                # append particle to model
                model.particles.append(particle)

                # append parameter dimensions to particle
                for key, value in py_parameter.items():
                    if isinstance(value, dict):
                        # parameter entry is itself a dictionary
                        for key_dict, value_dict in value.items():
                            # append nested dimension to parameter
                            particle.parameters.append(
                                Parameter(
                                    name=key + "_" + key_dict, value=value_dict
                                )
                            )
                    else:
                        # append dimension to parameter
                        particle.parameters.append(
                            Parameter(name=key, value=value)
                        )

                # create new sample from distance
                sample = Sample(distance=py_particle.distance)
                # append to particle
                particle.samples.append(sample)
                # append sum stat dimensions to sample
                if self.stores_sum_stats:
                    for name, value in py_particle.sum_stat.items():
                        if name is None:
                            raise Exception("Summary statistics need names.")
                        sample.summary_statistics.append(
                            SummaryStatistic(name=name, value=value)
                        )

        # commit changes
        self._session.commit()

        # log
        logger.debug("Appended population")

    @internal_docstring_warning
    def append_population(
        self,
        t: int,
        current_epsilon: float,
        population: PyPopulation,
        nr_simulations: int,
        model_names,
    ):
        """
        Append population to database.

        Parameters
        ----------
        t: int
            Population number.
        current_epsilon: float
            Current epsilon value.
        population: Population
            List of sampled particles.
        nr_simulations: int
            The number of model evaluations for this population.
        model_names: list
            The model names.
        """
        particles_by_model = population.get_particles_by_model()
        model_probabilities = population.get_model_probabilities()

        self._save_to_population_db(
            t,
            current_epsilon,
            nr_simulations,
            particles_by_model,
            model_probabilities,
            model_names,
        )

    @with_session
    def get_model_probabilities(
        self, t: Union[int, None] = None
    ) -> pd.DataFrame:
        """
        Model probabilities.

        Parameters
        ----------
        t: int or None (default = None)
            Population index. If None, all populations of indices >= 0 are
            considered.

        Returns
        -------
        probabilities: np.ndarray
            Model probabilities.
        """

        if t is not None:
            t = int(t)

        p_models = (
            self._session.query(Model.p_model, Model.m, Population.t)
            .join(Population)
            .join(ABCSMC)
            .filter(ABCSMC.id == self.id)
            .filter(Population.t == t if t is not None else Population.t >= 0)
            .order_by(Model.m)
            .all()
        )
        # TODO this is a mess
        if t is not None:
            p_models_df = pd.DataFrame(
                [p[:2] for p in p_models], columns=["p", "m"]
            ).set_index("m")
            # TODO the following line is redundant
            # only models with no-zero weight are stored for each population
            p_models_df = p_models_df[p_models_df.p >= 0]
            return p_models_df
        else:
            p_models_df = (
                pd.DataFrame(p_models, columns=["p", "m", "t"])
                .pivot("t", "m", "p")
                .fillna(0)
            )
            return p_models_df

    def nr_of_models_alive(self, t: int = None) -> int:
        """
        Number of models still alive.

        Parameters
        ----------
        t: int, optional (default = self.max_t)
            Population index.

        Returns
        -------
        nr_alive: int >= 0 or None
            Number of models still alive.
            None is for the last population
        """
        if t is None:
            t = self.max_t
        else:
            t = int(t)

        model_probs = self.get_model_probabilities(t)

        return int((model_probs.p > 0).sum())

    @with_session
    def get_weighted_distances(self, t: int = None) -> pd.DataFrame:
        """
        Population's weighted distances to the measured sample.
        These weights do not necessarily sum up to 1.

        Parameters
        ----------
        t: int, optional (default = self.max_t)
            Population index.
            If t is None, the last population is selected.

        Returns
        -------
        df_weighted: pd.DataFrame
            Weighted distances.
            The dataframe has column "w" for the weights
            and column "distance" for the distances.
        """
        if t is None:
            t = self.max_t
        else:
            t = int(t)

        models = (
            self._session.query(Model)
            .join(Population)
            .join(ABCSMC)
            .filter(ABCSMC.id == self.id)
            .filter(Population.t == t)
            .options(
                subqueryload(Model.particles).subqueryload(Particle.samples)
            )
            .all()
        )

        weights = []
        distances = []
        for model in models:
            for particle in model.particles:
                weight = particle.w * model.p_model
                for sample in particle.samples:
                    weights.append(weight)
                    distances.append(sample.distance)

        # query = (self._session.query(Sample.distance, Particle.w, Model.m)
        #         .join(Particle)
        #         .join(Model).join(Population).join(ABCSMC)
        #         .filter(ABCSMC.id == self.id)
        #         .filter(Population.t == t))
        # df = pd.read_sql_query(query.statement, self._engine)
        # model_probabilities = self.get_model_probabilities(t).reset_index()
        # df_weighted = df.merge(model_probabilities)
        # df_weighted["w"] *= df_weighted["p"]

        return pd.DataFrame({'distance': distances, 'w': weights})

    @with_session
    def get_nr_particles_per_population(self) -> pd.Series:
        """
        Get the number of particles per population.

        Returns
        -------
        nr_particles_per_population: pd.DataFrame
            A pandas DataFrame containing the number
            of particles for each population.

        """
        query = (
            self._session.query(Population.t)
            .join(ABCSMC)
            .join(Model)
            .join(Particle)
            .filter(ABCSMC.id == self.id)
        )
        df = pd.read_sql_query(query.statement, self._engine)
        nr_particles_per_population = df.t.value_counts().sort_index()
        return nr_particles_per_population

    @property
    @with_session
    def max_t(self):
        """
        The population number of the last populations.
        This is equivalent to ``n_populations - 1``.
        """
        max_t = (
            self._session.query(sa.func.max(Population.t))
            .join(ABCSMC)
            .filter(ABCSMC.id == self.id)
            .one()[0]
        )
        return max_t

    @property
    def n_populations(self):
        """
        Number of populations stored in the database.
        This is equivalent to ``max_t + 1``.
        """
        return self.max_t + 1

    @with_session
    def get_weighted_sum_stats_for_model(
        self, m: int = 0, t: int = None
    ) -> (np.ndarray, List):
        """
        Summary statistics for model `m`. The weights sum to 1, unless
        there were multiple acceptances per particle.

        Parameters
        ----------
        m: int, optional (default = 0)
            Model index.
        t: int, optional (default = self.max_t)
            Population index.

        Returns
        -------
        w, sum_stats: np.ndarray, list
            * w: the weights associated with the summary statistics
            * sum_stats: list of summary statistics
        """
        m = int(m)
        if t is None:
            t = self.max_t
        else:
            t = int(t)

        particles = (
            self._session.query(Particle)
            .join(Model)
            .join(Population)
            .join(ABCSMC)
            .filter(ABCSMC.id == self.id)
            .filter(Population.t == t)
            .filter(Model.m == m)
            .all()
        )

        results = []
        weights = []
        for particle in particles:
            for sample in particle.samples:
                weights.append(particle.w)
                sum_stats = {}
                for ss in sample.summary_statistics:
                    sum_stats[ss.name] = ss.value
                results.append(sum_stats)
        return np.array(weights), results

    @with_session
    def get_weighted_sum_stats(
        self, t: int = None
    ) -> (List[float], List[dict]):
        """
        Population's weighted summary statistics.
        These weights do not necessarily sum up to 1.

        Parameters
        ----------
        t: int, optional (default = self.max_t)
            Population index.
            If t is None, the latest population is selected.

        Returns
        -------
        weights, sum_stats:
            In the same order in the first array the weights (multiplied by
            the model probabilities), and tin the second array the summary
            statistics.
        """

        if t is None:
            t = self.max_t
        else:
            t = int(t)

        models = (
            self._session.query(Model)
            .join(Population)
            .join(ABCSMC)
            .filter(ABCSMC.id == self.id)
            .filter(Population.t == t)
            .options(
                subqueryload(Model.particles)
                .subqueryload(Particle.samples)
                .subqueryload(Sample.summary_statistics)
            )
            .all()
        )

        all_weights = []
        all_sum_stats = []

        for model in models:
            for particle in model.particles:
                weight = particle.w * model.p_model
                for sample in particle.samples:
                    # extract sum stats
                    sum_stats = {}
                    for ss in sample.summary_statistics:
                        sum_stats[ss.name] = ss.value
                    all_weights.append(weight)
                    all_sum_stats.append(sum_stats)

        return all_weights, all_sum_stats

    @with_session
    def get_population(self, t: int = None):
        """
        Create a pyabc.Population object containing all particles,
        as far as those can be recreated from the database. In particular,
        rejected particles are currently not stored.

        Parameters
        ----------
        t: int, optional (default = self.max_t)
            The population index.
        """
        if t is None:
            t = self.max_t
        else:
            t = int(t)

        models = (
            self._session.query(Model)
            .join(Population)
            .join(ABCSMC)
            .options(
                subqueryload(Model.particles)
                .subqueryload(Particle.samples)
                .subqueryload(Sample.summary_statistics),
                subqueryload(Model.particles).subqueryload(
                    Particle.parameters
                ),
            )
            .filter(ABCSMC.id == self.id)
            .filter(Population.t == t)
            .all()
        )

        py_particles = []

        # iterate over models
        for model in models:
            # model id
            py_m = model.m
            for particle in model.particles:
                # weight
                py_weight = particle.w * model.p_model

                # parameter
                py_parameter = {}
                for parameter in particle.parameters:
                    py_parameter[parameter.name] = parameter.value
                py_parameter = PyParameter(**py_parameter)

                # simulations
                # TODO this is legacy from when there were multiple
                if len(particle.samples) != 1:
                    raise AssertionError("There should be exactly one sample.")
                sample = particle.samples[0]
                # summary statistics
                py_sum_stat = {}
                for sum_stat in sample.summary_statistics:
                    py_sum_stat[sum_stat.name] = sum_stat.value

                # create particle
                py_particle = PyParticle(
                    m=py_m,
                    parameter=py_parameter,
                    weight=py_weight,
                    sum_stat=py_sum_stat,
                    distance=sample.distance,
                    accepted=True,
                    proposal_id=particle.proposal_id,
                )
                py_particles.append(py_particle)

        # create population
        py_population = PyPopulation(py_particles)

        return py_population

    @with_session
    def get_population_strategy(self):
        """Get information on the population size strategy.

        Returns
        -------
        population_strategy:
            The population strategy.
        """
        abc = self._session.query(ABCSMC).filter(ABCSMC.id == self.id).one()
        return json.loads(abc.population_strategy)

    @with_session
    def get_population_extended(
        self,
        *,
        m: Union[int, None] = None,
        t: Union[int, str] = "last",
        tidy: bool = True,
    ) -> pd.DataFrame:
        """
        Get extended population information, including parameters, distances,
        summary statistics, weights and more.

        Parameters
        ----------
        m: int or None, optional (default = None)
            The model to query. If omitted, all models are returned.
        t: int or str, optional (default = "last")
            Can be "last" or "all", or a population index (i.e. an int).
            In case of "all", all populations are returned.
            If "last", only the last population is returned, for an int value
            only the corresponding population at that time index.
        tidy: bool, optional
            If True, try to return a tidy DataFrame, where the individual
            parameters and summary statistics are pivoted.
            Setting tidy to true will only work for a single model and
            a single population.

        Returns
        -------
        full_population: DataFrame
        """
        query = (
            self._session.query(
                Population.t,
                Population.epsilon,
                Population.nr_samples.label("samples"),
                Model.m,
                Model.name.label("model_name"),
                Model.p_model,
                Particle.w,
                Particle.id.label("particle_id"),
                Sample.distance,
                Parameter.name.label("par_name"),
                Parameter.value.label("par_val"),
                SummaryStatistic.name.label("sumstat_name"),
                SummaryStatistic.value.label("sumstat_val"),
            )
            .join(ABCSMC)
            .join(Model)
            .join(Particle)
            .join(Sample)
            .join(SummaryStatistic)
            .join(Parameter)
            .filter(ABCSMC.id == self.id)
        )

        if m is not None:
            query = query.filter(Model.m == m)

        if t == "last":
            t = self.max_t

        # if t is not "all", filter for time point t
        if t != "all":
            query = query.filter(Population.t == t)

        df = pd.read_sql_query(query.statement, self._engine)

        if len(df.m.unique()) == 1:
            del df["m"]
            del df["model_name"]
            del df["p_model"]

        if isinstance(t, int):
            del df["t"]

        if tidy:
            if isinstance(t, int) and "m" not in df:
                df = df.set_index("particle_id")
                df_unique = df[["distance", "w"]].drop_duplicates()

                df_par = (
                    df[["par_name", "par_val"]]
                    .reset_index()
                    .drop_duplicates(subset=["particle_id", "par_name"])
                    .pivot(
                        index="particle_id",
                        columns="par_name",
                        values="par_val",
                    )
                )
                df_par.columns = ["par_" + c for c in df_par.columns]

                df_sumstat = (
                    df[["sumstat_name", "sumstat_val"]]
                    .reset_index()
                    .drop_duplicates(subset=["particle_id", "sumstat_name"])
                    .pivot(
                        index="particle_id",
                        columns="sumstat_name",
                        values="sumstat_val",
                    )
                )
                df_sumstat.columns = [
                    "sumstat_" + c for c in df_sumstat.columns
                ]

                df_tidy = df_unique.merge(
                    df_par, left_index=True, right_index=True
                ).merge(df_sumstat, left_index=True, right_index=True)
                df = df_tidy

        return df

    @with_session
    def get_ground_truth_parameter(self) -> PyParameter:
        """
        Create a pyabc.Parameter object from the ground truth parameters
        saved in the database, if existent.

        Returns
        -------
            A PyParameter dictionary.
        """
        # extract gt par from PRE_TIME iteration
        pars = (
            self._session.query(Parameter)
            .join(Particle)
            .join(Model)
            .join(Population)
            .join(ABCSMC)
            .filter(ABCSMC.id == self.id)
            .filter(Population.t == self.PRE_TIME)
            .filter(Model.p_model == 1)
            .all()
        )
        # create dict
        dct = {par.name: par.value for par in pars}
        # return as pyabc.Parameter
        return PyParameter(dct)
