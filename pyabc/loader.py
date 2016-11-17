import pandas as pd
import json
from collections import namedtuple
from itertools import product
from numbers import Number
from functools import lru_cache

class SQLDataStore:
    """
    SQLData store for the ABCLoader class.

    Parameters
    ----------

    db: str
        SQLAlchemy connection string.
        E.g.: sqlite:////home/user/my_database.db
    """

    def __init__(self, db: str):
        sqlite_prefix = "sqlite:///"
        self.db = db if db.startswith(sqlite_prefix) else  sqlite_prefix + db

    def __repr__(self):
        return "{}(\"{}\")".format(self.__class__.__name__, self.db)

    def __getitem__(self, item) -> pd.DataFrame:
        return pd.read_sql_table(item, self.db)


def load_json(string):
    return json.loads(string.replace('"', "").replace("'", '"').replace("(", "[").replace(")", "]"))


def extract_noise_par(par_name, string):
    dictionary = load_json(string)
    noise_pars = dictionary["model_definitions"]["noise"]["parameters"]
    par = noise_pars[par_name]
    return par


class ABCLoader:
    """
    Load ABC results from database and analyse.

    Parameters
    ----------

    data_store: DataStore
        The datastore provides the database's tables as pandas dataframes.
        Can be a SQLDataStore or pandas.HDFStore
    """
    group_parameters = []  #: Paramters for grouping ABC sweeps.
    group_parameters_from_pars = ["noise"]
    group_parameters_from_noise = ["subsampling_fraction", "fraction_remove_and_add"]
    tables_to_convert_to_singular = ["models", "populations", "parameters", "particles"]
    tables_with_name_to_rename = ["models", "parameters"]

    def __init__(self, data_store: SQLDataStore):
        self._data_store = data_store
        self.purge_group_parameters_from_pars()

    def __repr__(self):
        return "{}(store={})".format(type(self).__name__, self._data_store)

    def _clean_name(self, name):
        if name in self.tables_to_convert_to_singular:
            return name[:-1]
        return name

    @lru_cache(30)
    def _data(self, name):
        df = self._data_store[name].copy()
        df.rename(columns={'id': self._clean_name(name) + '_id'}, inplace=True)
        if name in self.tables_with_name_to_rename:
            df.rename(columns={"name": self._clean_name(name) + "_name"}, inplace=True)
        return df

    def __getattr__(self, item):
        return self._data(item)

    @property
    def noise_type(self):
        return "Undefined noise"

    @property
    def model_names(self):
        """
        Unique names of the models found in the database.
        """
        return self.models.model_name.unique()

    def results(self):
        """
        Final results of the ABC runs.
        """
        results = self.results_including_initial()
        results_without_initial = results[results.max_t > -1]
        return results_without_initial

    def means(self):
        """
        Means of the results, grouped by the ``group_parameters``.
        """
        means = self.results().groupby(self.group_parameters).mean()
        return means

    @property
    def maxs(self):
        """
        Maxima of the results, grouped by the ``group_parameters``.
        """
        maxs = self.results().groupby(self.group_parameters).max()
        return maxs

    @property
    def max_nr_populations(self):
        """
        Maximun nr of populations.
        """
        return self.maxs.max_t.unstack(level=0)

    def maximum_a_posteriori(self):
        """
        MAP estimates, grouped by ``group_parameters``.
        """
        means = self.means()
        return means.MAP_correct.unstack(level=0)

    def average_mass_at_tround_truth(self):
        """
        Averaged posterior probabilities, grouped by ``group_parameters``.
        """
        return self.means().mass_at_gt_model.unstack(level=0)

    @property
    def confusion_matrices_table(self):
        """
        Confusion matrices.
        """
        confusion_matrices = self.results().groupby(self.group_parameters + ["gt_model_name"]).mean()
        confusion_matrices = confusion_matrices[self.model_names]
        confusion_matrices.sort_index(axis=1, inplace=True)
        return confusion_matrices

    def confusion_matrix_dict(self):
        """
        Confusion matrices as dict, with keys indicating the sweep parameters.
        """
        confusion_matrices = self.confusion_matrices_table
        if len(self.group_parameters) == 0:
            return {None: confusion_matrices}
        names = confusion_matrices.index.names[:-1]
        levels = confusion_matrices.index.levels[:-1]
        Index = namedtuple("Index", names)
        Index.names = Index._fields
        conf_mat_dict = {}
        for ix in product(*levels):
            try:
                new_value = confusion_matrices.loc[ix]
            except KeyError:
                pass
            else:
                conf_mat_dict[Index(*ix)] = new_value

        return conf_mat_dict

    def terminated_abc_smc_ids(self):
        """
        IDs of already terminated ABCSMC runs.
        """
        abc_smc = self.abc_smc
        abc_smc["terminated"] = abc_smc.end_time.notnull()
        return abc_smc[["abc_smc_id", "terminated"]]

    def particles_of_population(self, abc_smc_id: int, model_name: str, t: int):
        """
        Return the particles of a given population.
        Useful if the posterior parameters are of interest.

        Parameters
        ----------
        abc_smc_id: int
            ID of the ABCSMC run
        model_name: str
            Name of the model
        t: int
            Population number

        Returns
        -------

        particles: DataFrame
            The particles of the chosen population.
        """
        df = self.particles.merge(self.parameters).merge(self.models).merge(self.populations).merge(self.abc_smc)
        df = df[(df.model_name == model_name) & (df.max_t == t) & (df.abc_smc_id == abc_smc_id)]
        df = df.pivot("particle_id", "parameter_name", "value")
        return df

    @property
    def group_parameters(self):
        group_pars = self.group_parameters_from_pars + self.group_parameters_from_noise
        if "noise" in group_pars and "fraction_remove_and_add" in group_pars:
            group_pars.remove("fraction_remove_and_add")
        return group_pars

    @group_parameters.setter
    def group_parameters(self, value):
        raise AttributeError("Attribute is read only")

    def max_t(self):
        return self.populations[["abc_smc_id", "t"]].groupby("abc_smc_id", as_index=False).max()

    def intermediate_results(self, min_t=0):
        raw_data = self.populations.merge(self.models)[['abc_smc_id', 't', 'model_name', 'p_model', 'epsilon', 'nr_samples']]
        ground_truth = raw_data[raw_data.max_t == -1][["abc_smc_id", "model_name"]].rename(columns={"model_name": "gt_model"})
        epsilon = raw_data.groupby(["abc_smc_id", "t"])['epsilon'].first().reset_index()
        intermediate_results = raw_data[raw_data.max_t >= min_t]
        intermediate_results = intermediate_results.pivot_table(index=["abc_smc_id", "t"], columns=["model_name"],
                                                                values=["p_model", 'nr_samples'])["p_model"].reset_index()
        intermediate_results = intermediate_results.merge(epsilon).merge(ground_truth)
        intermediate_results = intermediate_results.merge(self.terminated_abc_smc_ids())
        intermediate_results = intermediate_results.merge(self.populations[["abc_smc_id", "t", "nr_samples"]],
                                                          on=["abc_smc_id", "t"])
        return intermediate_results

    def sweep_pars_from_noise(self):
        abc_sweep_par = self.abc_smc[["abc_smc_id", "json_parameters"]].copy()
        for par_name in self.group_parameters_from_noise:
            # not a perfect check but better than no check
            example_noise_par = extract_noise_par(par_name, abc_sweep_par["json_parameters"][0])
            if isinstance(example_noise_par, Number):
                # keep here the map. the example check is not perfect but still better than nothing
                abc_sweep_par[par_name] = abc_sweep_par["json_parameters"].map(lambda x: extract_noise_par(par_name, x))
        del abc_sweep_par["json_parameters"]
        return abc_sweep_par

    def purge_group_parameters_from_pars(self):
        parameters = self.parameters
        unique_par_names = parameters.parameter_name.unique()
        purged_group_parameters_from_pars = list(set(self.group_parameters_from_pars) & set(unique_par_names))
        self.group_parameters_from_pars = purged_group_parameters_from_pars

    def results_including_initial(self):
        if len(self.group_parameters) > 0:
            sweep_parameters = (self.parameters[self.parameters.parameter_name.isin(self.group_parameters_from_pars)]
                                .pivot("particle_id", "parameter_name", "value")
                                .reset_index()
                                .merge(self.particles)
                                .merge(self.models)
                                .merge(self.populations)
                                 [["abc_smc_id"] + self.group_parameters_from_pars]
                                )

        abc_max_t = self.max_t()
        last_populations = self.populations.merge(abc_max_t, left_on=["abc_smc_id", "t"], right_on=["abc_smc_id", "t"]).rename(
            columns={"t": "max_t"})[["population_id", "abc_smc_id"]]
        last_models = self.models.merge(last_populations)
        posterior = last_models.pivot("abc_smc_id", "model_name", "p_model").reset_index()
        posterior["MAP"] = posterior[self.model_names].idxmax(axis=1)

        gt = self.populations[self.populations.max_t == -1].merge(self.models)[["abc_smc_id", "model_name"]].rename(
            columns={"model_name": "gt_model_name"})

        results = posterior.merge(gt)
        results["MAP_correct"] = (results.MAP == results.gt_model_name).astype(float)
        results["mass_at_gt_model"] = results.apply(lambda x: x[x.gt_model_name], axis=1)
        if len(self.group_parameters) > 0 and len(sweep_parameters) > 0:
            results = results.merge(sweep_parameters)
        results = results.merge(abc_max_t)
        results.rename(columns={"t": "max_t"}, inplace=True)
        results = results.merge(self.terminated_abc_smc_ids())

        results = results.merge(self.sweep_pars_from_noise())
        return results
