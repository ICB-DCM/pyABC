from sklearn.model_selection import GridSearchCV as GridSearchCVSKL
import logging

logger = logging.getLogger("GridSearchCV")


class GridSearchCV(GridSearchCVSKL):
    """
    Do a grid search to automatically select the best parameters for transition
    classes such as the :class:`pyabc.transition.MultivariateNormalTransition`.

    This is essentially a thin wrapper around
    'sklearn.model_selection.GridSearchCV'. It translates the scikit-learn
    interface to the interface used in pyABC. It implements hence a thin
    `adapter pattern <https://en.wikipedia.org/wiki/Adapter_pattern>`_.
    """
    def __init__(self, estimator, param_grid, scoring=None, fit_params=None,
                 n_jobs=1, iid=True, refit=True, cv=5,
                 verbose=0, pre_dispatch='2*n_jobs', error_score='raise',
                 return_train_score=True):
        super().__init__(estimator, param_grid, scoring, fit_params, n_jobs,
                         iid, refit, cv, verbose, pre_dispatch,
                         error_score, return_train_score)

    def fit(self, X, y=None, groups=None):
        if len(X) == 1:
            res = self.estimator.fit(X, y)
            self.best_estimator_ = self.estimator
            logging.debug("Single sample Gridsearch. Params: {}".format(
                self.estimator.get_params()))
            return res

        if self.cv > len(X):
            old_cv = self.cv
            self.cv = len(X)
            res = super().fit(X, y, groups)
            self.cv = old_cv
            logging.info("Rreduced CV Gridsearch {} -> {}. Best params: {}"
                         .format(self.cv, len(X), self.best_params_))
            return res

        res = super().fit(X, y, groups)
        logging.info("Best params: {}".format(self.best_params_))
        return res

    def __getattr__(self, item):
        if item == "best_estimator_":
            raise AttributeError
        return getattr(self.best_estimator_, item)
