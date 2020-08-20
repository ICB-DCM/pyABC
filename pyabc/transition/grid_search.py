from sklearn.model_selection import GridSearchCV as GridSearchCVSKL
import logging
import numpy as np
from .multivariatenormal import MultivariateNormalTransition

logger = logging.getLogger("GridSearchCV")


class GridSearchCV(GridSearchCVSKL):
    """
    Do a grid search to automatically select the best parameters for transition
    classes such as the :class:`pyabc.transition.MultivariateNormalTransition`.

    This is essentially a thin wrapper around
    'sklearn.model_selection.GridSearchCV'. It translates the scikit-learn
    interface to the interface used in pyABC. It implements hence a thin
    `adapter pattern <https://en.wikipedia.org/wiki/Adapter_pattern>`_.

    The parameters are just as for sklearn.model_selection.GridSearchCV.
    Major default values:

    - estimator = MultivariateNormalTransition()
    - param_grid = {'scaling': np.linspace(0.05, 1.0, 5)}
    - cv = 5

    """
    def __init__(self, estimator=None, param_grid=None,
                 scoring=None,
                 n_jobs=1, refit=True, cv=5,
                 verbose=0, pre_dispatch='2*n_jobs', error_score='raise',
                 return_train_score=True):

        if estimator is None:
            estimator = MultivariateNormalTransition()
        if param_grid is None:
            param_grid = {'scaling': np.linspace(0.05, 1.0, 5)}

        self.best_estimator_ = None

        super().__init__(
            estimator=estimator, param_grid=param_grid, scoring=scoring,
            n_jobs=n_jobs, pre_dispatch=pre_dispatch,
            cv=cv, refit=refit, verbose=verbose,
            error_score=error_score,
            return_train_score=return_train_score)

    def fit(self, X, y=None, groups=None):
        """
        Fit the density estimator (perturber) to the sampled data.
        """
        if len(X) == 1:
            res = self.estimator.fit(X, y)
            self.best_estimator_ = self.estimator
            logging.debug("Single sample Gridsearch. Params: {}".format(
                self.estimator.get_params()))
            return res

        if self.cv > len(X):  # pylint: disable=E0203
            old_cv = self.cv  # pylint: disable=E0203
            self.cv = len(X)
            res = super().fit(X, y, groups=groups)
            self.cv = old_cv
            logging.info("Reduced CV Gridsearch {} -> {}. Best params: {}"
                         .format(self.cv, len(X), self.best_params_))
            return res

        res = super().fit(X, y, groups=groups)
        logging.info("Best params: {}".format(self.best_params_))
        return res

    def __getattr__(self, item):
        if item == "best_estimator_":
            raise AttributeError
        return getattr(self.best_estimator_, item)
