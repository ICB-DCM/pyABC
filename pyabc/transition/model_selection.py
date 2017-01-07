from sklearn.model_selection import GridSearchCV as GridSearchCVSKL
import logging

logger = logging.getLogger("GridSearchCV")


class GridSearchCV(GridSearchCVSKL):
    def fit(self, X, y=None, groups=None):
        res = super().fit(X, y, groups)
        logging.info("Best params: {}".format(self.best_params_))
        return res

    def __getattr__(self, item):
        if item == "best_estimator_":
            raise AttributeError
        return getattr(self.best_estimator_, item)
