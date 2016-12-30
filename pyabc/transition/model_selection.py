from sklearn.model_selection import GridSearchCV as GridSearchCVSKL


class GridSearchCV(GridSearchCVSKL):
    def __getattr__(self, item):
        return getattr(self.estimator, item)