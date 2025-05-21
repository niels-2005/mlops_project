import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


class FeatureBinning(BaseEstimator, TransformerMixin):
    def __init__(self, binning_config):
        self.binning_config = binning_config

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for config in self.binning_config:
            column = config["column"]
            bins = config["bins"]
            labels = config["labels"]
            X[column] = pd.cut(X[column], bins=bins, labels=labels)
        return X
