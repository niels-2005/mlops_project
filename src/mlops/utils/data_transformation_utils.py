import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


class FeatureBinning(BaseEstimator, TransformerMixin):
    def __init__(self, column_name, bins, labels):
        self.column_name = column_name
        self.bins = bins
        self.labels = labels

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X[self.column_name] = pd.cut(
            X[self.column_name], bins=self.bins, labels=self.labels
        )
        return X


def perform_feature_binning(feature_binning_schema, train_df, test_df):
    feature_binning = FeatureBinning(
        feature_binning_schema["column"],
        feature_binning_schema["bins"],
        feature_binning_schema["labels"],
    )
    train_df = feature_binning.transform(train_df)
    test_df = feature_binning.transform(test_df)
    return feature_binning, train_df, test_df


def drop_null_values(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = train_df.dropna()
    test_df = test_df.dropna()
    return train_df, test_df


def drop_duplicates(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = train_df.drop_duplicates()
    test_df = test_df.drop_duplicates()
    return train_df, test_df


def perform_feature_scaling(
    train_df: pd.DataFrame, test_df: pd.DataFrame, features: list[str]
):
    scaler = StandardScaler()
    train_df[features] = scaler.fit_transform(train_df[features])
    test_df[features] = scaler.transform(test_df[features])
    return scaler, train_df, test_df
