import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from src.logger.get_logger import get_logger
from src.mlops.utils.common_utils import save_object

logger = get_logger()


class FeatureBinning(BaseEstimator, TransformerMixin):
    def __init__(self, binning_config):
        self.binning_config = binning_config

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        for config in self.binning_config:
            column = config["column"]
            bins = config["bins"]
            labels = config["labels"]
            X_copy[column] = pd.cut(X_copy[column], bins=bins, labels=labels)
        return X_copy


class FeatureScaler(BaseEstimator, TransformerMixin):
    def __init__(self, scaler, features):
        self.scaler = scaler
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy[self.features] = self.scaler.transform(X_copy[self.features])
        return X_copy


def drop_null_values(df: pd.DataFrame, split: str):
    try:
        logger.info(f"Dropping null values for {split} dataframe.")
        return df.dropna()
    except Exception as e:
        logger.exception(
            f"Error occured while dropping null values for {split} dataframe: {e}"
        )
        raise e


def drop_duplicates(df: pd.DataFrame, split: str):
    try:
        logger.info(f"Dropping duplicated values for {split} dataframe.")
        return df.drop_duplicates()
    except Exception as e:
        logger.exception(
            f"Error occured while dropping duplicated values for {split} dataframe: {e}"
        )
        raise e


def perform_feature_binning(
    binning_config, train_df, test_df, artifact_path, inference_path
):
    try:
        logger.info(f"Performing Feature Binning with config: {binning_config}")
        feature_binning = FeatureBinning(binning_config)
        train_df = feature_binning.transform(train_df)
        test_df = feature_binning.transform(test_df)
        save_object(feature_binning, artifact_path)
        save_object(feature_binning, inference_path)
        return train_df, test_df
    except Exception as e:
        logger.exception(
            f"Error occured while performing feature binning with config: {binning_config}: {e}"
        )
        raise e


def get_scaler(scaler_name):
    try:
        logger.info(f"Returning scaler: {scaler_name}")
        if scaler_name == "standard_scaler":
            return StandardScaler()
        elif scaler_name == "min_max_scaler":
            return MinMaxScaler()
        elif scaler_name == "robust_scaler":
            return RobustScaler()
    except Exception as e:
        logger.exception(
            f"Error occured while returning feature scaler: {scaler_name}: {e}"
        )
        raise e


def perform_feature_scaling(
    feature_scaling_schema, train_df, test_df, artifact_path, inference_path
):
    try:
        features = list(feature_scaling_schema["columns_to_scale"])
        logger.info(f"Performing feature scaling for columns: {features}")
        scaler = get_scaler(feature_scaling_schema["scaler_name"])

        train_df[features] = scaler.fit_transform(train_df[features])

        feature_scaler = FeatureScaler(scaler, features)
        test_df = feature_scaler.transform(test_df)

        save_object(feature_scaler, artifact_path)
        save_object(feature_scaler, inference_path)
        return train_df, test_df
    except Exception as e:
        logger.exception(
            f"Error occured while performing feature scaling for columns: {features}: {e}"
        )
        raise e
