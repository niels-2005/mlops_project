import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from mlops.artifacts.data_transformation_artifact import \
    DataTransformationArtifact
from mlops.artifacts.data_validation_artifact import DataValidationArtifact
from mlops.config.data_transformation_config import DataTransformationConfig
from mlops.utils.common_utils import (create_directories, read_dataset,
                                      read_yaml_file, save_file_as_csv,
                                      save_object, write_yaml_file)
from mlops.utils.data_transformation_utils import FeatureBinning
from src.logger.get_logger import get_logger


class DataTransformation:
    def __init__(
        self,
        data_validation_artifact: DataValidationArtifact,
        config: DataTransformationConfig,
    ):
        self.data_validation_artifact = data_validation_artifact
        self.config = config
        self.logger = get_logger()
        create_directories(
            [
                self.config.data_transformation_dir,
                self.config.preprocessors_dir,
                self.config.transformed_data_dir,
            ]
        )
        self.schema = read_yaml_file(self.config.schema_read_path)
        write_yaml_file(self.config.schema_save_path, self.schema)
        self.feature_binning_schema = self.schema["feature_binning"]
        self.feature_scaling_schema = self.schema["feature_scaling"]

    def drop_null_values(self, df: pd.DataFrame, split: str):
        try:
            self.logger.info(f"Dropping null values for {split} dataframe.")
            return df.dropna()
        except Exception as e:
            self.logger.exception(
                f"Error occured while dropping null values for {split} dataframe: {e}"
            )
            raise e

    def drop_duplicates(self, df: pd.DataFrame, split: str):
        try:
            self.logger.info(f"Dropping duplicated values for {split} dataframe.")
            return df.drop_duplicates()
        except Exception as e:
            self.logger.exception(
                f"Error occured while dropping duplicated values for {split} dataframe: {e}"
            )
            raise e

    def perform_feature_binning(
        self, feature_binning_schema, train_df, test_df, artifact_path, inference_path
    ):
        try:
            binning_config = list(feature_binning_schema["columns"])
            self.logger.info(
                f"Performing Feature Binning for columns: {binning_config}"
            )
            feature_binning = FeatureBinning(binning_config)
            train_df = feature_binning.transform(train_df)
            test_df = feature_binning.transform(test_df)
            save_object(feature_binning, artifact_path)
            save_object(feature_binning, inference_path)
            return train_df, test_df
        except Exception as e:
            self.logger.exception(
                f"Error occured while performing feature binning for columns: {binning_config}: {e}"
            )
            raise e

    def get_scaler(self, scaler_name):
        try:
            self.logger.info(f"Returning scaler: {scaler_name}")
            if scaler_name == "standard_scaler":
                return StandardScaler()
            elif scaler_name == "min_max_scaler":
                return MinMaxScaler()
            elif scaler_name == "robust_scaler":
                return RobustScaler()
        except Exception as e:
            self.logger.exception(
                f"Error occured while returning feature scaler: {scaler_name}: {e}"
            )
            raise e

    def perform_feature_scaling(
        self, feature_scaling_schema, train_df, test_df, artifact_path, inference_path
    ):
        try:
            features = list(feature_scaling_schema["columns_to_scale"])
            self.logger.info(f"Performing feature scaling for columns: {features}")
            scaler = self.get_scaler(feature_scaling_schema["scaler_name"])
            train_df[features] = scaler.fit_transform(train_df[features])
            test_df[features] = scaler.transform(test_df[features])
            save_object(scaler, artifact_path)
            save_object(scaler, inference_path)
            return train_df, test_df
        except Exception as e:
            self.logger.exception(
                f"Error occured while performing feature scaling for columns: {features}: {e}"
            )
            raise e

    def run_data_transformation(self):
        try:
            self.logger.info("Data Transformation started.")
            train_df = read_dataset(self.data_validation_artifact.validated_train_path)
            test_df = read_dataset(self.data_validation_artifact.validated_test_path)

            if self.schema["drop_null_values"]:
                train_df = self.drop_null_values(train_df, split="train")
                test_df = self.drop_null_values(test_df, split="test")

            if self.schema["remove_duplicates"]:
                train_df = self.drop_duplicates(train_df, split="train")
                test_df = self.drop_duplicates(test_df, split="test")

            if self.feature_binning_schema["enabled"]:
                train_df, test_df = self.perform_feature_binning(
                    self.feature_binning_schema,
                    train_df,
                    test_df,
                    self.config.feature_binning_artifact_path,
                    self.config.feature_binning_inference_path,
                )

            if self.feature_scaling_schema["enabled"]:
                train_df, test_df = self.perform_feature_scaling(
                    self.feature_scaling_schema,
                    train_df,
                    test_df,
                    self.config.standard_scaler_artifact_path,
                    self.config.standard_scaler_inference_path,
                )

            save_file_as_csv(train_df, self.config.transformed_train_path)
            save_file_as_csv(test_df, self.config.transformed_test_path)

            data_transformation_artifact = DataTransformationArtifact(
                self.config.transformed_train_path, self.config.transformed_test_path
            )
            self.logger.info(
                f"Data Transformation returns: {data_transformation_artifact}"
            )
            self.logger.info("Data Transformation completed.")
            return data_transformation_artifact
        except Exception as e:
            self.logger.error(f"Error occurred during data transformation: {e}")
            raise e
