import pandas as pd
from sklearn.preprocessing import StandardScaler

from mlops.artifacts.data_transformation_artifact import \
    DataTransformationArtifact
from mlops.artifacts.data_validation_artifact import DataValidationArtifact
from mlops.config.data_transformation_config import DataTransformationConfig
from mlops.utils.common_utils import (create_directory, read_dataset,
                                      read_yaml_file, save_file_as_csv,
                                      save_object)
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
        create_directory(self.config.data_transformation_dir)
        create_directory(self.config.preprocessors_dir)
        create_directory(self.config.transformed_data_dir)
        self.schema = read_yaml_file(self.config.schema_path)
        self.feature_binning_schema = self.schema["feature_binning"]

    def drop_null_values(self, df: pd.DataFrame, file_path: str) -> pd.DataFrame:
        try:
            original_length = len(df)
            df = df.dropna()
            without_null_length = len(df)
            self.logger.info(
                f"Dropping Null Values from {file_path}, Original Length: \
                {original_length}, removed {original_length - without_null_length} Null Values."
            )
            return df
        except Exception as e:
            self.logger.error(
                f"Error while dropping Null Values from: {file_path}: {e}"
            )
            raise e

    def drop_duplicates(self, df: pd.DataFrame, file_path: str) -> pd.DataFrame:
        try:
            original_length = len(df)
            df = df.drop_duplicates()
            without_duplicates_length = len(df)
            self.logger.info(
                f"Dropping Duplicated Values from {file_path}, Original Length: \
                {original_length}, removed {original_length - without_duplicates_length} Duplicated Values."
            )
            return df
        except Exception as e:
            self.logger.error(
                f"Error while dropping duplicated Values from: {file_path}: {e}"
            )
            raise e

    def perform_feature_binning(
        self,
        df: pd.DataFrame,
        file_path: str,
        column_name: str,
        bins: list[int],
        labels: list[int],
    ) -> pd.DataFrame:
        try:
            self.logger.info(
                f"Performing Feature Binning for: {file_path}, Column: {column_name}"
            )
            binning = pd.cut(df[column_name], bins=bins, labels=labels)
            df[column_name] = binning
            return df
        except Exception as e:
            self.logger.error(
                f"Error while performing Feature Binning for {file_path}: {e}"
            )
            raise e

    def perform_feature_scaling(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        train_file_path: str,
        test_file_path: str,
        features: list[str],
        scaler_save_path: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        try:
            scaler = StandardScaler()
            self.logger.info(
                f"Performing Standard Scaling for: {features}, on files: {train_file_path}, {test_file_path}"
            )
            df_train[features] = scaler.fit_transform(df_train[features])
            df_test[features] = scaler.transform(df_test[features])
            save_object(scaler, scaler_save_path)
            return df_train, df_test
        except Exception as e:
            self.logger.error(
                f"Error while performing Data Scaling for: {train_file_path}, {test_file_path}: {e}"
            )
            raise e

    def run_data_transformation(self):
        self.logger.info("Data Transformation started.")
        train_df = read_dataset(self.data_validation_artifact.validated_train_path)
        test_df = read_dataset(self.data_validation_artifact.validated_test_path)

        if self.schema["drop_null_values"]:
            train_df = self.drop_null_values(
                train_df, self.data_validation_artifact.validated_train_path
            )
            test_df = self.drop_null_values(
                test_df, self.data_validation_artifact.validated_train_path
            )

        if self.schema["remove_duplicates"]:
            train_df = self.drop_duplicates(
                train_df, self.data_validation_artifact.validated_train_path
            )
            test_df = self.drop_duplicates(
                test_df, self.data_validation_artifact.validated_test_path
            )

        train_df = self.perform_feature_binning(
            train_df,
            self.data_validation_artifact.validated_train_path,
            self.feature_binning_schema["column"],
            self.feature_binning_schema["bins"],
            self.feature_binning_schema["labels"],
        )

        test_df = self.perform_feature_binning(
            test_df,
            self.data_validation_artifact.validated_test_path,
            self.feature_binning_schema["column"],
            self.feature_binning_schema["bins"],
            self.feature_binning_schema["labels"],
        )

        train_df, test_df = self.perform_feature_scaling(
            train_df,
            test_df,
            self.data_validation_artifact.validated_train_path,
            self.data_validation_artifact.validated_test_path,
            list(self.schema["columns_to_scale"]),
            self.config.standard_scaler_path,
        )

        save_file_as_csv(train_df, self.config.transformed_train_path)
        save_file_as_csv(test_df, self.config.transformed_test_path)

        data_transformation_artifact = DataTransformationArtifact(
            self.config.transformed_train_path, self.config.transformed_test_path
        )
        self.logger.info(f"Data Transformation returns: {data_transformation_artifact}")
        self.logger.info("Data Transformation completed.")
        return data_transformation_artifact
