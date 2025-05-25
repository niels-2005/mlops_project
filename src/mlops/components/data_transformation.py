from mlops.artifacts.data_transformation_artifact import \
    DataTransformationArtifact
from mlops.artifacts.data_validation_artifact import DataValidationArtifact
from mlops.config.data_transformation_config import DataTransformationConfig
from mlops.utils.common_utils import (create_directories, read_dataset,
                                      read_yaml_file, save_file_as_csv,
                                      write_yaml_file)
from mlops.utils.data_transformation_utils import (drop_duplicates,
                                                   drop_null_values,
                                                   perform_feature_binning,
                                                   perform_feature_scaling)
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

    def run_data_transformation(self):
        try:
            self.logger.info("Data Transformation started.")
            train_df = read_dataset(self.data_validation_artifact.validated_train_path)
            test_df = read_dataset(self.data_validation_artifact.validated_test_path)

            if self.schema["drop_null_values"]:
                train_df = drop_null_values(train_df, split="train")
                test_df = drop_null_values(test_df, split="test")

            if self.schema["remove_duplicates"]:
                train_df = drop_duplicates(train_df, split="train")
                test_df = drop_duplicates(test_df, split="test")

            if self.feature_binning_schema["enabled"]:
                train_df, test_df = perform_feature_binning(
                    list(self.feature_binning_schema["columns"]),
                    train_df,
                    test_df,
                    self.config.feature_binning_artifact_path,
                    self.config.feature_binning_inference_path,
                )

            if self.feature_scaling_schema["enabled"]:
                train_df, test_df = perform_feature_scaling(
                    self.feature_scaling_schema,
                    train_df,
                    test_df,
                    self.config.scaler_artifact_path,
                    self.config.scaler_inference_path,
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
