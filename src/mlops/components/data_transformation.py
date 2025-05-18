from mlops.artifacts.data_transformation_artifact import \
    DataTransformationArtifact
from mlops.artifacts.data_validation_artifact import DataValidationArtifact
from mlops.config.data_transformation_config import DataTransformationConfig
from mlops.utils.common_utils import (create_directory, get_os_path,
                                      read_dataset, read_yaml_file,
                                      save_file_as_csv, save_object)
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
        create_directory(self.config.data_transformation_dir)
        create_directory(self.config.preprocessors_dir)
        create_directory(self.config.transformed_data_dir)
        self.schema = read_yaml_file(self.config.schema_path)
        self.feature_binning_schema = self.schema["feature_binning"]

    def run_data_transformation(self):
        try:
            self.logger.info("Data Transformation started.")
            train_df = read_dataset(self.data_validation_artifact.validated_train_path)
            test_df = read_dataset(self.data_validation_artifact.validated_test_path)

            if self.schema["drop_null_values"]:
                train_df, test_df = drop_null_values(train_df, test_df)

            if self.schema["remove_duplicates"]:
                train_df, test_df = drop_duplicates(train_df, test_df)

            feature_binning, train_df, test_df = perform_feature_binning(
                self.feature_binning_schema, train_df, test_df
            )
            save_object(feature_binning, self.config.feature_binning_artifact_path)
            save_object(feature_binning, self.config.feature_binning_inference_path)

            scaler, train_df, test_df = perform_feature_scaling(
                train_df, test_df, list(self.schema["columns_to_scale"])
            )
            save_object(scaler, self.config.standard_scaler_artifact_path)
            save_object(scaler, self.config.standard_scaler_inference_path)

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
