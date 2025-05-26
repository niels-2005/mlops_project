import pandas as pd

from logger.get_logger import get_logger
from mlops.artifacts.data_ingestion_artifact import DataIngestionArtifact
from mlops.artifacts.data_validation_artifact import DataValidationArtifact
from mlops.config.data_validation_config import DataValidationConfig
from mlops.utils.common_utils import (create_directories, read_dataset,
                                      read_yaml_file, save_file_as_csv,
                                      write_yaml_file)
from mlops.utils.data_validation_utils import generate_validation_report


class DataValidation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        config: DataValidationConfig,
    ):
        self.data_ingestion_artifact = data_ingestion_artifact
        self.config = config
        self.logger = get_logger()
        create_directories(
            [
                self.config.data_validation_dir,
                self.config.validation_reports_dir,
                self.config.validated_data_dir,
                self.config.invalidated_data_dir,
            ]
        )
        self.schema = read_yaml_file(self.config.schema_read_path)
        write_yaml_file(self.config.schema_save_path, self.schema)
        self.column_schema = {
            list(col.keys())[0]: list(col.values())[0] for col in self.schema["columns"]
        }

    def run_data_validation(self) -> DataValidationArtifact:
        try:
            self.logger.info("Data Validation started.")
            train_df = read_dataset(self.data_ingestion_artifact.train_file_path)
            test_df = read_dataset(self.data_ingestion_artifact.test_file_path)

            validation_status_train = generate_validation_report(
                train_df, self.column_schema, self.config.validation_report_train_path
            )
            validation_status_test = generate_validation_report(
                test_df, self.column_schema, self.config.validation_report_test_path
            )

            # raise Error if any validation status is False
            # error is replaceable with e.g. slack alert
            if validation_status_train == False or validation_status_test == False:
                save_file_as_csv(train_df, self.config.invalidated_train_path)
                save_file_as_csv(test_df, self.config.invalidated_test_path)
                self.logger.error(f"Data Validation failed, stopping Pipeline...")
                raise ValueError("Data validation failed. Check the validation report.")
            else:
                # continue mlops pipeline (no validation errors)
                save_file_as_csv(train_df, self.config.validated_train_path)
                save_file_as_csv(test_df, self.config.validated_test_path)
                data_validation_artifact = DataValidationArtifact(
                    self.config.validated_train_path, self.config.validated_test_path
                )
                self.logger.info(
                    f"Data Validation passed, returns: {data_validation_artifact}"
                )
                self.logger.info("Data Validation completed.")
                return data_validation_artifact
        except Exception as e:
            self.logger.exception(f"Error occurred during data_validation: {e}")
            raise e
