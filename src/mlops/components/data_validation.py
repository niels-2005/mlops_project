import pandas as pd

from mlops.artifacts.data_ingestion_artifact import DataIngestionArtifact
from mlops.artifacts.data_validation_artifact import DataValidationArtifact
from mlops.config.data_validation_config import DataValidationConfig
from mlops.utils.common_utils import (create_directories, read_dataset,
                                      read_yaml_file, save_file_as_csv,
                                      write_yaml_file)
from src.logger.get_logger import get_logger


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

    def get_validation_results(self, df: pd.DataFrame):
        try:
            self.logger.info("Generating Validation Results.")
            validation_status = True
            validation_results = []
            for col_name, expected_dtype in self.column_schema.items():

                # check if col_name in dataframe
                if col_name not in df.columns:
                    validation_status = False

                # check dtype validation
                actual_dtype = str(df[col_name].dtype)
                is_valid = expected_dtype in actual_dtype
                if not is_valid:
                    validation_status = False

                validation_results.append(
                    {
                        "column": col_name,
                        "expected_dtype": expected_dtype,
                        "got_dtype": actual_dtype,
                        "validated": validation_status,
                    }
                )
            return validation_results, validation_status
        except Exception as e:
            self.logger.exception(
                f"Error occured while generating validation results: {e}"
            )
            raise e

    def save_validation_report(self, validation_results, validation_status, file_path):
        try:
            self.logger.info(f"Saving validation report at: {file_path}")
            write_yaml_file(
                file_path,
                content={
                    "columns": validation_results,
                    "validation_status": validation_status,
                },
            )
        except Exception as e:
            self.logger.exception(
                f"Error occured while saving validation report at {file_path}: {e}"
            )
            raise e

    def generate_validation_report(self, df: pd.DataFrame, file_path: str) -> None:
        try:
            self.logger.info(f"Generating Validation Report for: {file_path}")
            validation_results, validation_status = self.get_validation_results(df)
            self.save_validation_report(
                validation_results, validation_status, file_path
            )
            return validation_status
        except Exception as e:
            self.logger.exception(
                f"Error occured while generating validation report for {file_path}: {e}"
            )
            raise e

    def run_data_validation(self) -> DataValidationArtifact:
        try:
            self.logger.info("Data Validation started.")
            train_df = read_dataset(self.data_ingestion_artifact.train_file_path)
            test_df = read_dataset(self.data_ingestion_artifact.test_file_path)

            validation_status_train = self.generate_validation_report(
                train_df, self.config.validation_report_train_path
            )
            validation_status_test = self.generate_validation_report(
                test_df, self.config.validation_report_test_path
            )

            # raise Error if any validation status is False
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
