import pandas as pd

from mlops.artifacts.data_ingestion_artifact import DataIngestionArtifact
from mlops.artifacts.data_validation_artifact import DataValidationArtifact
from mlops.config.data_validation_config import DataValidationConfig
from mlops.utils.common_utils import (
    create_directory,
    read_dataset,
    read_yaml_file,
    save_file_as_csv,
    write_yaml_file,
)
from src.logger.get_logger import get_logger

logger = get_logger()


class DataValidation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        config: DataValidationConfig,
    ):
        self.data_ingestion_artifact = data_ingestion_artifact
        self.config = config
        create_directory(self.config.data_validation_dir)
        create_directory(self.config.validated_data_dir)
        create_directory(self.config.invalidated_data_dir)
        self.schema = read_yaml_file(self.config.schema_path)
        self.len_original_columns = len(self.schema["columns"])
        self.column_schema = {
            list(col.keys())[0]: list(col.values())[0] for col in self.schema["columns"]
        }
        self.validation_status = True

    def validate_number_of_columns(self, df: pd.DataFrame) -> None:
        try:
            logger.info("Validating the Length of Columns in DataFrame.")
            if len(df.columns) != self.len_original_columns:
                self.validation_status = False
        except Exception as e:
            logger.error(f"Error while validating number of columns: {e}")
            raise e

    def generate_validation_report(self, df: pd.DataFrame, file_path: str) -> None:
        try:
            logger.info(f"Generating Validation Report for: {file_path}")
            validation_results = []
            for col_name, expected_dtype in self.column_schema.items():
                if col_name not in df.columns:
                    validation_results.append(
                        {
                            "column": col_name,
                            "expected_dtype": expected_dtype,
                            "got_dtype": "MISSING",
                            "validated": False,
                        }
                    )
                    self.validation_status = False
                    continue

                actual_dtype = str(df[col_name].dtype)
                is_valid = expected_dtype in actual_dtype
                if not is_valid:
                    self.validation_status = False

                validation_results.append(
                    {
                        "column": col_name,
                        "expected_dtype": expected_dtype,
                        "got_dtype": actual_dtype,
                        "validated": is_valid,
                    }
                )

            content = {
                "columns": validation_results,
                "validation_status": self.validation_status,
            }

            write_yaml_file(file_path, content)

        except Exception as e:
            logger.error(f"Error while generating validation report: {e}")
            raise e

    def run_data_validation(self) -> DataValidationArtifact:
        logger.info("Data Validation started.")
        train_df = read_dataset(self.data_ingestion_artifact.train_file_path)
        test_df = read_dataset(self.data_ingestion_artifact.test_file_path)

        self.validate_number_of_columns(train_df)
        self.validate_number_of_columns(test_df)

        self.generate_validation_report(train_df, self.config.is_validated_train_path)
        self.generate_validation_report(test_df, self.config.is_validated_test_path)

        if self.validation_status == False:
            save_file_as_csv(train_df, self.config.invalidated_train_path)
            save_file_as_csv(test_df, self.config.invalidated_test_path)
            logger.error(f"Data Validation failed, stopping Pipeline...")
            raise ValueError("Data validation failed. Check the validation report.")
        else:
            save_file_as_csv(train_df, self.config.validated_train_path)
            save_file_as_csv(test_df, self.config.validated_test_path)
            data_validation_artifact = DataValidationArtifact(
                self.config.validated_train_path, self.config.validated_test_path
            )
            logger.info(f"Data Validation returns: {data_validation_artifact}")
            logger.info("Data Validation completed.")
            return data_validation_artifact
