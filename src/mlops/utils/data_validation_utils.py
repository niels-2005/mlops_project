import pandas as pd

from logger.get_logger import get_logger
from mlops.utils.common_utils import write_yaml_file

logger = get_logger()


def generate_validation_report(df: pd.DataFrame, column_schema, file_path: str) -> None:
    try:
        logger.info(f"Generating Validation Report for: {file_path}")
        validation_results, validation_status = get_validation_results(
            df, column_schema
        )
        save_validation_report(validation_results, validation_status, file_path)
        return validation_status
    except Exception as e:
        logger.exception(
            f"Error occured while generating validation report for {file_path}: {e}"
        )
        raise e


def get_validation_results(df: pd.DataFrame, column_schema):
    try:
        logger.info("Generating Validation Results.")
        validation_status = True
        validation_results = []
        for col_name, expected_dtype in column_schema.items():

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
        logger.exception(f"Error occured while generating validation results: {e}")
        raise e


def save_validation_report(validation_results, validation_status, file_path):
    try:
        logger.info(f"Saving validation report at: {file_path}")
        write_yaml_file(
            file_path,
            content={
                "columns": validation_results,
                "validation_status": validation_status,
            },
        )
    except Exception as e:
        logger.exception(
            f"Error occured while saving validation report at {file_path}: {e}"
        )
        raise e
