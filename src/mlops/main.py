from mlops.components.data_ingestion import DataIngestion
from mlops.components.data_transformation import DataTransformation
from mlops.components.data_validation import DataValidation
from mlops.config.data_ingestion_config import DataIngestionConfig
from mlops.config.data_transformation_config import DataTransformationConfig
from mlops.config.data_validation_config import DataValidationConfig
from src.logger.get_logger import get_logger

logger = get_logger()


def run_mlops_pipeline():
    try:
        logger.info("Starting MLOps Pipeline")
        data_ingestion_config = DataIngestionConfig()
        data_ingestion = DataIngestion(data_ingestion_config)
        data_ingestion_artifact = data_ingestion.run_data_ingestion()
        data_validation_config = DataValidationConfig()
        data_validation = DataValidation(
            data_ingestion_artifact, data_validation_config
        )
        data_validation_artifact = data_validation.run_data_validation()
        data_transformation_config = DataTransformationConfig()
        data_transformation = DataTransformation(
            data_validation_artifact, data_transformation_config
        )
        data_transformation_artifact = data_transformation.run_data_transformation()
    except Exception as e:
        logger.error(f"Error while running Pipeline: {e}")


if __name__ == "__main__":
    run_mlops_pipeline()
