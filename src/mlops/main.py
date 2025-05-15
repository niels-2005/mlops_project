from mlops.components.data_ingestion import DataIngestion
from mlops.components.data_validation import DataValidation
from mlops.config.data_ingestion_config import DataIngestionConfig
from mlops.config.data_validation_config import DataValidationConfig


def run_mlops_pipeline():
    data_ingestion_config = DataIngestionConfig()
    data_ingestion = DataIngestion(data_ingestion_config)
    data_ingestion_artifact = data_ingestion.run_data_ingestion()
    data_validation_config = DataValidationConfig()
    data_validation = DataValidation(data_ingestion_artifact, data_validation_config)
    data_validation_artifact = data_validation.run_data_validation()


if __name__ == "__main__":
    run_mlops_pipeline()
