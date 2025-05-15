from mlops.components.data_ingestion import DataIngestion
from mlops.config.data_ingestion_config import DataIngestionConfig


def run_mlops_pipeline():
    data_ingestion_config = DataIngestionConfig()
    data_ingestion = DataIngestion(data_ingestion_config)
    data_ingestion.run_data_ingestion()


if __name__ == "__main__":
    run_mlops_pipeline()
