from mlops.artifacts.data_ingestion_artifact import DataIngestionArtifact
from mlops.config.data_ingestion_config import DataIngestionConfig
from mlops.utils.common_utils import (create_directory, read_dataset,
                                      save_file_as_csv)
from mlops.utils.data_ingestion_utils import perform_train_test_split
from src.logger.get_logger import get_logger


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        self.logger = get_logger()
        create_directory(self.config.artifact_dir)
        create_directory(self.config.best_run_dir)
        create_directory(self.config.runs_dir)
        create_directory(self.config.current_artifact_dir)
        create_directory(self.config.pipeline_steps_dir)
        create_directory(self.config.data_ingestion_dir)
        create_directory(self.config.raw_data_dir)
        create_directory(self.config.ingested_data_dir)

    def run_data_ingestion(self) -> DataIngestionArtifact:
        try:
            self.logger.info("Data Ingestion started.")
            df = read_dataset(self.config.data_path)

            save_file_as_csv(df, self.config.raw_data_path)

            train_df, test_df = perform_train_test_split(
                df, self.config.train_test_split_ratio, self.config.seed, self.logger
            )

            save_file_as_csv(train_df, self.config.train_file_path)
            save_file_as_csv(test_df, self.config.test_file_path)
            data_ingestion_artifact = DataIngestionArtifact(
                self.config.train_file_path, self.config.test_file_path
            )

            self.logger.info(f"Data Ingestion returns: {data_ingestion_artifact}")
            self.logger.info("Data Ingestion completed.")
            return data_ingestion_artifact
        except Exception as e:
            self.logger.error(f"Error occurred during data ingestion: {e}")
            raise e
