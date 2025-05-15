from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from mlops.artifacts.data_ingestion_artifact import DataIngestionArtifact
from mlops.config.data_ingestion_config import DataIngestionConfig
from mlops.utils.common_utils import (create_directory, read_dataset,
                                      save_file_as_csv)
from src.logger.get_logger import get_logger

logger = get_logger()


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        create_directory(self.config.artifact_dir)
        create_directory(self.config.current_artifact_dir)
        create_directory(self.config.data_ingestion_dir)
        create_directory(self.config.raw_data_dir)
        create_directory(self.config.ingested_data_dir)

    def perform_train_test_split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        try:
            train_df, test_df = train_test_split(
                df,
                test_size=self.config.train_test_split_ratio,
                random_state=self.config.seed,
            )
            logger.info(
                f"Train Test Split Successful, Train Shape: {train_df.shape}, Test Shape: {test_df.shape}"
            )
            return train_df, test_df
        except Exception as e:
            logger.error(f"Error at Train Test Split: {e}")
            raise e

    def run_data_ingestion(self) -> DataIngestionArtifact:
        logger.info("Data Ingestion started.")
        df = read_dataset(self.config.data_path)
        save_file_as_csv(df, self.config.raw_data_path)
        train_df, test_df = self.perform_train_test_split(df)
        save_file_as_csv(train_df, self.config.train_file_path)
        save_file_as_csv(test_df, self.config.test_file_path)
        data_ingestion_artifact = DataIngestionArtifact(
            self.config.train_file_path, self.config.test_file_path
        )
        logger.info(f"Data Ingestion returns: {data_ingestion_artifact}")
        logger.info("Data Ingestion completed.")
        return data_ingestion_artifact
