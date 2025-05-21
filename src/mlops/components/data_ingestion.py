from logging import Logger
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from mlops.artifacts.data_ingestion_artifact import DataIngestionArtifact
from mlops.config.data_ingestion_config import DataIngestionConfig
from mlops.utils.common_utils import (create_directories, read_dataset,
                                      save_file_as_csv, write_yaml_file)
from src.logger.get_logger import get_logger


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        self.logger = get_logger()
        create_directories(
            [
                self.config.artifact_dir,
                self.config.runs_dir,
                self.config.current_artifact_dir,
                self.config.pipeline_steps_dir,
                self.config.data_ingestion_dir,
                self.config.raw_data_dir,
                self.config.ingested_data_dir,
            ]
        )
        write_yaml_file(self.config.run_config_save_path, self.config.config)

    def perform_train_test_split(
        self, df: pd.DataFrame, train_test_split_ratio: float, seed: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        try:
            train_df, test_df = train_test_split(
                df,
                test_size=train_test_split_ratio,
                random_state=seed,
            )
            self.logger.info(
                f"Train Test Split Successful, Train Shape: {train_df.shape}, Test Shape: {test_df.shape}"
            )
            return train_df, test_df
        except Exception as e:
            self.logger.error(f"Error occured at Train Test Split: {e}")
            raise e

    def run_data_ingestion(self) -> DataIngestionArtifact:
        try:
            self.logger.info("Data Ingestion started.")
            df = read_dataset(self.config.data_path)

            save_file_as_csv(df, self.config.raw_data_path)

            train_df, test_df = self.perform_train_test_split(
                df, self.config.train_test_split_ratio, self.config.seed
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
