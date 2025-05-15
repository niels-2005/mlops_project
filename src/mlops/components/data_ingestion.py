from src.mlops.config.data_ingestion_config import DataIngestionConfig
from src.mlops.utils.common_utils import create_directory
from src.logging.get_logger import get_logger
import pandas as pd
from sklearn.model_selection import train_test_split

logger = get_logger()


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        create_directory(self.config.artifact_dir)
        create_directory(self.config.current_artifact_dir)
        create_directory(self.config.data_ingestion_dir)
        create_directory(self.config.raw_data_dir)
        create_directory(self.config.ingested_data_dir)

    def save_file_as_csv(self, df, file_path):
        try:
            df.to_csv(file_path, index=False)
            logger.info(f"csv saved at: {file_path}")
        except Exception as e:
            logger.error(f"Error while saving csv {file_path}: {e} ")
            raise e

    def perform_train_test_split(self, df):
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

    def load_dataset(self):
        try:
            df = pd.read_csv(self.config.data_path)
            logger.info(
                f"Dataset loaded: {self.config.data_path}, Dataset Shape: {df.shape}"
            )
            return df
        except Exception as e:
            logger.error(f"Error while loading file {self.config.data_path}: {e}")
            raise

    def run_data_ingestion(self):
        logger.info("Data Ingestion started.")
        df = self.load_dataset()
        self.save_file_as_csv(df, self.config.raw_data_path)
        train_df, test_df = self.perform_train_test_split(df)
        self.save_file_as_csv(train_df, self.config.train_file_path)
        self.save_file_as_csv(test_df, self.config.test_file_path)
        logger.info("Data Ingestion completed.")
        logger.info(
            f"Data Ingestion returns: {self.config.train_file_path}, {self.config.test_file_path}"
        )
        return self.config.train_file_path, self.config.test_file_path
