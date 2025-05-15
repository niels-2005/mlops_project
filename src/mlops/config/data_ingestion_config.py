from mlops.utils.common_utils import get_os_path

from .common_config import CommonConfig


class DataIngestionConfig(CommonConfig):
    def __init__(self):
        super().__init__()
        self.data_ingestion_config = self.config["data_ingestion"]
        self.data_path = self.data_ingestion_config["data_path"]
        self.data_ingestion_dir = get_os_path(
            self.current_artifact_dir, self.data_ingestion_config["data_ingestion_dir"]
        )
        self.raw_data_dir = get_os_path(
            self.data_ingestion_dir, self.data_ingestion_config["raw_data_dir"]
        )
        self.raw_data_path = get_os_path(
            self.raw_data_dir, self.data_ingestion_config["raw_data_path"]
        )
        self.ingested_data_dir = get_os_path(
            self.data_ingestion_dir, self.data_ingestion_config["ingested_data_dir"]
        )
        self.train_file_path = get_os_path(
            self.ingested_data_dir, self.data_ingestion_config["train_file_path"]
        )
        self.test_file_path = get_os_path(
            self.ingested_data_dir, self.data_ingestion_config["test_file_path"]
        )
        self.train_test_split_ratio = self.data_ingestion_config[
            "train_test_split_ratio"
        ]
