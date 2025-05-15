from src.mlops.utils.common_utils import (
    read_yaml_file,
    get_os_path,
    create_directory,
)
from datetime import datetime


class CommonConfig:
    def __init__(
        self,
        config_path="src/mlops/config/config.yaml",
        timestamp=datetime.now().strftime("%m_%d_%Y_%H_%M_%S"),
    ):
        self.config = read_yaml_file(config_path)
        self.timestamp = timestamp
        self.common_config = self.config["common"]
        self.artifact_dir = self.common_config["artifact_dir"]
        self.seed = self.common_config["seed"]
        self.current_artifact_dir = get_os_path(self.artifact_dir, self.timestamp)
