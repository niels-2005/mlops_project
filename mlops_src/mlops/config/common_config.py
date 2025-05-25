from datetime import datetime

from mlops.utils.common_utils import get_os_path, read_yaml_file


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
        self.best_run_dir = get_os_path(
            self.artifact_dir, self.common_config["best_run_dir"]
        )
        self.runs_dir = get_os_path(self.artifact_dir, self.common_config["runs_dir"])
        self.current_artifact_dir = get_os_path(self.runs_dir, self.timestamp)
        self.pipeline_steps_dir = get_os_path(
            self.current_artifact_dir, self.common_config["pipeline_steps_dir"]
        )
        self.run_config_save_path = get_os_path(
            self.current_artifact_dir, self.common_config["run_config_save_path"]
        )
