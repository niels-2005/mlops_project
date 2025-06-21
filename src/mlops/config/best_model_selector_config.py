from mlops.utils.common_utils import get_os_path

from .common_config import CommonConfig


class BestModelSelectorConfig(CommonConfig):
    def __init__(self):
        """
        Configuration for best model selection.
        Includes paths and settings for MLflow registration and pipeline artifacts.
        """
        super().__init__()
        self.best_model_selector_config = self.config["best_model_selector"]
        self.registered_model_name = self.best_model_selector_config[
            "registered_model_name"
        ]
        self.input_example = self.best_model_selector_config["input_example"]
        self.output_example = self.best_model_selector_config["output_example"]
        self.mlflow_uri = self.common_config["mlflow_uri"]
        self.best_model_summary_path = get_os_path(
            self.best_run_dir, self.common_config["best_model_summary_path"]
        )
        self.feature_binning_pkl_path = get_os_path(
            self.pipeline_steps_dir, self.common_config["feature_binning_pkl_path"]
        )
        self.scaler_pkl_path = get_os_path(
            self.pipeline_steps_dir, self.common_config["scaler_pkl_path"]
        )
        self.feature_selector_pkl_path = get_os_path(
            self.pipeline_steps_dir, self.common_config["feature_selector_pkl_path"]
        )
        self.classifier_pkl_path = get_os_path(
            self.pipeline_steps_dir, self.common_config["classifier_pkl_path"]
        )
        self.pipeline_pkl_path = get_os_path(
            self.pipeline_steps_dir, self.common_config["pipeline_pkl_path"]
        )
