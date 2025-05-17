from mlops.utils.common_utils import get_os_path

from .common_config import CommonConfig


class ModelTrainingConfig(CommonConfig):
    def __init__(self):
        super().__init__()
        self.model_training_config = self.config["model_training"]
        self.schema_path = self.model_training_config["schema_path"]
        self.best_pipeline_path = self.model_training_config["best_pipeline_path"]
        self.best_model_path = self.model_training_config["best_model_path"]
        self.tuning_summary_path = self.model_training_config["tuning_summary_path"]
        self.feature_selector_path = self.model_training_config["feature_selector_path"]
        self.target_feature = self.model_training_config["target_feature"]
        self.model_training_dir = get_os_path(
            self.current_artifact_dir, self.model_training_config["model_training_dir"]
        )
        self.logistic_regression_dir = get_os_path(
            self.model_training_dir,
            self.model_training_config["logistic_regression_dir"],
        )
        self.random_forest_dir = get_os_path(
            self.model_training_dir, self.model_training_config["random_forest_dir"]
        )
        self.xgboost_dir = get_os_path(
            self.model_training_dir, self.model_training_config["xgboost_dir"]
        )
