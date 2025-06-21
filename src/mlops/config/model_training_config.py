from mlops.utils.common_utils import get_os_path

from .common_config import CommonConfig


class ModelTrainingConfig(CommonConfig):
    def __init__(self):
        """
        Configuration for model training.
        Holds paths for training outputs, model artifacts, and tuning summaries.
        """
        super().__init__()
        self.model_training_config = self.config["model_training"]
        self.schema_read_path = self.model_training_config["schema_read_path"]
        self.tuning_summary_path = self.model_training_config["tuning_summary_path"]
        self.estimator_pkl_path = self.common_config["estimator_pkl_path"]
        self.classifier_pkl_path = self.common_config["classifier_pkl_path"]
        self.feature_selector_pkl_path = self.common_config["feature_selector_pkl_path"]
        self.target_feature = self.model_training_config["target_feature"]
        self.model_training_dir = get_os_path(
            self.current_artifact_dir, self.model_training_config["model_training_dir"]
        )
        self.schema_save_path = get_os_path(
            self.model_training_dir, self.model_training_config["schema_save_path"]
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
        self.catboost_dir = get_os_path(
            self.model_training_dir, self.model_training_config["catboost_dir"]
        )
        self.svc_dir = get_os_path(
            self.model_training_dir, self.model_training_config["svc_dir"]
        )
        self.mlp_dir = get_os_path(
            self.model_training_dir, self.model_training_config["mlp_dir"]
        )
