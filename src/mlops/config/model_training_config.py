from mlops.utils.common_utils import get_os_path

from .common_config import CommonConfig


class ModelTrainingConfig(CommonConfig):
    def __init__(self):
        super().__init__()
        self.model_training_config = self.config["model_training"]
        self.schema_path = self.model_training_config["schema_path"]
        self.model_training_dir = get_os_path(
            self.current_artifact_dir, self.model_training_config["model_training_dir"]
        )

        # === Logistic Regression ===
        self.logistic_regression_dir = get_os_path(
            self.model_training_dir,
            self.model_training_config["logistic_regression_dir"],
        )
        self.logistic_regression_model_path = get_os_path(
            self.logistic_regression_dir, self.model_training_config["best_model_path"]
        )
        self.logistic_regression_tuning_summary_path = get_os_path(
            self.logistic_regression_dir,
            self.model_training_config["tuning_summary_path"],
        )
        self.logistic_regression_feature_selector_path = get_os_path(
            self.logistic_regression_dir,
            self.model_training_config["feature_selector_path"],
        )

        # === Random Forest ===
        self.random_forest_dir = get_os_path(
            self.model_training_dir, self.model_training_config["random_forest_dir"]
        )
        self.random_forest_model_path = get_os_path(
            self.random_forest_dir, self.model_training_config["best_model_path"]
        )
        self.random_forest_tuning_summary_path = get_os_path(
            self.random_forest_dir, self.model_training_config["tuning_summary_path"]
        )
        self.random_forest_feature_selector_path = get_os_path(
            self.random_forest_dir, self.model_training_config["feature_selector_path"]
        )

        # === XGBoost ===
        self.xgboost_dir = get_os_path(
            self.model_training_dir, self.model_training_config["xgboost_dir"]
        )
        self.xgboost_model_path = get_os_path(
            self.xgboost_dir, self.model_training_config["best_model_path"]
        )
        self.xgboost_tuning_summary_path = get_os_path(
            self.xgboost_dir, self.model_training_config["tuning_summary_path"]
        )
        self.xgboost_feature_selector_path = get_os_path(
            self.xgboost_dir, self.model_training_config["feature_selector_path"]
        )
