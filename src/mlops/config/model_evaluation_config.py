from mlops.utils.common_utils import get_os_path

from .common_config import CommonConfig


class ModelEvaluationConfig(CommonConfig):
    def __init__(self):
        super().__init__()
        self.model_evaluation_config = self.config["model_evaluation"]
        self.schema_read_path = self.model_evaluation_config["schema_read_path"]
        self.target_feature = self.model_evaluation_config["target_feature"]
        self.evaluation_summary_path = self.model_evaluation_config[
            "evaluation_summary_path"
        ]
        self.model_evaluation_dir = get_os_path(
            self.current_artifact_dir,
            self.model_evaluation_config["model_evaluation_dir"],
        )
        self.best_model_summary_path = get_os_path(
            self.current_artifact_dir,
            self.model_evaluation_config["best_model_summary_path"],
        )
        self.model_pkl_path = get_os_path(
            self.pipeline_steps_dir, self.common_config["model_pkl_path"]
        )
        self.feature_selector_pkl_path = get_os_path(
            self.pipeline_steps_dir,
            self.common_config["feature_selector_pkl_path"],
        )
        self.schema_save_path = get_os_path(
            self.model_evaluation_dir, self.model_evaluation_config["schema_save_path"]
        )
        self.logistic_regression_dir = get_os_path(
            self.model_evaluation_dir,
            self.model_evaluation_config["logistic_regression_dir"],
        )
        self.random_forest_dir = get_os_path(
            self.model_evaluation_dir, self.model_evaluation_config["random_forest_dir"]
        )
        self.xgboost_dir = get_os_path(
            self.model_evaluation_dir, self.model_evaluation_config["xgboost_dir"]
        )
        self.catboost_dir = get_os_path(
            self.model_evaluation_dir, self.model_evaluation_config["catboost_dir"]
        )
        self.svc_dir = get_os_path(
            self.model_evaluation_dir, self.model_evaluation_config["svc_dir"]
        )
        self.mlp_dir = get_os_path(
            self.model_evaluation_dir, self.model_evaluation_config["mlp_dir"]
        )
        self.sgd_dir = get_os_path(
            self.model_evaluation_dir, self.model_evaluation_config["sgd_dir"]
        )
