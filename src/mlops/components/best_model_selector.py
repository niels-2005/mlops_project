import os

from logger.get_logger import get_logger
from mlops.artifacts.model_evaluation_artifact import ModelEvaluationArtifact
from mlops.config.best_model_selector_config import BestModelSelectorConfig
from mlops.utils.best_model_selector_utils import compare_models, promote_run


class BestModelSelector:
    def __init__(
        self,
        model_evaluation_artifact: ModelEvaluationArtifact,
        config: BestModelSelectorConfig,
    ):
        """
        Initialize BestModelSelector with evaluation results and config.

        Args:
            model_evaluation_artifact (ModelEvaluationArtifact): Evaluation scores of the current model.
            config (BestModelSelectorConfig): Configuration for model promotion and paths.
        """
        self.model_evaluation_artifact = model_evaluation_artifact
        self.config = config
        self.logger = get_logger()
        self.current_artifact_dir = self.config.current_artifact_dir
        self.best_run_dir = self.config.best_run_dir
        self.pipeline_steps_dir = self.config.pipeline_steps_dir
        self.promote_run_config = {
            "feature_binning_pkl_path": self.config.feature_binning_pkl_path,
            "scaler_pkl_path": self.config.scaler_pkl_path,
            "feature_selector_pkl_path": self.config.feature_selector_pkl_path,
            "classifier_pkl_path": self.config.classifier_pkl_path,
            "pipeline_pkl_path": self.config.pipeline_pkl_path,
            "current_artifact_dir": self.config.current_artifact_dir,
            "best_run_dir": self.config.best_run_dir,
            "mlflow_uri": self.config.mlflow_uri,
            "timestamp": self.config.timestamp,
            "registered_model_name": self.config.registered_model_name,
            "best_f2_score": self.model_evaluation_artifact.best_f2_score,
            "best_recall_score": self.model_evaluation_artifact.best_recall_score,
            "best_precision_score": self.model_evaluation_artifact.best_precision_score,
        }

    def run_best_model_selector(self):
        """
        Run the best model selection process:
        - If no previous best model exists, promote current model.
        - Else, compare current model with best model.
        - Promote current model if it outperforms the best.

        Raises:
            Exception: If any error occurs during model selection.
        """
        try:
            self.logger.info("Best Model Selection started.")
            if not os.path.exists(self.best_run_dir):
                promote_run(
                    is_first_run=True, promote_run_config=self.promote_run_config
                )
            else:
                new_champion = compare_models(
                    self.config.best_model_summary_path,
                    self.model_evaluation_artifact.best_f2_score,
                    self.model_evaluation_artifact.best_recall_score,
                    self.model_evaluation_artifact.best_precision_score,
                )
                if new_champion:
                    self.logger.info(f"New Champion here! {self.config.timestamp}")
                    promote_run(
                        is_first_run=False, promote_run_config=self.promote_run_config
                    )
        except Exception as e:
            self.logger.exception(
                f"Error occured while running best model selection: {e}"
            )
            raise e
