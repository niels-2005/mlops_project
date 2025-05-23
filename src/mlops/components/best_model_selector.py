import os
import shutil

import mlflow
from mlflow.models.signature import infer_signature
from sklearn.pipeline import Pipeline

from mlops.artifacts.model_evaluation_artifact import ModelEvaluationArtifact
from mlops.config.best_model_selector_config import BestModelSelectorConfig
from mlops.utils.common_utils import load_object, read_yaml_file, save_object
from src.logger.get_logger import get_logger


class BestModelSelector:
    def __init__(
        self,
        model_evaluation_artifact: ModelEvaluationArtifact,
        config: BestModelSelectorConfig,
    ):
        self.model_evaluation_artifact = model_evaluation_artifact
        self.config = config
        self.logger = get_logger()
        self.current_artifact_dir = self.config.current_artifact_dir
        self.best_run_dir = self.config.best_run_dir
        self.pipeline_steps_dir = self.config.pipeline_steps_dir

    def get_pipeline(self):
        return Pipeline(
            steps=[
                ("feature_binning", load_object(self.config.feature_binning_pkl_path)),
                ("scaler", load_object(self.config.scaler_pkl_path)),
                (
                    "feature_selector",
                    load_object(self.config.feature_selector_pkl_path),
                ),
                ("classifier", load_object(self.config.classifier_pkl_path)),
            ]
        )

    def save_pipeline(self, pipeline):
        save_object(pipeline, self.config.pipeline_pkl_path)

    def register_model(self, pipeline):
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        mlflow.set_experiment("registered_models")

        with mlflow.start_run(run_name=self.config.timestamp):
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                registered_model_name=self.config.registered_model_name,
            )
            mlflow.log_metric("f2_score", self.model_evaluation_artifact.best_f2_score)
            mlflow.log_metric(
                "recall", self.model_evaluation_artifact.best_recall_score
            )
            mlflow.log_metric(
                "precision", self.model_evaluation_artifact.best_precision_score
            )

    def promote_run(self, is_first_run: bool):
        pipeline = self.get_pipeline()
        self.save_pipeline(pipeline)

        if is_first_run:
            shutil.copytree(self.current_artifact_dir, self.best_run_dir)
        else:
            shutil.rmtree(self.best_run_dir)
            shutil.copytree(self.current_artifact_dir, self.best_run_dir)

        self.register_model(pipeline)

    def compare_models(self):
        best_model_summary = read_yaml_file(self.config.best_model_summary_path)[
            "best_model_summary"
        ]
        f2_champion_score = best_model_summary["f2_score"]
        recall_champion_score = best_model_summary["recall"]
        precision_champion_score = best_model_summary["precision"]

        f2_challenger_score = self.model_evaluation_artifact.best_f2_score
        recall_challenger_score = self.model_evaluation_artifact.best_recall_score
        precision_challenger_score = self.model_evaluation_artifact.best_precision_score

        if (
            f2_challenger_score > f2_champion_score
            and recall_challenger_score > recall_champion_score
            and precision_challenger_score > precision_champion_score
        ):
            return True
        return False

    def run_best_model_selector(self):
        try:
            self.logger.info("Best Model Selection started.")
            if not os.path.exists(self.best_run_dir):
                self.promote_run(is_first_run=True)
            else:
                new_champion = self.compare_models()
                if new_champion:
                    self.logger.info(f"New Champion here! {self.config.timestamp}")
                    self.promote_run(is_first_run=False)
        except Exception as e:
            self.logger.exception(
                f"Error occured while running best model selection: {e}"
            )
            raise e
