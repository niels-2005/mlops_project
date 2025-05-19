import os
import shutil

from mlops.artifacts.model_training_artifact import ModelTrainingArtifact
from mlops.config.model_evaluation_config import ModelEvaluationConfig
from mlops.utils.common_utils import (create_directory, get_X_y, read_dataset,
                                      read_yaml_file, write_yaml_file)
from mlops.utils.model_evaluation_utils import (evaluate_single_model,
                                                finalize_best_model,
                                                save_evaluation_summary,
                                                update_best_model)
from src.logger.get_logger import get_logger


class ModelEvaluation:
    def __init__(
        self,
        model_training_artifact: ModelTrainingArtifact,
        config: ModelEvaluationConfig,
    ):
        self.model_training_artifact = model_training_artifact
        self.config = config
        self.logger = get_logger()
        create_directory(self.config.model_evaluation_dir)
        create_directory(self.config.logistic_regression_dir)
        create_directory(self.config.random_forest_dir)
        create_directory(self.config.xgboost_dir)
        self.schema = read_yaml_file(self.config.schema_read_path)
        self.metrics_to_compute_schema = self.schema["metrics_to_compute"]
        write_yaml_file(self.config.schema_save_path, self.schema)
        self.estimators = {
            "logistic_regression": self.model_training_artifact.logistic_regression_estimator,
            "random_forest": self.model_training_artifact.random_forest_estimator,
            "xgboost": self.model_training_artifact.xgboost_estimator,
        }
        self.thresholds = {
            "logistic_regression": self.model_training_artifact.logistic_regression_best_threshold,
            "random_forest": self.model_training_artifact.random_forest_best_threshold,
            "xgboost": self.model_training_artifact.xgboost_best_threshold,
        }

    def run_model_evaluation(self):
        try:
            self.logger.info("Model Evaluation started.")
            df_test = read_dataset(self.model_training_artifact.transformed_test_path)
            X_test, y_test = get_X_y(df_test, self.config.target_feature)

            best_model_data = {
                "f2_score": 0,
                "recall": 0,
                "precision": 0,
                "estimator": None,
                "model_name": None,
                "threshold": 0.5,
                "threshold_used": False,
            }

            for model_name, estimator in self.estimators.items():
                model_dir = getattr(self.config, f"{model_name}_dir")
                eval_metrics = evaluate_single_model(
                    self.metrics_to_compute_schema,
                    estimator,
                    X_test,
                    y_test,
                    self.thresholds[model_name],
                    model_dir,
                    model_name,
                )
                best_model_data = update_best_model(
                    eval_metrics, best_model_data, self.thresholds[model_name]
                )
                save_evaluation_summary(
                    self.config.evaluation_summary_path,
                    model_dir,
                    eval_metrics,
                )

            finalize_best_model(self.config, best_model_data)

            if not os.path.exists(self.config.best_run_dir):
                destination_dir = os.path.join(
                    self.config.best_run_dir, self.config.timestamp
                )
                if not os.path.exists(destination_dir):
                    shutil.copytree(self.config.current_artifact_dir, destination_dir)

        except Exception as e:
            self.logger.error(f"Error occurred during model evaluation: {e}")
            raise e
