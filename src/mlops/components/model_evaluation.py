import os

from sklearn.metrics import (confusion_matrix, fbeta_score, precision_score,
                             recall_score, roc_auc_score)

from mlops.artifacts.model_training_artifact import ModelTrainingArtifact
from mlops.config.model_evaluation_config import ModelEvaluationConfig
from mlops.utils.common_utils import (create_directory, get_X_y, read_dataset,
                                      read_yaml_file, write_yaml_file)
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

    def compute_metrics(self, metrics_to_compute, y_test, y_pred):
        metrics = {}

        if metrics_to_compute["f2_score"]:
            f2_score = fbeta_score(y_test, y_pred, beta=2)
            metrics["f2_score"] = f2_score

        if metrics_to_compute["recall"]:
            recall = recall_score(y_test, y_pred)
            metrics["recall"] = recall

        if metrics_to_compute["precision"]:
            precision = precision_score(y_test, y_pred)
            metrics["precision"] = precision

        if metrics_to_compute["confusion_matrix"]:
            pass

        return metrics

    def run_model_evaluation(self):
        try:
            self.logger.info("Model Evaluation started.")
            df_test = read_dataset(self.model_training_artifact.transformed_test_path)
            X_test, y_test = get_X_y(df_test, self.config.target_feature)

            for model_name, estimator in self.estimators.items():
                y_pred = estimator.predict(X_test)
                y_proba = estimator.predict_proba(X_test)[:, 1]
                threshold = self.thresholds[model_name]
                y_pred_threshold = (y_proba >= threshold).astype(int)

                roc_auc = roc_auc_score(y_test, y_proba)

                metrics_without_threshold = self.compute_metrics(
                    self.metrics_to_compute_schema, y_test, y_pred
                )

                metrics_with_threshold = self.compute_metrics(
                    self.metrics_to_compute_schema, y_test, y_pred_threshold
                )

                model_dir = getattr(self.config, f"{model_name}_dir")
                evaluation_summary_path = os.path.join(
                    model_dir, self.config.evaluation_summary_path
                )

                content = {
                    "metrics_without_threshold": metrics_without_threshold,
                    "metrics_with_threshold": metrics_with_threshold,
                    "roc_auc_score": float(roc_auc),
                }

                write_yaml_file(evaluation_summary_path, content)

        except Exception as e:
            self.logger.error(f"Error occurred during model evaluation: {e}")
            raise e
