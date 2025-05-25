from mlops.artifacts.model_evaluation_artifact import ModelEvaluationArtifact
from mlops.artifacts.model_training_artifact import ModelTrainingArtifact
from mlops.config.model_evaluation_config import ModelEvaluationConfig
from mlops.utils.common_utils import (create_directories, get_X_y,
                                      read_dataset, read_yaml_file,
                                      write_yaml_file)
from mlops.utils.model_evaluation_utils import find_best_model_data

from mlops_src.logger.get_logger import get_logger


class ModelEvaluation:
    def __init__(
        self,
        model_training_artifact: ModelTrainingArtifact,
        config: ModelEvaluationConfig,
    ):
        self.model_training_artifact = model_training_artifact
        self.config = config
        self.logger = get_logger()
        create_directories(
            [
                self.config.model_evaluation_dir,
                self.config.logistic_regression_dir,
                self.config.random_forest_dir,
                self.config.xgboost_dir,
                self.config.catboost_dir,
                self.config.svc_dir,
                self.config.mlp_dir,
            ]
        )
        self.schema = read_yaml_file(self.config.schema_read_path)
        self.metrics_to_compute_schema = self.schema["metrics_to_compute"]
        write_yaml_file(self.config.schema_save_path, self.schema)
        self.estimators = self.model_training_artifact.best_estimators
        self.thresholds = self.model_training_artifact.best_thresholds

    def run_model_evaluation(self):
        try:
            self.logger.info("Model Evaluation started.")
            df_test = read_dataset(self.model_training_artifact.transformed_test_path)
            X_test, y_test = get_X_y(df_test, self.config.target_feature)
            best_model_data = find_best_model_data(
                self.config,
                self.estimators,
                self.metrics_to_compute_schema,
                self.thresholds,
                X_test,
                y_test,
            )

            model_evaluation_artifact = ModelEvaluationArtifact(
                best_f2_score=best_model_data["f2_score"],
                best_recall_score=best_model_data["recall"],
                best_precision_score=best_model_data["precision"],
            )
            self.logger.info(f"Model Evaluation returns: {model_evaluation_artifact}")
            self.logger.info("Model evaluation completed.")
            return model_evaluation_artifact
        except Exception as e:
            self.logger.error(f"Error occurred during model evaluation: {e}")
            raise e
