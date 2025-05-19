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
        write_yaml_file(self.config.schema_save_path, self.schema)
        self.estimators = {
            "logistic_regression": self.model_training_artifact.logistic_regression_pipeline,
            "random_forest": self.model_training_artifact.random_forest_pipeline,
            "xgboost": self.model_training_artifact.xboost_pipeline,
        }

    def run_model_evaluation(self):
        try:
            self.logger.info("Model Evaluation started.")
            df_test = read_dataset(self.model_training_artifact.transformed_test_path)
            X_test, y_test = get_X_y(df_test, self.config.target_feature)

        except Exception as e:
            self.logger.error(f"Error occurred during model evaluation: {e}")
            raise e
