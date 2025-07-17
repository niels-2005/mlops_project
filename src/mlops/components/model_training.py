from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from logger.get_logger import get_logger
from mlops.artifacts.data_transformation_artifact import DataTransformationArtifact
from mlops.artifacts.model_training_artifact import ModelTrainingArtifact
from mlops.config.model_training_config import ModelTrainingConfig
from mlops.utils.common_utils import (
    create_directories,
    get_X_y,
    read_dataset,
    read_yaml_file,
    write_yaml_file,
)
from mlops.utils.model_training_utils import get_training_results


class ModelTraining:
    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        config: ModelTrainingConfig,
    ):
        """
        Initialize ModelTraining with transformation artifact and config.
        Create required directories and load schema.
        Initialize models.
        """
        self.data_transformation_artifact = data_transformation_artifact
        self.config = config
        self.logger = get_logger()
        create_directories(
            [
                self.config.model_training_dir,
                self.config.logistic_regression_dir,
                self.config.random_forest_dir,
                self.config.xgboost_dir,
                self.config.catboost_dir,
                self.config.svc_dir,
                self.config.mlp_dir,
            ]
        )
        self.schema = read_yaml_file(self.config.schema_read_path)
        write_yaml_file(self.config.schema_save_path, self.schema)
        self.random_search_schema = self.schema["random_search"]
        self.threshold_tuning_schema = self.schema["threshold_tuning"]
        self.feature_selection_schema = self.schema["feature_selection"]
        self.models_schema = self.schema["models"]
        self.models = {
            "logistic_regression": LogisticRegression(random_state=self.config.seed),
            "random_forest": RandomForestClassifier(random_state=self.config.seed),
            "xgboost": XGBClassifier(seed=self.config.seed),
            "catboost": CatBoostClassifier(random_state=self.config.seed),
            "svc": SVC(random_state=self.config.seed, probability=True),
            "mlp": MLPClassifier(random_state=self.config.seed),
        }

    def run_model_training(self):
        """
        Run model training
        """
        try:
            self.logger.info("Starting model training...")
            df_train = read_dataset(
                self.data_transformation_artifact.transformed_train_path
            )
            X_train, y_train = get_X_y(df_train, self.config.target_feature)

            best_estimators, best_thresholds = get_training_results(
                X_train,
                y_train,
                self.models,
                self.models_schema,
                self.feature_selection_schema,
                self.random_search_schema,
                self.threshold_tuning_schema,
                self.config,
            )

            model_training_artifact = ModelTrainingArtifact(
                self.data_transformation_artifact.transformed_test_path,
                best_estimators,
                best_thresholds,
            )
            self.logger.info(f"Model Training returns: {model_training_artifact}")
            self.logger.info("Model training completed.")
            return model_training_artifact
        except Exception as e:
            self.logger.exception(f"Error occurred during model training: {e}")
            raise e
