from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from mlops.artifacts.data_transformation_artifact import \
    DataTransformationArtifact
from mlops.artifacts.model_training_artifact import ModelTrainingArtifact
from mlops.config.model_training_config import ModelTrainingConfig
from mlops.utils.common_utils import (create_directory, get_X_y, read_dataset,
                                      read_yaml_file, write_yaml_file)
from mlops.utils.model_training_utils import (get_param_distributions,
                                              get_sklearn_estimator,
                                              get_training_save_paths,
                                              perform_hyperparameter_tuning,
                                              perform_threshold_tuning,
                                              save_pipeline_objects,
                                              save_tuning_summary)
from src.logger.get_logger import get_logger


class ModelTraining:
    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        config: ModelTrainingConfig,
    ):
        self.data_transformation_artifact = data_transformation_artifact
        self.config = config
        self.logger = get_logger()
        create_directory(self.config.model_training_dir)
        create_directory(self.config.logistic_regression_dir)
        create_directory(self.config.random_forest_dir)
        create_directory(self.config.xgboost_dir)
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
        }

    def get_best_estimators(self, X_train, y_train):
        try:
            best_estimators = {}
            best_thresholds = {}
            for model_name, model in self.models.items():
                self.logger.info(f"Running randomized search for: {model_name}")
                estimator = get_sklearn_estimator(model)
                param_distributions = get_param_distributions(
                    self.models_schema[model_name]["param_distributions"],
                    self.feature_selection_schema,
                )
                random_search = perform_hyperparameter_tuning(
                    self.random_search_schema,
                    estimator,
                    param_distributions,
                    X_train,
                    y_train,
                    self.config.seed,
                )

                best_estimator = random_search.best_estimator_

                best_treshold = perform_threshold_tuning(
                    self.threshold_tuning_schema,
                    best_estimator,
                    X_train,
                    y_train,
                    self.config.seed,
                )

                training_save_paths = get_training_save_paths(self.config, model_name)
                save_pipeline_objects(best_estimator, training_save_paths)

                save_tuning_summary(
                    random_search,
                    best_treshold,
                    X_train,
                    training_save_paths["tuning_summary_path"],
                )

                best_estimators[model_name] = best_estimator
                best_thresholds[model_name] = best_treshold
            return best_estimators, best_thresholds
        except Exception as e:
            self.logger.error(f"Error occurred during best estimator search: {e}")
            raise e

    def run_model_training(self):
        try:
            self.logger.info("Starting model training...")
            df_train = read_dataset(
                self.data_transformation_artifact.transformed_train_path
            )
            X_train, y_train = get_X_y(df_train, self.config.target_feature)

            best_estimators, best_thresholds = self.get_best_estimators(
                X_train, y_train
            )

            model_training_artifact = ModelTrainingArtifact(
                transformed_test_path=self.data_transformation_artifact.transformed_test_path,
                logistic_regression_estimator=best_estimators["logistic_regression"],
                logistic_regression_best_threshold=best_thresholds[
                    "logistic_regression"
                ],
                random_forest_estimator=best_estimators["random_forest"],
                random_forest_best_threshold=best_thresholds["random_forest"],
                xgboost_estimator=best_estimators["xgboost"],
                xgboost_best_threshold=best_thresholds["xgboost"],
            )
            self.logger.info(f"Model Training returns: {model_training_artifact}")
            self.logger.info("Model training completed.")
            return model_training_artifact
        except Exception as e:
            self.logger.error(f"Error occurred during model training: {e}")
            raise e
