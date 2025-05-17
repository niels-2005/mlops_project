from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (SelectKBest, f_classif,
                                       mutual_info_classif)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, make_scorer, precision_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from mlops.artifacts.data_transformation_artifact import \
    DataTransformationArtifact
from mlops.artifacts.model_training_artifact import ModelTrainingArtifact
from mlops.config.model_training_config import ModelTrainingConfig
from mlops.utils.common_utils import (create_directory, get_os_path,
                                      read_dataset, read_yaml_file,
                                      save_object, write_yaml_file)
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
        self.schema = read_yaml_file(self.config.schema_path)
        self.random_search_schema = self.schema["random_search"]
        self.feature_selection_schema = self.schema["feature_selection"]
        self.models_schema = self.schema["models"]
        self.models = {
            "logistic_regression": LogisticRegression(random_state=self.config.seed),
            "random_forest": RandomForestClassifier(random_state=self.config.seed),
            "xgboost": XGBClassifier(seed=self.config.seed),
        }

    def get_sklearn_pipeline(self, model) -> Pipeline:
        try:
            return Pipeline(
                [("feature_selector", SelectKBest()), ("classifier", model)]
            )
        except Exception as e:
            self.logger.error(f"Failed to create sklearn pipeline: {e}")
            raise e

    def get_model_param_distributions(self, model_name: str) -> dict:
        try:
            model_params = self.models_schema[model_name]["param_distributions"]
            return {f"classifier__{k}": v for k, v in model_params.items()}
        except Exception as e:
            self.logger.error(
                f"Failed to get parameter distribution for {model_name}: {e}"
            )
            raise e

    def get_param_distributions(self, model_param_distributions: dict) -> dict:
        try:
            return {
                "feature_selector__score_func": [f_classif, mutual_info_classif],
                "feature_selector__k": self.feature_selection_schema[
                    "param_distributions"
                ]["k"],
                **model_param_distributions,
            }
        except Exception as e:
            self.logger.error(f"Failed to construct parameter distribution: {e}")
            raise e

    def get_randomized_search(self, pipeline, param_distributions, X_train, y_train):
        try:
            random_search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=param_distributions,
                n_iter=self.random_search_schema["n_iter"],
                cv=self.random_search_schema["cv"],
                scoring=self.random_search_schema["scoring"],
                verbose=self.random_search_schema["verbose"],
                n_jobs=self.random_search_schema["n_jobs"],
                random_state=self.config.seed,
            )
            random_search.fit(X_train, y_train)
            self.logger.info(
                f"Randomized search completed for: {type(pipeline.named_steps['classifier']).__name__}"
            )
            return random_search
        except Exception as e:
            self.logger.error(f"Failed to create RandomizedSearchCV object: {e}")
            raise e

    def get_current_training_paths(self, model_name):
        try:
            model_dir = getattr(self.config, f"{model_name}_dir")
            return {
                "best_pipeline_path": get_os_path(
                    model_dir, self.config.best_pipeline_path
                ),
                "best_model_path": get_os_path(model_dir, self.config.best_model_path),
                "feature_selector_path": get_os_path(
                    model_dir, self.config.feature_selector_path
                ),
                "tuning_summary_path": get_os_path(
                    model_dir, self.config.tuning_summary_path
                ),
            }
        except Exception as e:
            self.logger.error(f"Failed getting Current Training Paths")
            raise e

    def save_current_training_objects(
        self, random_search: RandomizedSearchCV, training_paths: dict
    ):
        try:
            best_pipeline = random_search.best_estimator_
            save_object(best_pipeline, training_paths["best_pipeline_path"])
            feature_selector = best_pipeline.named_steps["feature_selector"]
            save_object(feature_selector, training_paths["feature_selector_path"])
            best_model = best_pipeline.named_steps["classifier"]
            save_object(best_model, training_paths["best_model_path"])
            return best_pipeline, feature_selector
        except Exception as e:
            self.logger.error(f"Error saving training objects: {e}")
            raise e

    def save_current_tuning_summary(
        self, random_search, X_train, feature_selector, training_paths
    ):
        try:
            best_recall_score = float(random_search.best_score_)
            best_params = random_search.best_params_
            best_params["feature_selector__score_func"] = f_classif.__name__
            selected_features = X_train.columns[feature_selector.get_support()].tolist()

            content = {
                "best_recall_score": best_recall_score,
                "best_params": best_params,
                "selected_features": selected_features,
            }
            write_yaml_file(training_paths["tuning_summary_path"], content)
        except Exception as e:
            self.logger.error(f"Error saving tuning summary: {e}")
            raise e

    def get_best_pipelines(self, X_train, y_train):
        try:
            best_pipelines = {}
            for model_name, model in self.models.items():
                self.logger.info(f"Running randomized search for: {model_name}")
                pipeline = self.get_sklearn_pipeline(model)
                model_param_distributions = self.get_model_param_distributions(
                    model_name
                )
                param_distributions = self.get_param_distributions(
                    model_param_distributions
                )

                random_search = self.get_randomized_search(
                    pipeline, param_distributions, X_train, y_train
                )
                training_paths = self.get_current_training_paths(model_name)

                best_pipeline, feature_selector = self.save_current_training_objects(
                    random_search, training_paths
                )
                best_pipelines[model_name] = best_pipeline
                self.save_current_tuning_summary(
                    random_search, X_train, feature_selector, training_paths
                )

            return best_pipelines
        except Exception as e:
            self.logger.error(f"Error occurred during best pipeline search: {e}")
            raise e

    def run_model_training(self):
        try:
            self.logger.info("Starting model training...")
            df_train = read_dataset(
                self.data_transformation_artifact.transformed_train_path
            )
            X_train = df_train.drop(self.config.target_feature, axis=1)
            y_train = df_train[self.config.target_feature]

            best_pipelines = self.get_best_pipelines(X_train, y_train)

            model_training_artifact = ModelTrainingArtifact(
                transformed_test_path=self.data_transformation_artifact.transformed_test_path,
                logistic_regression_pipeline=best_pipelines["logistic_regression"],
                random_forest_pipeline=best_pipelines["random_forest"],
                xboost_pipeline=best_pipelines["xgboost"],
            )
            self.logger.info(f"Model Training returns: {model_training_artifact}")
            self.logger.info("Model training completed.")
            return model_training_artifact
        except Exception as e:
            self.logger.error(f"Error occurred during model training: {e}")
            raise e
