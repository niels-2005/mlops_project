from dataclasses import dataclass

from sklearn.pipeline import Pipeline


@dataclass
class ModelTrainingArtifact:
    transformed_test_path: str
    logistic_regression_estimator: Pipeline
    logistic_regression_best_threshold: float
    random_forest_estimator: Pipeline
    random_forest_best_threshold: float
    xgboost_estimator: Pipeline
    xgboost_best_threshold: float
