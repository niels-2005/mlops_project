from dataclasses import dataclass
from sklearn.pipeline import Pipeline


@dataclass
class ModelTrainingArtifact:
    transformed_test_path: str
    logistic_regression_pipeline: Pipeline
    random_forest_pipeline: Pipeline
    xboost_pipeline: Pipeline
