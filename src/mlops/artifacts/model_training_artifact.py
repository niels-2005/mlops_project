from dataclasses import dataclass

from sklearn.pipeline import Pipeline


@dataclass
class ModelTrainingArtifact:
    transformed_test_path: str
    best_estimators: dict
    best_thresholds: dict
