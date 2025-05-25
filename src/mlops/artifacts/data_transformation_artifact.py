from dataclasses import dataclass


@dataclass
class DataTransformationArtifact:
    transformed_train_path: str
    transformed_test_path: str
