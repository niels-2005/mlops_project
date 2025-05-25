from dataclasses import dataclass


@dataclass
class DataValidationArtifact:
    validated_train_path: str
    validated_test_path: str
