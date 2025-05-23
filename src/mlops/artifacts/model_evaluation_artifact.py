from dataclasses import dataclass


@dataclass
class ModelEvaluationArtifact:
    best_f2_score: float
    best_recall_score: float
    best_precision_score: float
