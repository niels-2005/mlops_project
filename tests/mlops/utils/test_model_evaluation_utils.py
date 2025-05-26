import pytest

from mlops.utils.model_evaluation_utils import update_best_model


def test_update_best_model_threshold_used():
    eval_metrics = {
        "f2_score": 0.6,
        "f2_score_threshold": 0.7,
        "recall_threshold": 0.9,
        "precision_threshold": 0.8,
        "recall": 0.6,
        "precision": 0.5,
        "estimator": "est",
        "model_name": "model_a",
    }

    best_model_data = {
        "f2_score": 0.5,
        "recall": 0,
        "precision": 0,
        "estimator": None,
        "model_name": None,
        "threshold": 0.5,
        "threshold_used": False,
    }

    updated = update_best_model(eval_metrics, best_model_data, 0.6)
    assert updated["f2_score"] == 0.7
    assert updated["threshold_used"] is True
