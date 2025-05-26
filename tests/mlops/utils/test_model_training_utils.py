from sklearn.dummy import DummyClassifier

from mlops.utils.model_training_utils import (get_model_param_distributions,
                                              get_scoring_function,
                                              get_sklearn_estimator)


def test_get_sklearn_estimator_returns_pipeline():
    model = DummyClassifier()
    pipe = get_sklearn_estimator(model)
    assert hasattr(pipe, "named_steps")
    assert "feature_selector" in pipe.named_steps
    assert "classifier" in pipe.named_steps


def test_get_model_param_distributions():
    schema = {"param1": [1, 2], "param2": [3, 4]}
    dist = get_model_param_distributions(schema)
    assert "classifier__param1" in dist
    assert "classifier__param2" in dist


def test_get_scoring_function_returns_callable():
    scorer = get_scoring_function(beta=2)
    assert callable(scorer)
