from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (fbeta_score, make_scorer, precision_score,
                             recall_score)
from sklearn.model_selection import (RandomizedSearchCV,
                                     TunedThresholdClassifierCV)
from sklearn.pipeline import Pipeline

from mlops.utils.common_utils import get_os_path, save_object, write_yaml_file


def save_pipeline_objects(best_estimator, training_paths: dict):
    save_object(best_estimator, training_paths["best_pipeline_path"])
    save_object(
        best_estimator.named_steps["feature_selector"],
        training_paths["feature_selector_path"],
    )
    save_object(
        best_estimator.named_steps["classifier"], training_paths["best_model_path"]
    )


def get_sklearn_estimator(model) -> Pipeline:
    return Pipeline([("feature_selector", SelectKBest()), ("classifier", model)])


def get_model_param_distributions(model_param_distributions_schema):
    return {f"classifier__{k}": v for k, v in model_param_distributions_schema.items()}


def get_param_distributions(model_param_distributions_schema, feature_selection_schema):
    model_param_distributions = get_model_param_distributions(
        model_param_distributions_schema
    )
    return {
        "feature_selector__score_func": [f_classif],
        "feature_selector__k": feature_selection_schema["param_distributions"]["k"],
        **model_param_distributions,
    }


def get_scoring_function(beta):
    return make_scorer(fbeta_score, beta=beta)


def perform_hyperparameter_tuning(
    random_search_schema, estimator, param_distributions, X_train, y_train, seed
):
    random_search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=random_search_schema["n_iter"],
        cv=random_search_schema["cv"],
        scoring=get_scoring_function(random_search_schema["fbeta"]),
        verbose=random_search_schema["verbose"],
        n_jobs=random_search_schema["n_jobs"],
        random_state=seed,
    ).fit(X_train, y_train)
    return random_search


def perform_threshold_tuning(
    threshold_tuning_schema, estimator, X_train, y_train, seed
):
    tuned_threshold = TunedThresholdClassifierCV(
        estimator=estimator,
        scoring=get_scoring_function(beta=threshold_tuning_schema["fbeta"]),
        response_method=threshold_tuning_schema["response_method"],
        thresholds=threshold_tuning_schema["thresholds"],
        cv=threshold_tuning_schema["cv"],
        n_jobs=threshold_tuning_schema["n_jobs"],
        random_state=seed,
    ).fit(X_train, y_train)
    return tuned_threshold.best_threshold_


def get_training_save_paths(config, model_name):
    model_dir = getattr(config, f"{model_name}_dir")
    return {
        "best_pipeline_path": get_os_path(model_dir, config.best_pipeline_path),
        "best_model_path": get_os_path(model_dir, config.best_model_path),
        "feature_selector_path": get_os_path(model_dir, config.feature_selector_path),
        "tuning_summary_path": get_os_path(model_dir, config.tuning_summary_path),
    }


def get_best_params(random_search: RandomizedSearchCV):
    best_params = random_search.best_params_
    best_params["feature_selector__score_func"] = f_classif.__name__
    return best_params


def get_selected_features(random_search: RandomizedSearchCV, X_train):
    feature_selector = random_search.best_estimator_.named_steps["feature_selector"]
    selected_features = X_train.columns[feature_selector.get_support()].tolist()
    return selected_features


def save_tuning_summary(
    random_search: RandomizedSearchCV, best_threshold, X_train, save_path: str
) -> None:
    best_params = get_best_params(random_search)

    selected_features = get_selected_features(random_search, X_train)

    content = {
        "best_fbeta_score": float(random_search.best_score_),
        "best_threshold": float(best_threshold),
        "best_params": best_params,
        "selected_features": selected_features,
    }
    write_yaml_file(save_path, content)
