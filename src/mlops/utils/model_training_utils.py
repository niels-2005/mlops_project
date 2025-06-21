from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV, TunedThresholdClassifierCV
from sklearn.pipeline import Pipeline

from logger.get_logger import get_logger
from mlops.utils.common_utils import get_os_path, save_object, write_yaml_file

logger = get_logger()


def save_estimator_objects(best_estimator, training_paths: dict):
    """
    Save the estimator pipeline and its components (feature selector and classifier) to disk.

    Args:
        best_estimator (Pipeline): The trained sklearn pipeline estimator.
        training_paths (dict): Dictionary containing file paths for saving objects, keys include
                               'estimator_pkl_path', 'feature_selector_pkl_path', and 'model_pkl_path'.

    Raises:
        Exception: Logs and re-raises any exceptions during the saving process.
    """
    try:
        logger.info(f"Saving estimator objects to {training_paths}")
        save_object(best_estimator, training_paths["estimator_pkl_path"])
        save_object(
            best_estimator.named_steps["feature_selector"],
            training_paths["feature_selector_pkl_path"],
        )
        save_object(
            best_estimator.named_steps["classifier"], training_paths["model_pkl_path"]
        )
    except Exception as e:
        logger.exception(
            f"Error occured while saving estimator objects to {training_paths}: {e}"
        )
        raise e


def get_sklearn_estimator(model) -> Pipeline:
    """
    Create a sklearn Pipeline consisting of a feature selector and a classifier.

    Args:
        model: The sklearn classifier model to use as the final estimator step.

    Returns:
        Pipeline: sklearn pipeline with 'feature_selector' and 'classifier' steps.

    Raises:
        Exception: Logs and re-raises any exceptions during pipeline creation.
    """
    try:
        logger.info("Getting sklearn estimator")
        return Pipeline([("feature_selector", SelectKBest()), ("classifier", model)])
    except Exception as e:
        logger.exception(f"Error occured while getting sklearn estimator: {e}")
        raise e


def get_model_param_distributions(model_param_distributions_schema):
    """
    Prefix model parameter distribution keys with 'classifier__' for sklearn pipeline search.

    Args:
        model_param_distributions_schema (dict): Parameter distributions for the classifier.

    Returns:
        dict: Parameter distributions with keys prefixed by 'classifier__'.

    Raises:
        Exception: Logs and re-raises any exceptions during processing.
    """
    try:
        logger.info("Getting model param distributions")
        return {
            f"classifier__{k}": v for k, v in model_param_distributions_schema.items()
        }
    except Exception as e:
        logger.exception(f"Error occured while getting model param distributions: {e}")
        raise e


def get_param_distributions(model_param_distributions_schema, feature_selection_schema):
    """
    Combine feature selector parameter distributions with model classifier parameter distributions.

    Args:
        model_param_distributions_schema (dict): Classifier parameter distributions.
        feature_selection_schema (dict): Feature selector parameter distributions.

    Returns:
        dict: Combined parameter distributions for randomized search.

    Raises:
        Exception: Logs and re-raises any exceptions during processing.
    """
    try:
        logger.info("Getting randomized search param distributions")
        model_param_distributions = get_model_param_distributions(
            model_param_distributions_schema
        )
        return {
            "feature_selector__score_func": [f_classif],
            "feature_selector__k": feature_selection_schema["param_distributions"]["k"],
            **model_param_distributions,
        }
    except Exception as e:
        logger.exception(
            f"Error occured while getting randomized search param distributions: {e}"
        )
        raise e


def get_scoring_function(beta):
    """
    Create a sklearn scorer for the F-beta score with the specified beta.

    Args:
        beta (float): Beta parameter for the F-beta score.

    Returns:
        callable: Scorer function compatible with sklearn.

    Raises:
        Exception: Logs and re-raises any exceptions during scorer creation.
    """
    try:
        logger.info(f"Getting fbeta scoring function with beta={beta}")
        return make_scorer(fbeta_score, beta=beta)
    except Exception as e:
        logger.exception(
            f"Error occured while getting fbeta scoring function with beta={beta}: {e}"
        )


def perform_hyperparameter_tuning(
    random_search_schema, estimator, param_distributions, X_train, y_train, seed
):
    """
    Perform randomized hyperparameter tuning using RandomizedSearchCV.

    Args:
        random_search_schema (dict): Configuration schema for randomized search.
        estimator (Pipeline): Estimator pipeline to tune.
        param_distributions (dict): Parameter distributions to sample.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series or np.ndarray): Training labels.
        seed (int): Random seed for reproducibility.

    Returns:
        RandomizedSearchCV: Fitted randomized search object.

    Raises:
        Exception: Logs and re-raises any exceptions during hyperparameter tuning.
    """
    try:
        logger.info("Performing hyperparameter tuning")
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
    except Exception as e:
        logger.exception(f"Error occurred during hyperparameter tuning: {e}")
        raise e


def perform_threshold_tuning(
    threshold_tuning_schema, estimator, X_train, y_train, seed
):
    """
    Tune classification threshold to optimize the F-beta score using cross-validation.

    Args:
        threshold_tuning_schema (dict): Configuration for threshold tuning.
        estimator (Pipeline): Estimator pipeline to tune threshold for.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series or np.ndarray): Training labels.
        seed (int): Random seed for reproducibility.

    Returns:
        float: Best threshold found.

    Raises:
        Exception: Logs and re-raises any exceptions during threshold tuning.
    """
    try:
        logger.info("Performing threshold tuning")
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
    except Exception as e:
        logger.exception(f"Error occurred during threshold tuning: {e}")
        raise e


def get_training_save_paths(config, model_dir):
    """
    Construct absolute paths for saving training artifacts based on model directory and config.

    Args:
        config: Configuration object containing relative paths.
        model_dir (str): Base directory for model artifacts.

    Returns:
        dict: Paths for saving estimator, feature selector, classifier, and tuning summary.

    Raises:
        Exception: Logs and re-raises any exceptions during path construction.
    """
    try:
        logger.info(f"Getting training save paths for model_dir: {model_dir}")
        return {
            "estimator_pkl_path": get_os_path(model_dir, config.estimator_pkl_path),
            "model_pkl_path": get_os_path(model_dir, config.classifier_pkl_path),
            "feature_selector_pkl_path": get_os_path(
                model_dir, config.feature_selector_pkl_path
            ),
            "tuning_summary_path": get_os_path(model_dir, config.tuning_summary_path),
        }
    except Exception as e:
        logger.exception(f"Error occurred while getting training save paths: {e}")
        raise e


def get_best_params(random_search: RandomizedSearchCV):
    """
    Extract best parameters from fitted RandomizedSearchCV, formatting feature selector score_func.

    Args:
        random_search (RandomizedSearchCV): Fitted randomized search object.

    Returns:
        dict: Best parameters with 'feature_selector__score_func' as function name.

    Raises:
        Exception: Logs and re-raises any exceptions during extraction.
    """
    try:
        logger.info("Getting best parameters from RandomizedSearchCV")
        best_params = random_search.best_params_
        best_params["feature_selector__score_func"] = f_classif.__name__
        return best_params
    except Exception as e:
        logger.exception(f"Error occurred while getting best params: {e}")
        raise e


def get_selected_features(random_search: RandomizedSearchCV, X_train):
    """
    Retrieve the list of selected feature names from the best estimator's feature selector.

    Args:
        random_search (RandomizedSearchCV): Fitted randomized search object.
        X_train (pd.DataFrame): Training data features.

    Returns:
        list: List of selected feature names.

    Raises:
        Exception: Logs and re-raises any exceptions during retrieval.
    """
    try:
        logger.info("Getting selected features from feature selector")
        feature_selector = random_search.best_estimator_.named_steps["feature_selector"]
        selected_features = X_train.columns[feature_selector.get_support()].tolist()
        return selected_features
    except Exception as e:
        logger.exception(f"Error occurred while getting selected features: {e}")
        raise e


def save_tuning_summary(
    random_search: RandomizedSearchCV, best_threshold, X_train, save_path: str
) -> None:
    """
    Save a YAML file summarizing hyperparameter tuning results and selected features.

    Args:
        random_search (RandomizedSearchCV): Fitted randomized search object.
        best_threshold (float): Best classification threshold.
        X_train (pd.DataFrame): Training features.
        save_path (str): File path to save tuning summary YAML.

    Raises:
        Exception: Logs and re-raises any exceptions during saving.
    """
    try:
        logger.info(f"Saving tuning summary to {save_path}")
        best_params = get_best_params(random_search)
        selected_features = get_selected_features(random_search, X_train)

        write_yaml_file(
            save_path,
            content={
                "best_fbeta_score": float(random_search.best_score_),
                "best_threshold": float(best_threshold),
                "best_params": best_params,
                "selected_features": selected_features,
            },
        )
    except Exception as e:
        logger.exception(
            f"Error occurred while saving tuning summary to {save_path}: {e}"
        )
        raise e


def get_training_results(
    X_train,
    y_train,
    models: dict,
    models_schema,
    feature_selection_schema,
    random_search_schema,
    threshold_tuning_schema,
    config,
):
    """
    Train models, perform hyperparameter and threshold tuning, save artifacts and tuning summaries.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series or np.ndarray): Training labels.
        models (dict): Dictionary of model name to sklearn estimator.
        models_schema (dict): Schema containing parameter distributions per model.
        feature_selection_schema (dict): Schema for feature selector parameters.
        random_search_schema (dict): Configuration for randomized hyperparameter search.
        threshold_tuning_schema (dict): Configuration for threshold tuning.
        config: Configuration object containing paths and seed.

    Returns:
        tuple: Two dicts: best_estimators (model_name -> fitted estimator),
                          best_thresholds (model_name -> best threshold).

    Raises:
        Exception: Logs and re-raises any exceptions during training.
    """
    try:
        best_estimators = {}
        best_thresholds = {}
        for model_name, model in models.items():
            logger.info(f"Getting training results for: {model_name}")
            estimator = get_sklearn_estimator(model)

            # get param distributions for randomizedsearch
            param_distributions = get_param_distributions(
                models_schema[model_name]["param_distributions"],
                feature_selection_schema,
            )
            # get fitted RandomizedSearchCV Object
            random_search = perform_hyperparameter_tuning(
                random_search_schema,
                estimator,
                param_distributions,
                X_train,
                y_train,
                config.seed,
            )

            # find best threshold for given estimator
            best_estimator = random_search.best_estimator_
            best_treshold = perform_threshold_tuning(
                threshold_tuning_schema,
                best_estimator,
                X_train,
                y_train,
                config.seed,
            )

            # save current estimator objects (best_estimator, feature_selector, classifier)
            model_dir = getattr(config, f"{model_name}_dir")
            training_save_paths = get_training_save_paths(config, model_dir)
            save_estimator_objects(best_estimator, training_save_paths)

            # save current tuning summary for {model_name}
            save_tuning_summary(
                random_search,
                best_treshold,
                X_train,
                training_save_paths["tuning_summary_path"],
            )

            # save current training result to dict
            best_estimators[model_name] = best_estimator
            best_thresholds[model_name] = best_treshold
        return best_estimators, best_thresholds
    except Exception as e:
        logger.exception(f"Error occurred getting training results: {e}")
        raise e
