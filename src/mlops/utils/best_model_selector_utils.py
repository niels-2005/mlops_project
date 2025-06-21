import shutil

import mlflow
from sklearn.pipeline import Pipeline

from logger.get_logger import get_logger
from mlops.utils.common_utils import load_object, read_yaml_file, save_object

logger = get_logger()


def get_pipeline(
    feature_binning_pkl_path,
    scaler_pkl_path,
    feature_selector_pkl_path,
    classifier_pkl_path,
):
    """
    Loads and reconstructs a machine learning pipeline from serialized pickle files.

    Args:
        feature_binning_pkl_path (str): Path to saved feature binning object.
        scaler_pkl_path (str): Path to saved scaler object.
        feature_selector_pkl_path (str): Path to saved feature selector object.
        classifier_pkl_path (str): Path to saved classifier object.

    Returns:
        sklearn.pipeline.Pipeline: Reconstructed pipeline.

    Raises:
        Exception: Propagates exceptions during loading.
    """
    try:
        logger.info("Loading pipeline steps from pickle files.")
        return Pipeline(
            steps=[
                ("feature_binning", load_object(feature_binning_pkl_path)),
                ("scaler", load_object(scaler_pkl_path)),
                ("feature_selector", load_object(feature_selector_pkl_path)),
                ("classifier", load_object(classifier_pkl_path)),
            ]
        )
    except Exception as e:
        logger.exception(f"Error occured loading pipeline steps: {e}")
        raise e


def save_pipeline(pipeline, pipeline_pkl_path):
    """
    Saves the given pipeline object to disk as a pickle file.

    Args:
        pipeline (sklearn.pipeline.Pipeline): Pipeline object to save.
        pipeline_pkl_path (str): Path to save the pipeline pickle file.

    Raises:
        Exception: Propagates exceptions during saving.
    """
    try:
        logger.info(f"Saving pipeline to {pipeline_pkl_path}")
        save_object(pipeline, pipeline_pkl_path)
    except Exception as e:
        logger.exception(f"Error saving pipeline: {e}")
        raise e


def register_model(
    pipeline,
    mlflow_uri,
    timestamp,
    registered_model_name,
    best_f2_score,
    best_recall_score,
    best_precision_score,
):
    """
    Registers a trained pipeline model in MLflow with metrics logged.

    Args:
        pipeline (sklearn.pipeline.Pipeline): The trained pipeline to register.
        mlflow_uri (str): URI of the MLflow tracking server.
        timestamp (str): Timestamp for the run name.
        registered_model_name (str): Name for the registered model in MLflow.
        best_f2_score (float): Best F2 score to log.
        best_recall_score (float): Best recall score to log.
        best_precision_score (float): Best precision score to log.

    Raises:
        Exception: Propagates exceptions during MLflow registration.
    """
    try:
        logger.info(f"Registering model {registered_model_name} in MLflow.")
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("registered_models")

        with mlflow.start_run(run_name=timestamp):
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                registered_model_name=registered_model_name,
            )
            mlflow.log_metric("f2_score", best_f2_score)
            mlflow.log_metric("recall", best_recall_score)
            mlflow.log_metric("precision", best_precision_score)
    except Exception as e:
        logger.exception(f"Error occured registering model: {e}")
        raise e


def promote_run(is_first_run: bool, promote_run_config: dict[str, any]):
    """
    Promotes the current run by saving the pipeline, copying artifacts to the best run directory,
    and registering the model in MLflow.

    Args:
        is_first_run (bool): Flag indicating if this is the first run.
        promote_run_config (dict): Configuration dictionary with paths and URIs for promotion.

    Raises:
        Exception: Propagates exceptions during promotion.
    """
    try:
        logger.info("Promoting run and saving pipeline.")
        pipeline = get_pipeline(
            promote_run_config["feature_binning_pkl_path"],
            promote_run_config["scaler_pkl_path"],
            promote_run_config["feature_selector_pkl_path"],
            promote_run_config["classifier_pkl_path"],
        )
        save_pipeline(pipeline, promote_run_config["pipeline_pkl_path"])

        current_artifact_dir = promote_run_config["current_artifact_dir"]
        best_run_dir = promote_run_config["best_run_dir"]

        if is_first_run:
            shutil.copytree(current_artifact_dir, best_run_dir)
        else:
            shutil.rmtree(best_run_dir)
            shutil.copytree(current_artifact_dir, best_run_dir)

        register_model(
            pipeline,
            promote_run_config["mlflow_uri"],
            promote_run_config["timestamp"],
            promote_run_config["registered_model_name"],
            promote_run_config["best_f2_score"],
            promote_run_config["best_recall_score"],
            promote_run_config["best_precision_score"],
        )
    except Exception as e:
        logger.exception(f"Error occured during run promotion: {e}")
        raise e


def compare_models(
    best_model_summary_path,
    f2_challenger_score,
    recall_challenger_score,
    precision_challenger_score,
):
    """
    Compares challenger model metrics against the current best model metrics to decide
    if the challenger is better.

    Args:
        best_model_summary_path (str): Path to YAML file with best model summary.
        f2_challenger_score (float): Challenger model's F2 score.
        recall_challenger_score (float): Challenger model's recall score.
        precision_challenger_score (float): Challenger model's precision score.

    Returns:
        bool: True if challenger outperforms the champion, else False.

    Raises:
        Exception: Propagates exceptions during comparison.
    """
    try:
        best_model_summary = read_yaml_file(best_model_summary_path)[
            "best_model_summary"
        ]
        f2_champion_score = best_model_summary["f2_score"]
        recall_champion_score = best_model_summary["recall"]
        precision_champion_score = best_model_summary["precision"]

        is_better = (
            f2_challenger_score > f2_champion_score
            and recall_challenger_score > recall_champion_score
            and precision_challenger_score > precision_champion_score
        )
        logger.info(
            f"Model comparison result: {'Challenger is better' if is_better else 'Champion remains'}"
        )
        return is_better
    except Exception as e:
        logger.exception(f"Error occured comparing models: {e}")
        raise e
