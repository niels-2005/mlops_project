import shutil

import mlflow
from mlops.utils.common_utils import load_object, read_yaml_file, save_object
from sklearn.pipeline import Pipeline

from mlops_src.logger.get_logger import get_logger

logger = get_logger()


def get_pipeline(
    feature_binning_pkl_path,
    scaler_pkl_path,
    feature_selector_pkl_path,
    classifier_pkl_path,
):
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
