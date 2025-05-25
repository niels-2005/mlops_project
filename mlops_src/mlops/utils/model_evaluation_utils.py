import itertools

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from mlops.utils.common_utils import get_os_path, save_object, write_yaml_file
from sklearn.metrics import (confusion_matrix, fbeta_score, precision_score,
                             recall_score, roc_auc_score)

from mlops_src.logger.get_logger import get_logger

logger = get_logger()


def make_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str,
    classes: np.ndarray = None,
    figsize: tuple[int, int] = (10, 10),
    text_size: int = 15,
    cmap: str = "Blues",
) -> None:
    try:
        logger.info("Creating confusion matrix plot")
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=figsize)
        cax = ax.matshow(cm, cmap=cmap)
        fig.colorbar(cax)

        labels = classes if classes is not None else np.arange(len(cm))
        ax.set(
            title="Confusion Matrix",
            xlabel="Predicted label",
            ylabel="True label",
            xticks=np.arange(len(labels)),
            yticks=np.arange(len(labels)),
            xticklabels=labels,
            yticklabels=labels,
        )
        ax.xaxis.set_label_position("bottom")
        ax.xaxis.tick_bottom()
        plt.xticks(rotation=70, fontsize=text_size)
        plt.yticks(fontsize=text_size)

        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(
                j,
                i,
                f"{cm[i, j]}",
                horizontalalignment="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                size=text_size,
            )

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Saved confusion matrix to {save_path}")
    except Exception as e:
        logger.exception(f"Error while creating confusion matrix: {e}")
        raise e


def evaluate_single_model(
    timestamp,
    metrics_to_compute_schema,
    estimator,
    X_test,
    y_test,
    threshold,
    model_dir,
    model_name,
):
    try:
        logger.info(f"Evaluating model: {model_name}")

        y_pred = estimator.predict(X_test)
        y_proba = estimator.predict_proba(X_test)[:, 1]
        y_pred_threshold = (y_proba >= threshold).astype(int)
        mlflow.set_experiment(timestamp)
        with mlflow.start_run(run_name=model_name):
            logger.info(
                f"Started MLflow run for model {model_name} with timestamp: {timestamp}"
            )

            mlflow.log_param("threshold", threshold)

            eval_metrics = {
                "roc_auc": float(roc_auc_score(y_test, y_proba)),
            }

            if metrics_to_compute_schema["f2_score"]:
                eval_metrics["f2_score"] = fbeta_score(y_test, y_pred, beta=2)
                eval_metrics["f2_score_threshold"] = fbeta_score(
                    y_test, y_pred_threshold, beta=2
                )

            if metrics_to_compute_schema["recall"]:
                eval_metrics["recall"] = recall_score(y_test, y_pred)
                eval_metrics["recall_threshold"] = recall_score(
                    y_test, y_pred_threshold
                )

            if metrics_to_compute_schema["precision"]:
                eval_metrics["precision"] = precision_score(y_test, y_pred)
                eval_metrics["precision_threshold"] = precision_score(
                    y_test, y_pred_threshold
                )

            mlflow.log_metrics(eval_metrics)

            if metrics_to_compute_schema["confusion_matrix"]:
                confusion_matrix_save_path = f"{model_dir}/confusion_matrix.png"
                confusion_matrix_save_path_threshold = (
                    f"{model_dir}/confusion_matrix_threshold.png"
                )

                make_confusion_matrix(y_test, y_pred, confusion_matrix_save_path)
                make_confusion_matrix(
                    y_test, y_pred_threshold, confusion_matrix_save_path_threshold
                )

                mlflow.log_artifact(
                    confusion_matrix_save_path, artifact_path="confusion_matrices"
                )
                mlflow.log_artifact(
                    confusion_matrix_save_path_threshold,
                    artifact_path="confusion_matrices",
                )

        # add important informations to eval_metrics for later purpose.
        eval_metrics["estimator"] = estimator
        eval_metrics["model_name"] = model_name
        eval_metrics["threshold"] = threshold
        return eval_metrics
    except Exception as e:
        logger.exception(f"Error during evaluation of model {model_name}: {e}")
        raise e


def update_best_model(eval_metrics, best_model_data, threshold):
    try:
        logger.info("Updating best model")
        f2_without_threshold = eval_metrics["f2_score"]
        f2_with_threshold = eval_metrics["f2_score_threshold"]
        f2_best = best_model_data["f2_score"]

        if f2_without_threshold > f2_best or f2_with_threshold > f2_best:
            if f2_with_threshold >= f2_without_threshold:
                best_model_data.update(
                    {
                        "f2_score": f2_with_threshold,
                        "recall": eval_metrics["recall_threshold"],
                        "precision": eval_metrics["precision_threshold"],
                        "estimator": eval_metrics["estimator"],
                        "model_name": eval_metrics["model_name"],
                        "threshold": float(threshold),
                        "threshold_used": True,
                    }
                )
            else:
                best_model_data.update(
                    {
                        "f2_score": f2_without_threshold,
                        "recall": eval_metrics["recall"],
                        "precision": eval_metrics["precision"],
                        "estimator": eval_metrics["estimator"],
                        "model_name": eval_metrics["model_name"],
                        "threshold": 0.5,
                        "threshold_used": False,
                    }
                )
        return best_model_data
    except Exception as e:
        logger.exception(f"Error updating best model: {e}")
        raise e


def save_evaluation_summary(evaluation_summary_path, model_dir, metrics):
    try:
        logger.info(f"Saving evaluation summary to {evaluation_summary_path}")
        evaluation_summary_path = get_os_path(model_dir, evaluation_summary_path)
        del metrics["estimator"]
        write_yaml_file(
            evaluation_summary_path, content={"evaluation_summary": metrics}
        )
    except Exception as e:
        logger.exception(f"Error saving evaluation summary: {e}")
        raise e


def finalize_best_model(config, best_model_data):
    try:
        logger.info("Finalizing best model")
        save_object(
            best_model_data["estimator"].named_steps["feature_selector"],
            config.feature_selector_pkl_path,
        )
        save_object(
            best_model_data["estimator"].named_steps["classifier"],
            config.classifier_pkl_path,
        )
        del best_model_data["estimator"]
        write_yaml_file(
            config.best_model_summary_path,
            content={"best_model_summary": best_model_data},
        )
    except Exception as e:
        logger.exception(f"Error finalizing best model: {e}")
        raise e


def find_best_model_data(
    config, estimators, metrics_to_compute_schema, thresholds, X_test, y_test
):
    try:
        logger.info("Finding best model among trained estimators")
        best_model_data = {
            "f2_score": 0,
            "recall": 0,
            "precision": 0,
            "estimator": None,
            "model_name": None,
            "threshold": 0.5,
            "threshold_used": False,
        }

        for model_name, estimator in estimators.items():
            model_dir = getattr(config, f"{model_name}_dir")
            eval_metrics = evaluate_single_model(
                config.timestamp,
                metrics_to_compute_schema,
                estimator,
                X_test,
                y_test,
                thresholds[model_name],
                model_dir,
                model_name,
            )
            best_model_data = update_best_model(
                eval_metrics, best_model_data, thresholds[model_name]
            )
            save_evaluation_summary(
                config.evaluation_summary_path,
                model_dir,
                eval_metrics,
            )

        finalize_best_model(config, best_model_data)
        return best_model_data
    except Exception as e:
        logger.exception(f"Error occurred while finding the best model: {e}")
        raise e
