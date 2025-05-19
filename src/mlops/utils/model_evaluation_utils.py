import itertools

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (confusion_matrix, fbeta_score, precision_score,
                             recall_score, roc_auc_score)

from mlops.utils.common_utils import get_os_path, save_object, write_yaml_file


def make_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_folder: str,
    threshold: bool = False,
    classes: np.ndarray = None,
    figsize: tuple[int, int] = (10, 10),
    text_size: int = 15,
    cmap: str = "Blues",
    norm: bool = False,
) -> None:
    """
    Makes a labelled confusion matrix comparing predictions and ground truth labels, with options to normalize
    and save the figure.

    Args:
      y_true (np.ndarray): Array of truth labels (must be same shape as y_pred).
      y_pred (np.ndarray): Array of predicted labels (must be same shape as y_true).
      classes (np.ndarray): Array of class labels (e.g., string form). If `None`, integer labels are used.
      figsize (tuple[int, int]): Size of output figure (default=(10, 10)).
      text_size (int): Size of output figure text (default=15).
      norm (bool): If True, normalize the values in the confusion matrix (default=False).
      savefig (bool): If True, save the confusion matrix plot to the current working directory (default=False).

    Returns:
        None: This function does not return a value but displays a Confusion Matrix. Optionally, it saves the plot.

    Example usage:
      make_confusion_matrix(y_true=test_labels, # ground truth test labels
                            y_pred=y_preds, # predicted labels
                            classes=class_names, # array of class label names
                            figsize=(15, 15),
                            text_size=10,
                            norm=True,
                            savefig=True)
    """
    # Create the confusion matrix
    cm = (
        confusion_matrix(y_true, y_pred, normalize="true")
        if norm
        else confusion_matrix(y_true, y_pred)
    )

    # Plot the figure
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm, cmap=cmap)
    fig.colorbar(cax)

    # Set class labels
    if classes is not None:
        labels = classes
    else:
        labels = np.arange(len(cm))

    # Set the labels and titles
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

    # Annotate the cells with the appropriate values
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            f"{cm[i, j]:.2f}" if norm else f"{cm[i, j]}",
            horizontalalignment="center",
            color="white" if cm[i, j] > cm.max() / 2 else "black",
            size=text_size,
        )

    plt.tight_layout()
    # Save the figure if requested
    threshold = "threshold" if threshold else ""
    plt.savefig(f"{save_folder}/confusions_matrix_{threshold}.png")
    plt.close()


def evaluate_single_model(
    metrics_to_compute, estimator, X_test, y_test, threshold, model_dir, model_name
):
    y_pred = estimator.predict(X_test)
    y_proba = estimator.predict_proba(X_test)[:, 1]
    y_pred_threshold = (y_proba >= threshold).astype(int)

    eval_metrics = {
        "estimator": estimator,
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "model_name": model_name,
    }

    if metrics_to_compute["f2_score"]:
        eval_metrics["f2_score"] = fbeta_score(y_test, y_pred, beta=2)
        eval_metrics["f2_score_threshold"] = fbeta_score(
            y_test, y_pred_threshold, beta=2
        )

    if metrics_to_compute["recall"]:
        eval_metrics["recall"] = recall_score(y_test, y_pred)
        eval_metrics["recall_threshold"] = recall_score(y_test, y_pred_threshold)

    if metrics_to_compute["precision"]:
        eval_metrics["precision"] = precision_score(y_test, y_pred)
        eval_metrics["precision_threshold"] = precision_score(y_test, y_pred_threshold)

    if metrics_to_compute["confusion_matrix"]:
        make_confusion_matrix(y_test, y_pred, model_dir)
        make_confusion_matrix(y_test, y_pred_threshold, model_dir, threshold=True)

    return eval_metrics


def update_best_model(eval_metrics, best_model_data, threshold):
    f2_without_threshold = eval_metrics["f2_score"]
    f2_with_threshold = eval_metrics["f2_score_threshold"]

    if (
        f2_without_threshold > best_model_data["f2_score"]
        or f2_with_threshold > best_model_data["f2_score"]
    ):
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


def save_evaluation_summary(evaluation_summary_path, model_dir, metrics):
    evaluation_summary_path = get_os_path(model_dir, evaluation_summary_path)
    del metrics["estimator"]
    content = {"evaluation_summary": metrics}
    write_yaml_file(evaluation_summary_path, content)


def finalize_best_model(config, best_model_data):
    save_object(
        best_model_data["estimator"].named_steps["feature_selector"],
        config.feature_selector_pkl_path,
    )
    save_object(
        best_model_data["estimator"].named_steps["classifier"],
        config.model_pkl_path,
    )
    del best_model_data["estimator"]
    content = {"best_model_summary": best_model_data}
    write_yaml_file(config.best_model_summary_path, content)
