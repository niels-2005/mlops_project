import joblib
import yaml


def load_pipeline(
    file_path: str = "mlops_artifacts/best_run/pipeline_steps/pipeline.pkl",
):
    """
    Loads and returns a serialized ML pipeline from the specified file.

    Args:
        file_path (str): Path to the pipeline pickle file.

    Returns:
        Pipeline: Loaded machine learning pipeline object.

    Raises:
        Exception: Propagates exceptions during file read or deserialization.
    """
    try:
        with open(file_path, "rb") as file:
            return joblib.load(file)
    except Exception as e:
        raise e


def load_threshold(
    file_path: str = "mlops_artifacts/best_run/best_model_summary.yaml",
) -> float:
    """
    Loads the threshold value for classification from a YAML summary file.
    Returns default threshold 0.50 if none is specified.

    Args:
        file_path (str): Path to the YAML file with model summary.

    Returns:
        float: Threshold value for binary classification.

    Raises:
        Exception: Propagates exceptions during file read or parsing.
    """
    try:
        with open(file_path, "r") as yaml_file:
            content = yaml.safe_load(yaml_file)

            if content["best_model_summary"]["threshold_used"] == True:
                return content["best_model_summary"]["threshold"]
            else:
                return 0.50
    except Exception as e:
        raise e
