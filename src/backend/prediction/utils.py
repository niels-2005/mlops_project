import joblib
import yaml


def load_pipeline(
    file_path: str = "mlops_artifacts/best_run/pipeline_steps/pipeline.pkl",
):
    try:
        with open(file_path, "rb") as file:
            return joblib.load(file)
    except Exception as e:
        raise e


def load_threshold(
    file_path: str = "mlops_artifacts/best_run/best_model_summary.yaml",
) -> float:
    try:
        with open(file_path, "r") as yaml_file:
            content = yaml.safe_load(yaml_file)  # YAML-Datei als dict laden

            if content["best_model_summary"]["threshold_used"] == True:
                return content["best_model_summary"]["threshold"]
            else:
                return 0.50
    except Exception as e:
        raise e
