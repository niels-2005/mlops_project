import joblib


def load_pipeline(
    file_path: str = "mlops_artifacts/best_run/pipeline_steps/pipeline.pkl",
):
    with open(file_path, "rb") as file:
        return joblib.load(file)
