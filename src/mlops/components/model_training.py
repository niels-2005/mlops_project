from mlops.artifacts.data_transformation_artifact import \
    DataTransformationArtifact
from mlops.config.model_training_config import ModelTrainingConfig
from mlops.utils.common_utils import create_directory
from src.logger.get_logger import get_logger


class ModelTraining:
    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        config: ModelTrainingConfig,
    ):
        self.data_transformation_artifact = data_transformation_artifact
        self.config = config
        self.logger = get_logger()
        create_directory(self.config.model_training_dir)
        create_directory(self.config.logistic_regression_dir)
        create_directory(self.config.random_forest_dir)
        create_directory(self.config.xgboost_dir)

    def run_model_training(self):
        print("Testing")
