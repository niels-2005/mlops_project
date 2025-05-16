from .common_config import CommonConfig
from mlops.utils.common_utils import get_os_path


class DataTransformationConfig(CommonConfig):
    def __init__(self):
        super().__init__()
        self.data_transformation_config = self.config["data_transformation"]
        self.schema_path = self.data_transformation_config["schema_path"]
        self.data_transformation_dir = get_os_path(
            self.current_artifact_dir,
            self.data_transformation_config["data_transformation_dir"],
        )
        self.preprocessors_dir = get_os_path(
            self.data_transformation_dir,
            self.data_transformation_config["preprocessor_objects_dir"],
        )
        self.standard_scaler_path = get_os_path(
            self.preprocessors_dir,
            self.data_transformation_config["standard_scaler_path"],
        )
        self.transformed_data_dir = get_os_path(
            self.data_transformation_dir,
            self.data_transformation_config["transformed_data_dir"],
        )
        self.transformed_train_path = get_os_path(
            self.transformed_data_dir,
            self.data_transformation_config["transformed_train_path"],
        )
        self.transformed_test_path = get_os_path(
            self.transformed_data_dir,
            self.data_transformation_config["transformed_test_path"],
        )
