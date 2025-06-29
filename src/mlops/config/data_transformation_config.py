from mlops.utils.common_utils import get_os_path

from .common_config import CommonConfig


class DataTransformationConfig(CommonConfig):
    def __init__(self):
        """
        Configuration for data transformation.
        Contains paths for schema, preprocessors, and transformed datasets.
        """
        super().__init__()
        self.data_transformation_config = self.config["data_transformation"]
        self.schema_read_path = self.data_transformation_config["schema_read_path"]
        self.data_transformation_dir = get_os_path(
            self.current_artifact_dir,
            self.data_transformation_config["data_transformation_dir"],
        )
        self.schema_save_path = get_os_path(
            self.data_transformation_dir,
            self.data_transformation_config["schema_save_path"],
        )
        self.preprocessors_dir = get_os_path(
            self.data_transformation_dir,
            self.data_transformation_config["preprocessor_objects_dir"],
        )
        self.scaler_artifact_path = get_os_path(
            self.preprocessors_dir,
            self.common_config["scaler_pkl_path"],
        )
        self.scaler_inference_path = get_os_path(
            self.pipeline_steps_dir,
            self.common_config["scaler_pkl_path"],
        )
        self.feature_binning_artifact_path = get_os_path(
            self.preprocessors_dir,
            self.common_config["feature_binning_pkl_path"],
        )
        self.feature_binning_inference_path = get_os_path(
            self.pipeline_steps_dir,
            self.common_config["feature_binning_pkl_path"],
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
