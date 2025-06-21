from mlops.utils.common_utils import get_os_path

from .common_config import CommonConfig


class DataValidationConfig(CommonConfig):
    def __init__(self):
        """
        Configuration for data validation.
        Includes paths for schema, validation reports, and validated/invalidated data.
        """
        super().__init__()
        self.data_validation_config = self.config["data_validation"]
        self.schema_read_path = self.data_validation_config["schema_read_path"]
        self.data_validation_dir = get_os_path(
            self.current_artifact_dir,
            self.data_validation_config["data_validation_dir"],
        )
        self.schema_save_path = get_os_path(
            self.data_validation_dir, self.data_validation_config["schema_save_path"]
        )
        self.validation_reports_dir = get_os_path(
            self.data_validation_dir,
            self.data_validation_config["validation_reports_dir"],
        )
        self.validation_report_train_path = get_os_path(
            self.validation_reports_dir,
            self.data_validation_config["validation_report_train_path"],
        )
        self.validation_report_test_path = get_os_path(
            self.validation_reports_dir,
            self.data_validation_config["validation_report_test_path"],
        )
        self.validated_data_dir = get_os_path(
            self.data_validation_dir, self.data_validation_config["validated_data_dir"]
        )
        self.invalidated_data_dir = get_os_path(
            self.data_validation_dir,
            self.data_validation_config["invalidated_data_dir"],
        )
        self.validated_train_path = get_os_path(
            self.validated_data_dir, self.data_validation_config["valid_train_path"]
        )
        self.validated_test_path = get_os_path(
            self.validated_data_dir, self.data_validation_config["valid_test_path"]
        )
        self.invalidated_train_path = get_os_path(
            self.invalidated_data_dir, self.data_validation_config["invalid_train_path"]
        )
        self.invalidated_test_path = get_os_path(
            self.invalidated_data_dir, self.data_validation_config["invalid_test_path"]
        )
