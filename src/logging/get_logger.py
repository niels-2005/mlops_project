import logging
import logging.config
import os
from datetime import datetime
from src.mlops.utils.common_utils import read_yaml_file

_logger = None


def get_logger(name="ml_logger", config_path="src/logging/logging_config.yaml"):
    global _logger

    if _logger is None:
        # logger configuration
        log_folder = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        logs_dir = os.path.join(os.getcwd(), "logs", log_folder)
        os.makedirs(logs_dir, exist_ok=True)
        log_file_path = os.path.join(logs_dir, f"{log_folder}.log")

        config = read_yaml_file(config_path)

        # replace log file
        config["handlers"]["file_handler"]["filename"] = log_file_path

        logging.config.dictConfig(config)
        _logger = logging.getLogger(name)

    return _logger
