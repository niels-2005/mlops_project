import logging
import logging.config
import os
from datetime import datetime

import yaml

_logger = None


def get_logger(name="ml_logger", config_path="src/logger/logging_config.yaml"):
    global _logger

    if _logger is None:
        # logger configuration
        log_folder = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        logs_dir = os.path.join(os.getcwd(), "mlops_logs", log_folder)
        os.makedirs(logs_dir, exist_ok=True)
        log_file_path = os.path.join(logs_dir, f"{log_folder}.log")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # replace log file
        config["handlers"]["file_handler"]["filename"] = log_file_path

        logging.config.dictConfig(config)
        _logger = logging.getLogger(name)

    return _logger
