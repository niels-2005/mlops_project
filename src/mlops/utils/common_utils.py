from src.logging.get_logger import get_logger
import yaml
import os

logger = get_logger()


def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            logger.info(f"Sucessful readed yaml file from: {file_path}")
            return yaml.safe_load(yaml_file)
    except Exception as e:
        logger.error(e)
        raise e


def get_os_path(str1, str2, str3=None):
    if str3 is not None:
        return os.path.join(str1, str2, str3)
    else:
        return os.path.join(str1, str2)


def create_directory(dir_name):
    return os.makedirs(dir_name, exist_ok=True)
