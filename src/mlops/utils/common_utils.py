import os

import pandas as pd
import yaml

from src.logger.get_logger import get_logger

logger = get_logger()


def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            logger.info(f"Sucessful readed yaml file from: {file_path}")
            return yaml.safe_load(yaml_file)
    except Exception as e:
        logger.error(e)
        raise e


def write_yaml_file(file_path: str, content: object) -> None:
    try:
        with open(file_path, "w") as file:
            logger.info(f"Sucessful saved yaml file in: {file_path}")
            yaml.dump(content, file)
    except Exception as e:
        logger.error(e)
        raise e


def get_os_path(str1, str2, str3=None):
    if str3 is not None:
        return os.path.join(str1, str2, str3)
    else:
        return os.path.join(str1, str2)


def create_directory(dir_name):
    logger.info(f"Created Directory at: {dir_name}")
    return os.makedirs(dir_name, exist_ok=True)


def read_dataset(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Dataset loaded: {file_path}, Dataset Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error while loading file {file_path}: {e}")
        raise e


def save_file_as_csv(df: pd.DataFrame, file_path: str) -> None:
    try:
        df.to_csv(file_path, index=False)
        logger.info(f"csv saved at: {file_path}")
    except Exception as e:
        logger.error(f"Error while saving csv {file_path}: {e} ")
        raise e
