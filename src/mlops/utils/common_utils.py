import os

import joblib
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
        logger.exception(f"Error while reading yaml file for {file_path}: {e}")
        raise e


def write_yaml_file(file_path: str, content: object) -> None:
    try:
        with open(file_path, "w") as file:
            logger.info(f"Sucessful saved yaml file in: {file_path}")
            yaml.dump(content, file, sort_keys=False)
    except Exception as e:
        logger.exception(f"Error while writing yaml file for {file_path}: {e}")
        raise e


def get_os_path(str1, str2):
    logger.info(f"Returning Path: {str1}/{str2}")
    return os.path.join(str1, str2)


def create_directories(dir_list):
    for dir in dir_list:
        logger.info(f"Created Directory at: {dir}")
        os.makedirs(dir, exist_ok=True)


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


def save_object(object, file_path: str):
    try:
        logger.info(f"Saving Object to: {file_path}")
        with open(file_path, "wb") as file:
            joblib.dump(object, file)
    except Exception as e:
        logger.error(f"Error while saving Object to {file_path}: {e}")
        raise e


def load_object(file_path: str):
    try:
        logger.info(f"Loading Object from: {file_path}")
        with open(file_path, "rb") as file:
            return joblib.load(file)
    except Exception as e:
        logger.error(f"Error while loading Object from {file_path}: {e}")
        raise e


def get_X_y(df: pd.DataFrame, target_feature: str) -> tuple[pd.DataFrame, pd.Series]:
    return df.drop(target_feature, axis=1), df[target_feature]
