import pandas as pd
from sklearn.model_selection import train_test_split

from src.logger.get_logger import get_logger

logger = get_logger()


def perform_train_test_split(
    df: pd.DataFrame, train_test_split_ratio: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        train_df, test_df = train_test_split(
            df,
            test_size=train_test_split_ratio,
            random_state=seed,
        )
        logger.info(
            f"Train Test Split Successful, Train Shape: {train_df.shape}, Test Shape: {test_df.shape}"
        )
        return train_df, test_df
    except Exception as e:
        logger.error(f"Error occured at Train Test Split: {e}")
        raise e
