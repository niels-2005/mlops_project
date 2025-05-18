from logging import Logger
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def perform_train_test_split(
    df: pd.DataFrame, train_test_split_ratio: float, seed: int, logger: Logger
) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
        logger.error(f"Error at Train Test Split: {e}")
        raise e
