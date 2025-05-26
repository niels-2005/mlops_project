import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from mlops.utils.data_transformation_utils import (drop_duplicates,
                                                   drop_null_values,
                                                   get_scaler)


def test_drop_null_values():
    df = pd.DataFrame({"A": [1, None, 3], "B": [4, 5, None]})
    result = drop_null_values(df, "train")
    assert result.shape[0] == 1


def test_drop_duplicates():
    df = pd.DataFrame({"A": [1, 1, 2], "B": [3, 3, 4]})
    result = drop_duplicates(df, "train")
    assert result.shape[0] == 2


def test_get_scaler_returns_correct_instance():
    assert isinstance(get_scaler("standard_scaler"), StandardScaler)
    assert isinstance(get_scaler("min_max_scaler"), MinMaxScaler)
    assert isinstance(get_scaler("robust_scaler"), RobustScaler)
