import pandas as pd

from mlops.utils.data_ingestion_utils import perform_train_test_split


def test_perform_train_test_split():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, 4, 5]})
    train_test_split_ratio = 0.2
    seed = 42

    train_df, test_df = perform_train_test_split(df, train_test_split_ratio, seed)

    assert len(train_df) == 4
    assert len(test_df) == 1
