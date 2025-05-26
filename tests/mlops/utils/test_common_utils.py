import os

import pandas as pd
import pytest

from mlops.utils.common_utils import (create_directories, get_os_path, get_X_y,
                                      load_object, read_dataset,
                                      read_yaml_file, save_file_as_csv,
                                      save_object, write_yaml_file)


def test_write_and_read_yaml(tmp_path):
    file = tmp_path / "test.yaml"
    data = {"key": "value"}

    write_yaml_file(str(file), data)

    result = read_yaml_file(str(file))

    assert result == data


def test_read_yaml_file_invalid_path():
    with pytest.raises(Exception):
        read_yaml_file("nicht_existierender_pfad.yaml")


def test_get_os_path():
    result = get_os_path("folder1", "file.txt")
    expected = os.path.join("folder1", "file.txt")
    assert result == expected


def test_create_directories(tmp_path):
    dirs = [tmp_path / "dir1", tmp_path / "dir2"]
    dirs = [str(d) for d in dirs]

    create_directories(dirs)

    for d in dirs:
        assert os.path.exists(d)
        assert os.path.isdir(d)


def test_save_and_read_dataset(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    file = tmp_path / "test.csv"

    save_file_as_csv(df, str(file))

    df_loaded = read_dataset(str(file))

    pd.testing.assert_frame_equal(df, df_loaded)


def test_read_dataset_invalid_path():
    with pytest.raises(Exception):
        read_dataset("nicht_existierende_datei.csv")


def test_save_and_load_object(tmp_path):
    obj = {"a": 1, "b": 2}
    file = tmp_path / "obj.pkl"

    save_object(obj, str(file))
    loaded_obj = load_object(str(file))

    assert obj == loaded_obj


def test_load_object_invalid_path():
    with pytest.raises(Exception):
        load_object("nicht_existierende_datei.pkl")


def test_get_X_y():
    df = pd.DataFrame({"feature1": [1, 2], "feature2": [3, 4], "target": [0, 1]})

    X, y = get_X_y(df, "target")

    assert "target" not in X.columns
    assert list(y) == [0, 1]
