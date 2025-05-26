import pandas as pd
import pytest

from mlops.utils.common_utils import write_yaml_file
from mlops.utils.data_validation_utils import (generate_validation_report,
                                               get_validation_results,
                                               save_validation_report)


def test_get_validation_results_success():
    df = pd.DataFrame({"age": [25, 30], "name": ["Alice", "Bob"]})
    column_schema = {"age": "int64", "name": "object"}

    results, status = get_validation_results(df, column_schema)

    assert status is True
    assert isinstance(results, list)
    assert any(r["column"] == "age" and r["validated"] for r in results)
    assert any(r["column"] == "name" and r["validated"] for r in results)


def test_get_validation_results_fail_dtype():
    df = pd.DataFrame({"age": [25, 30], "name": ["Alice", "Bob"]})
    column_schema = {
        "age": "float64",
        "name": "object",
    }

    results, status = get_validation_results(df, column_schema)

    assert status is False
    assert any(not r["validated"] for r in results if r["column"] == "age")


def test_save_validation_report_creates_yaml(tmp_path):
    validation_results = [
        {
            "column": "age",
            "expected_dtype": "int64",
            "got_dtype": "int64",
            "validated": True,
        }
    ]
    validation_status = True
    file_path = tmp_path / "validation_report.yaml"

    save_validation_report(validation_results, validation_status, str(file_path))

    assert file_path.exists()


def test_generate_validation_report_returns_status(tmp_path):
    df = pd.DataFrame({"age": [25, 30], "name": ["Alice", "Bob"]})
    column_schema = {"age": "int64", "name": "object"}
    file_path = tmp_path / "validation_report.yaml"

    status = generate_validation_report(df, column_schema, str(file_path))

    assert status is True
    assert file_path.exists()
