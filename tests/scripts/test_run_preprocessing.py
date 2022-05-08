from subprocess import run

import pandas as pd
import pytest
import yaml
from pandas.testing import assert_index_equal
from pyprojroot import here


@pytest.fixture
def script_path():
    return here().joinpath("scripts", "run_preprocessing.py")


@pytest.fixture
def input_path():
    return here().joinpath("tests", "data", "chembl.parquet")


@pytest.fixture
def valid_config_section():
    return {
        "convert": ["neutralise_salts", "remove_stereochemistry"],
        "filter": {
            "allowed_elements": ["H", "C", "N", "O", "F", "S", "Cl", "Br"],
            "range_filters": {
                "molecular_weight": {"min": 180, "max": 480},
            },
        },
    }


@pytest.fixture
def config_path(tmpdir, valid_config_section):
    config_path = tmpdir.join("preprocessing.yml")

    with open(config_path, "w") as f:
        yaml.dump(valid_config_section, f)

    return config_path


class TestRunPreprocessing:
    def test_completes_given_valid_input(
        self, tmpdir, script_path, input_path, config_path
    ):
        run(
            [
                "python",
                script_path,
                "--input",
                input_path,
                "--output",
                tmpdir,
                "--config",
                config_path,
                "--column",
                "SMILES",
            ],
            check=True,
        )

    def test_writes_smiles_parquet(self, tmpdir, script_path, input_path, config_path):
        run(
            [
                "python",
                script_path,
                "--input",
                input_path,
                "--output",
                tmpdir,
                "--config",
                config_path,
                "--column",
                "SMILES",
            ],
            check=True,
        )
        actual = pd.read_parquet(tmpdir.join("smiles"))

        assert isinstance(actual, pd.DataFrame)
        assert_index_equal(actual.columns, pd.Index(["SMILES"]))

    def test_writes_selfies_parquet(self, tmpdir, script_path, input_path, config_path):
        run(
            [
                "python",
                script_path,
                "--input",
                input_path,
                "--output",
                tmpdir,
                "--config",
                config_path,
                "--column",
                "SMILES",
            ],
            check=True,
        )
        actual = pd.read_parquet(tmpdir.join("selfies", "parquet"))

        assert isinstance(actual, pd.DataFrame)
        assert_index_equal(actual.columns, pd.Index(["SELFIES"]))
