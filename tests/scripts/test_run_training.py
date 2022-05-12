from subprocess import run

import pytest
import yaml
from pyprojroot import here


@pytest.fixture
def input_path():
    return here().joinpath("tests", "data", "preprocessed")


@pytest.fixture
def valid_config_section():
    return {
        "dataset": {
            "buffer_size": 1000000,
            "batch_size": 1024,
        },
        "model": {
            "embedding_dim": 64,
            "lstm_units": 128,
        },
    }


@pytest.fixture
def config_path(tmpdir, valid_config_section):
    config_path = tmpdir.join("training.yml")

    with open(config_path, "w") as f:
        yaml.dump(valid_config_section, f)

    return config_path


class TestRunTraining:
    def test_completes_given_valid_input(self, tmpdir, input_path, config_path):
        run(
            [
                "train",
                "--input",
                input_path,
                "--output",
                tmpdir,
                "--config",
                config_path,
            ],
            check=True,
        )
