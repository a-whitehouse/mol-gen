import json
from pathlib import Path
from subprocess import run

import pytest
import tensorflow as tf
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
            "dropout": 0.5,
            "patience": 5,
            "epochs": 2,
        },
        "evaluate": {"n_molecules": 10, "subset_size": 2},
    }


@pytest.fixture
def config_path(tmpdir, valid_config_section):
    config_path = tmpdir.join("training.yml")

    with open(config_path, "w") as f:
        yaml.dump(valid_config_section, f)

    return config_path


class TestRunTraining:
    def test_completes_and_writes_expected_files_given_valid_input(
        self, tmpdir, input_path, config_path
    ):
        tmpdir = Path(tmpdir)
        run(
            [
                "mol-gen",
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

        # Check model checkpoints
        checkpoint_files = list(tmpdir.joinpath("checkpoints").glob("model.*.h5"))
        assert len(checkpoint_files) == 2
        tf.keras.models.load_model(checkpoint_files[0])
        tf.keras.models.load_model(checkpoint_files[1])

        # Check model evaluation reports
        report_files = list(tmpdir.joinpath("reports").glob("model.*.html"))
        assert len(report_files) == 2

        # Check string lookup JSON
        string_lookup_file = tmpdir / "string_lookup.json"
        assert string_lookup_file.exists()
        with open(string_lookup_file) as fh:
            json.load(fh)
