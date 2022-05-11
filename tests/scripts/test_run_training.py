from subprocess import run

import pytest
from pyprojroot import here


@pytest.fixture
def script_path():
    return here().joinpath("scripts", "run_training.py")


@pytest.fixture
def input_path():
    return here().joinpath("tests", "data", "preprocessed")


class TestRunTraining:
    def test_completes_given_valid_input(self, tmpdir, script_path, input_path):
        run(
            ["python", script_path, "--input", input_path, "--output", tmpdir],
            check=True,
        )
