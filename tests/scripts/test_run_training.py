from subprocess import run

import pytest
from pyprojroot import here


@pytest.fixture
def input_path():
    return here().joinpath("tests", "data", "preprocessed")


class TestRunTraining:
    def test_completes_given_valid_input(self, tmpdir, input_path):
        run(
            ["train", "--input", input_path, "--output", tmpdir],
            check=True,
        )
