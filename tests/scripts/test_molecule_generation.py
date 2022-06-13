from pathlib import Path
from subprocess import run

import pandas as pd
import pytest
from pyprojroot import here


@pytest.fixture
def input_path():
    return here() / "tests" / "data" / "trained" / "checkpoints" / "model.02.h5"


@pytest.fixture
def vocab_path():
    return here() / "tests" / "data" / "trained" / "string_lookup.json"


@pytest.fixture
def output_path(tmpdir):
    return Path(tmpdir) / "generated.txt"


class TestMoleculeGeneration:
    def test_completes_and_writes_expected_file_given_valid_input(
        self, input_path, output_path, vocab_path
    ):
        run(
            [
                "mol-gen",
                "generate",
                "--model",
                input_path,
                "--output",
                output_path,
                "--vocab",
                vocab_path,
                "--n-mols",
                "50",
            ],
            check=True,
        )

        actual = pd.read_csv(output_path, header=None)
        assert isinstance(actual, pd.DataFrame)
        assert len(actual)
