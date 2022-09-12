import pytest
import yaml

from mol_gen.config.training import TrainingConfig
from mol_gen.config.training.dataset import DatasetConfig
from mol_gen.config.training.evaluate import EvaluateConfig
from mol_gen.config.training.model import ModelConfig


class TestTrainingConfig:
    @pytest.fixture
    def valid_config_section(self):
        return {
            "dataset": {"buffer_size": 1000000, "batch_size": 1024},
            "model": {
                "embedding_dim": 64,
                "lstm_layers": [
                    {"units": 64, "dropout": 0.3},
                    {"units": 64, "dropout": 0.5},
                ],
                "patience": 2,
                "epochs": 50,
            },
            "evaluate": {"n_molecules": 100, "subset_size": 25},
        }

    def test_parse_config_completes_given_valid_config_section(
        self, valid_config_section
    ):
        TrainingConfig.parse_config(valid_config_section)

    def test_parse_config_returns_expected_config_given_valid_config_section(
        self, valid_config_section
    ):
        config = TrainingConfig.parse_config(valid_config_section)

        assert isinstance(config, TrainingConfig)

    def test_parse_config_sets_expected_dataset_config_given_valid_config_section(
        self, valid_config_section
    ):
        config = TrainingConfig.parse_config(valid_config_section)

        assert isinstance(config.dataset, DatasetConfig)

    def test_parse_config_calls_dataset_config_as_expected(
        self, mocker, valid_config_section
    ):
        spy_config = mocker.spy(DatasetConfig, "parse_config")
        TrainingConfig.parse_config(valid_config_section)

        spy_config.assert_called_once_with({"buffer_size": 1000000, "batch_size": 1024})

    def test_parse_config_sets_expected_model_config_given_valid_config_section(
        self, valid_config_section
    ):
        config = TrainingConfig.parse_config(valid_config_section)

        assert isinstance(config.model, ModelConfig)

    def test_parse_config_calls_model_config_as_expected(
        self, mocker, valid_config_section
    ):
        spy_config = mocker.spy(ModelConfig, "parse_config")
        TrainingConfig.parse_config(valid_config_section)

        spy_config.assert_called_once_with(
            {
                "embedding_dim": 64,
                "lstm_layers": [
                    {"units": 64, "dropout": 0.3},
                    {"units": 64, "dropout": 0.5},
                ],
                "patience": 2,
                "epochs": 50,
            }
        )

    def test_parse_config_sets_expected_evaluate_config_given_valid_config_section(
        self, valid_config_section
    ):
        config = TrainingConfig.parse_config(valid_config_section)

        assert isinstance(config.evaluate, EvaluateConfig)

    def test_parse_config_calls_evaluate_config_as_expected(
        self, mocker, valid_config_section
    ):
        spy_config = mocker.spy(EvaluateConfig, "parse_config")
        TrainingConfig.parse_config(valid_config_section)

        spy_config.assert_called_once_with(
            {"n_molecules": 100, "subset_size": 25},
        )

    @pytest.fixture
    def valid_config_file(self, tmpdir, valid_config_section):
        fp = tmpdir.join("training.yml")

        with open(fp, "w") as fh:
            yaml.dump(valid_config_section, fh)

        return fp

    def test_from_file_completes_given_valid_file(
        self,
        valid_config_file,
    ):
        TrainingConfig.from_file(valid_config_file)

    def test_from_file_returns_expected_config_given_valid_file(
        self,
        valid_config_file,
    ):
        config = TrainingConfig.from_file(valid_config_file)

        assert isinstance(config, TrainingConfig)
