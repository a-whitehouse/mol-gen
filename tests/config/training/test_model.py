import pytest

from mol_gen.config.training.model import ModelConfig
from mol_gen.exceptions import ConfigException


class TestModelConfig:
    @pytest.fixture
    def valid_config_section(self):
        return {
            "embedding_dim": 64,
            "lstm_units": 128,
            "dropout": 0.5,
            "patience": 2,
            "epochs": 50,
        }

    @pytest.fixture
    def minimal_config_section(self):
        return {"embedding_dim": 64, "lstm_units": 128}

    def test_parse_config_completes_given_valid_config_section(
        self, valid_config_section
    ):
        ModelConfig.parse_config(valid_config_section)

    def test_parse_config_returns_expected_config_given_valid_config_section(
        self, valid_config_section
    ):
        config = ModelConfig.parse_config(valid_config_section)

        assert isinstance(config, ModelConfig)

    def test_parse_config_sets_expected_embedding_dim_given_valid_config_section(
        self, valid_config_section
    ):
        config = ModelConfig.parse_config(valid_config_section)

        assert config.embedding_dim == 64

    def test_parse_config_sets_expected_lstm_units_given_valid_config_section(
        self, valid_config_section
    ):
        config = ModelConfig.parse_config(valid_config_section)

        assert config.lstm_units == 128

    def test_parse_config_sets_expected_dropout_given_valid_config_section(
        self, valid_config_section
    ):
        config = ModelConfig.parse_config(valid_config_section)

        assert config.dropout == 0.5

    def test_parse_config_sets_expected_dropout_given_none_given(
        self, minimal_config_section
    ):
        config = ModelConfig.parse_config(minimal_config_section)

        assert config.dropout == 0

    def test_parse_config_sets_expected_patience_given_valid_config_section(
        self, valid_config_section
    ):
        config = ModelConfig.parse_config(valid_config_section)

        assert config.patience == 2

    def test_parse_config_sets_expected_patience_given_none_given(
        self, minimal_config_section
    ):
        config = ModelConfig.parse_config(minimal_config_section)

        assert config.patience == 5

    def test_parse_config_sets_expected_epochs_given_valid_config_section(
        self, valid_config_section
    ):
        config = ModelConfig.parse_config(valid_config_section)

        assert config.epochs == 50

    def test_parse_config_sets_expected_epochs_given_none_given(
        self, minimal_config_section
    ):
        config = ModelConfig.parse_config(minimal_config_section)

        assert config.epochs == 100

    def test_parse_config_raises_exception_given_embedding_dim_missing(self):
        with pytest.raises(ConfigException):
            ModelConfig.parse_config({"embeddingdim": 1000000, "lstm_units": 128})

    def test_parse_config_raises_exception_given_lstm_units_missing(self):
        with pytest.raises(ConfigException):
            ModelConfig.parse_config({"embedding_dim": 1000000, "lstmunits": 128})
