import pytest

from mol_gen.config.training.model import LSTMLayerConfig, ModelConfig
from mol_gen.exceptions import ConfigException


class TestModelConfig:
    @pytest.fixture
    def valid_config_section(self):
        return {
            "embedding_dim": 64,
            "lstm_layers": [
                {"units": 256, "dropout": 0.3},
                {"units": 256, "dropout": 0.5},
            ],
            "patience": 2,
            "epochs": 50,
        }

    @pytest.fixture
    def minimal_config_section(self):
        return {"embedding_dim": 64, "lstm_layers": [{"units": 256}]}

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

    def test_parse_config_sets_expected_lstm_layer_configs_given_valid_config_section(
        self, valid_config_section
    ):
        config = ModelConfig.parse_config(valid_config_section)

        assert len(config.lstm_layers) == 2

        assert isinstance(config.lstm_layers[0], LSTMLayerConfig)
        assert isinstance(config.lstm_layers[1], LSTMLayerConfig)

        assert config.lstm_layers[0] == LSTMLayerConfig(units=256, dropout=0.3)
        assert config.lstm_layers[1] == LSTMLayerConfig(units=256, dropout=0.5)

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
            ModelConfig.parse_config(
                {"embeddingdim": 1000000, "lstm_layers": [{"units": 256}]}
            )

    def test_parse_config_raises_exception_given_lstm_layer_missing(self):
        with pytest.raises(ConfigException):
            ModelConfig.parse_config(
                {"embedding_dim": 1000000, "lstmlayers": [{"units": 256}]}
            )

    def test_parse_config_raises_exception_given_lstm_layer_not_a_list(self):
        with pytest.raises(ConfigException):
            ModelConfig.parse_config(
                {"embedding_dim": 1000000, "lstm_layers": {"units": 256}}
            )


class TestLSTMLayerConfig:
    @pytest.fixture
    def valid_config_section(self):
        return {"units": 256, "dropout": 0.3}

    @pytest.fixture
    def minimal_config_section(self):
        return {"units": 256}

    def test_parse_config_completes_given_valid_config_section(
        self, valid_config_section
    ):
        LSTMLayerConfig.parse_config(valid_config_section)

    def test_parse_config_returns_expected_config_given_valid_config_section(
        self, valid_config_section
    ):
        config = LSTMLayerConfig.parse_config(valid_config_section)

        assert isinstance(config, LSTMLayerConfig)

    def test_parse_config_sets_expected_units_given_valid_config_section(
        self, valid_config_section
    ):
        config = LSTMLayerConfig.parse_config(valid_config_section)

        assert config.units == 256

    def test_parse_config_sets_expected_dropout_given_valid_config_section(
        self, valid_config_section
    ):
        config = LSTMLayerConfig.parse_config(valid_config_section)

        assert config.dropout == 0.3

    def test_parse_config_sets_expected_dropout_given_none_given(
        self, minimal_config_section
    ):
        config = LSTMLayerConfig.parse_config(minimal_config_section)

        assert config.dropout == 0

    @pytest.mark.parametrize("value", [-0.1, 1])
    def test_parse_config_raises_exception_given_dropout_outside_range(self, value):
        with pytest.raises(ConfigException) as excinfo:
            LSTMLayerConfig.parse_config({"units": 256, "dropout": value})

        assert str(excinfo.value).endswith("is not within the interval [0, 1).")
