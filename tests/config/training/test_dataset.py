import pytest

from mol_gen.config.training.dataset import DatasetConfig
from mol_gen.exceptions import ConfigException


class TestDatasetConfig:
    @pytest.fixture
    def valid_config_section(self):
        return {"buffer_size": 1000000, "batch_size": 1024}

    def test_parse_config_completes_given_valid_config_section(
        self, valid_config_section
    ):
        DatasetConfig.parse_config(valid_config_section)

    def test_parse_config_returns_expected_config_given_valid_config_section(
        self, valid_config_section
    ):
        config = DatasetConfig.parse_config(valid_config_section)

        assert isinstance(config, DatasetConfig)

    def test_parse_config_sets_expected_buffer_size_given_valid_config_section(
        self, valid_config_section
    ):
        config = DatasetConfig.parse_config(valid_config_section)

        assert config.buffer_size == 1000000

    def test_parse_config_sets_expected_batch_size_given_valid_config_section(
        self, valid_config_section
    ):
        config = DatasetConfig.parse_config(valid_config_section)

        assert config.batch_size == 1024

    def test_parse_config_raises_exception_given_buffer_size_missing(self):
        with pytest.raises(ConfigException):
            DatasetConfig.parse_config(
                {"buffersize": 1000000, "batch_size": 1024},
            )

    def test_parse_config_raises_exception_given_batch_size_missing(self):
        with pytest.raises(ConfigException):
            DatasetConfig.parse_config(
                {"buffer_size": 1000000, "batchsize": 1024},
            )
