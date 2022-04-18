import pytest

from mol_gen.config.preprocessing.convert import ConvertConfig
from mol_gen.exceptions import ConfigException
from mol_gen.preprocessing.convert import neutralise_salts, remove_stereochemistry


class TestConvertConfig:
    @pytest.fixture
    def valid_config_section(self):
        return ["neutralise_salts", "remove_stereochemistry"]

    def test_parse_config_completes_given_valid_config_section(
        self, valid_config_section
    ):
        ConvertConfig.parse_config(valid_config_section)

    def test_parse_config_returns_expected_config_given_valid_config_section(
        self, valid_config_section
    ):
        config = ConvertConfig.parse_config(valid_config_section)

        assert isinstance(config, ConvertConfig)

    def test_parse_config_sets_expected_methods_given_valid_config_section(
        self, valid_config_section
    ):
        config = ConvertConfig.parse_config(valid_config_section)

        assert config.methods == [neutralise_salts, remove_stereochemistry]

    def test_parse_config_returns_expected_config_given_no_methods_requested(self):
        config = ConvertConfig.parse_config([])

        assert config.methods == []

    def test_parse_config_raises_exception_given_invalid_method_requested(self):
        with pytest.raises(ConfigException):
            ConvertConfig.parse_config(
                ["neutralise_salts", "unsupported"],
            )
