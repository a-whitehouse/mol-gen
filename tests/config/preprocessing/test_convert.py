import pytest

from mol_gen.config.preprocessing.convert import ConvertConfig
from mol_gen.exceptions import ConfigException
from mol_gen.preprocessing.convert import neutralise_salts, remove_stereochemistry


class TestConvertConfig:
    @pytest.mark.parametrize(
        "config, expected",
        [
            (
                ["neutralise_salts", "remove_stereochemistry"],
                [neutralise_salts, remove_stereochemistry],
            ),
            (["remove_stereochemistry"], [remove_stereochemistry]),
        ],
    )
    def test_parse_config_returns_expected_config_given_valid_methods_requested(
        self, config, expected
    ):
        config = ConvertConfig.parse_config(config)

        assert config.methods == expected

    def test_parse_config_returns_expected_config_given_no_methods_requested(self):
        config = ConvertConfig.parse_config([])

        assert config.methods == []

    def test_parse_config_raises_exception_given_invalid_method_requested(self):
        with pytest.raises(ConfigException):
            ConvertConfig.parse_config(
                ["neutralise_salts", "unsupported"],
            )
