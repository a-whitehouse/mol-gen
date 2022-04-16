from mol_gen.config.preprocessing.convert import ConvertConfig
from mol_gen.preprocessing.convert import neutralise_salts, remove_stereochemistry


class TestConvertConfig:
    def test_parse_config_returns_expected_config_given_multiple_methods_requested(
        self,
    ):
        expected = [neutralise_salts, remove_stereochemistry]

        config = ConvertConfig.parse_config(
            ["neutralise_salts", "remove_stereochemistry"]
        )

        assert config.methods == expected

    def test_parse_config_returns_expected_config_given_single_method_requested(self):
        expected = [remove_stereochemistry]

        config = ConvertConfig.parse_config(["remove_stereochemistry"])

        assert config.methods == expected

    def test_parse_config_returns_expected_config_given_no_methods_requested(self):
        config = ConvertConfig.parse_config([])

        assert config.methods == []
