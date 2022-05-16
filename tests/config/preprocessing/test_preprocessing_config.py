import pytest
import yaml

from mol_gen.config.preprocessing import PreprocessingConfig
from mol_gen.config.preprocessing.convert import ConvertConfig
from mol_gen.config.preprocessing.filter import FilterConfig
from mol_gen.config.preprocessing.split import SplitConfig


class TestPreprocessingConfig:
    @pytest.fixture
    def valid_config_section(self):
        return {
            "convert": ["neutralise_salts", "remove_stereochemistry"],
            "filter": {
                "allowed_elements": ["H", "C", "N", "O", "F", "S", "Cl", "Br"],
                "range_filters": {
                    "molecular_weight": {"min": 180, "max": 480},
                },
            },
            "split": {"validate": 0.1, "test": 0.2},
        }

    def test_parse_config_completes_given_valid_config_section(
        self, valid_config_section
    ):
        PreprocessingConfig.parse_config(valid_config_section)

    def test_parse_config_returns_expected_config_given_valid_config_section(
        self, valid_config_section
    ):
        config = PreprocessingConfig.parse_config(valid_config_section)

        assert isinstance(config, PreprocessingConfig)

    def test_parse_config_sets_expected_convert_config_given_valid_config_section(
        self, valid_config_section
    ):
        config = PreprocessingConfig.parse_config(valid_config_section)

        assert isinstance(config.convert, ConvertConfig)

    def test_parse_config_calls_convert_config_as_expected(
        self, mocker, valid_config_section
    ):
        spy_config = mocker.spy(ConvertConfig, "parse_config")
        PreprocessingConfig.parse_config(valid_config_section)

        spy_config.assert_called_once_with(
            ["neutralise_salts", "remove_stereochemistry"]
        )

    def test_parse_config_sets_expected_filter_config_given_valid_config_section(
        self, valid_config_section
    ):
        config = PreprocessingConfig.parse_config(valid_config_section)

        assert isinstance(config.filter, FilterConfig)

    def test_parse_config_calls_filter_config_as_expected(
        self, mocker, valid_config_section
    ):
        spy_config = mocker.spy(FilterConfig, "parse_config")
        PreprocessingConfig.parse_config(valid_config_section)

        spy_config.assert_called_once_with(
            {
                "allowed_elements": ["H", "C", "N", "O", "F", "S", "Cl", "Br"],
                "range_filters": {
                    "molecular_weight": {"min": 180, "max": 480},
                },
            }
        )

    def test_parse_config_sets_expected_split_config_given_valid_config_section(
        self, valid_config_section
    ):
        config = PreprocessingConfig.parse_config(valid_config_section)

        assert isinstance(config.split, SplitConfig)

    def test_parse_config_calls_split_config_as_expected(
        self, mocker, valid_config_section
    ):
        spy_config = mocker.spy(SplitConfig, "parse_config")
        PreprocessingConfig.parse_config(valid_config_section)

        spy_config.assert_called_once_with(
            {"validate": 0.1, "test": 0.1},
        )

    @pytest.fixture
    def valid_config_file(self, tmpdir, valid_config_section):
        fp = tmpdir.join("preprocessing.yml")

        with open(fp, "w") as fh:
            yaml.dump(valid_config_section, fh)

        return fp

    def test_from_file_completes_given_valid_file(
        self,
        valid_config_file,
    ):
        PreprocessingConfig.from_file(valid_config_file)

    def test_from_file_returns_expected_config_given_valid_file(
        self,
        valid_config_file,
    ):
        config = PreprocessingConfig.from_file(valid_config_file)

        assert isinstance(config, PreprocessingConfig)
