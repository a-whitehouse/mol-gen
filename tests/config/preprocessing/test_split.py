import pytest

from mol_gen.config.preprocessing.split import SplitConfig
from mol_gen.exceptions import ConfigException


class TestSplitConfig:
    @pytest.fixture
    def config_section(self):
        return {"validate": 0.1, "test": 0.2}

    def test_parse_config_completes_given_valid_config_section(self, config_section):
        SplitConfig.parse_config(config_section)

    def test_parse_config_returns_expected_config_given_valid_config_section(
        self, config_section
    ):
        config = SplitConfig.parse_config(config_section)

        assert isinstance(config, SplitConfig)

    def test_parse_config_sets_expected_proportions_given_valid_config_section(
        self, config_section
    ):
        config = SplitConfig.parse_config(config_section)

        assert config.validate == 0.1
        assert config.test == 0.2

    @pytest.mark.parametrize("split", ["validate", "test"])
    def test_parse_config_raises_exception_given_split_missing(
        self, config_section, split
    ):
        config_section.pop(split)

        with pytest.raises(ConfigException):
            SplitConfig.parse_config(config_section)

    @pytest.mark.parametrize("split", ["validate", "test"])
    @pytest.mark.parametrize("size", [-0.1, 0.0, "0.2", 1.0, 1.1])
    def test_parse_config_raises_exception_given_invalid_size(
        self, config_section, split, size
    ):
        config_section[split] = size

        with pytest.raises(ConfigException) as excinfo:
            SplitConfig.parse_config(config_section)

        assert "should be a number between 0 and 1." in str(excinfo.value)

    def test_parse_config_raises_exception_given_insufficient_train_size(self):
        with pytest.raises(ConfigException) as excinfo:
            SplitConfig.parse_config({"validate": 0.9, "test": 0.2})

        assert (
            str(excinfo.value)
            == "Total size for calibrate and test sets should be less than 1."
        )

    def test_apply_completes_given_valid_config_section(self, config_section):
        config = SplitConfig.parse_config(config_section)

        config.apply()

    def test_apply_calls_function_as_expected(self, mocker, config_section):
        mock_assign_to_split = mocker.patch(
            "mol_gen.config.preprocessing.split.assign_to_split",
        )
        config = SplitConfig.parse_config(config_section)

        config.apply()

        mock_assign_to_split.assert_called_once_with(0.1, 0.2)

    def test_apply_returns_assignment(self, mocker, config_section):
        mocker.patch(
            "mol_gen.config.preprocessing.split.assign_to_split",
            return_value="assigned_split",
        )
        config = SplitConfig.parse_config(config_section)

        actual = config.apply()

        assert actual == "assigned_split"
