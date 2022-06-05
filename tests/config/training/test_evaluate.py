import pytest

from mol_gen.config.training.evaluate import EvaluateConfig
from mol_gen.exceptions import ConfigException


class TestEvaluateConfig:
    @pytest.fixture
    def valid_config_section(self):
        return {"n_molecules": 100, "subset_size": 25}

    def test_parse_config_completes_given_valid_config_section(
        self, valid_config_section
    ):
        EvaluateConfig.parse_config(valid_config_section)

    def test_parse_config_returns_expected_config_given_valid_config_section(
        self, valid_config_section
    ):
        config = EvaluateConfig.parse_config(valid_config_section)

        assert isinstance(config, EvaluateConfig)

    def test_parse_config_sets_expected_n_molecules_given_valid_config_section(
        self, valid_config_section
    ):
        config = EvaluateConfig.parse_config(valid_config_section)

        assert config.n_molecules == 100

    def test_parse_config_sets_expected_subset_size_given_valid_config_section(
        self, valid_config_section
    ):
        config = EvaluateConfig.parse_config(valid_config_section)

        assert config.subset_size == 25

    def test_parse_config_raises_exception_given_n_molecules_missing(self):
        with pytest.raises(ConfigException):
            EvaluateConfig.parse_config(
                {"nmolecules": 100, "batch_size": 25},
            )

    def test_parse_config_raises_exception_given_subset_size_missing(self):
        with pytest.raises(ConfigException):
            EvaluateConfig.parse_config(
                {"n_molecules": 100, "subsetsize": 25},
            )
