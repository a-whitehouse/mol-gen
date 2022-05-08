import pytest
from rdkit.Chem import MolFromSmiles, MolToSmiles

from mol_gen.config.preprocessing.convert import CONVERT_METHODS, ConvertConfig
from mol_gen.exceptions import ConfigException
from mol_gen.preprocessing.convert import (
    neutralise_salts,
    remove_fragments,
    remove_isotopes,
    remove_stereochemistry,
)


@pytest.fixture
def mol():
    return MolFromSmiles("CCC")


class TestConvertConfig:
    @pytest.fixture
    def valid_config_section(self):
        return [
            "neutralise_salts",
            "remove_fragments",
            "remove_isotopes",
            "remove_stereochemistry",
        ]

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

        assert config.methods == [
            neutralise_salts,
            remove_fragments,
            remove_isotopes,
            remove_stereochemistry,
        ]

    def test_parse_config_returns_expected_config_given_no_methods_requested(self):
        config = ConvertConfig.parse_config([])

        assert config.methods == []

    def test_parse_config_raises_exception_given_invalid_method_requested(self):
        with pytest.raises(ConfigException):
            ConvertConfig.parse_config(
                ["neutralise_salts", "unsupported"],
            )

    def test_apply_completes_given_valid_config_section(
        self, valid_config_section, mol
    ):
        config = ConvertConfig.parse_config(valid_config_section)

        config.apply(mol)

    def test_apply_calls_methods_and_returns_molecule_as_expected(
        self, mocker, valid_config_section, mol
    ):
        mocker.patch.dict(
            CONVERT_METHODS,
            {
                "neutralise_salts": lambda x: MolToSmiles(x) + "_neutral",
                "remove_fragments": lambda x: x + "_single_fragment",
                "remove_isotopes": lambda x: x + "_isotope_free",
                "remove_stereochemistry": lambda x: x + "_achiral",
            },
        )
        config = ConvertConfig.parse_config(valid_config_section)

        converted_mol = config.apply(mol)

        assert converted_mol == "CCC_neutral_single_fragment_isotope_free_achiral"
