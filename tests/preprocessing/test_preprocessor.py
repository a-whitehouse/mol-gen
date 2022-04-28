import pandas as pd
import pytest
from pandas.testing import assert_series_equal
from rdkit.Chem import MolFromSmiles, MolToSmiles

from mol_gen.config.preprocessing import PreprocessingConfig, filter
from mol_gen.config.preprocessing.convert import CONVERT_METHODS
from mol_gen.exceptions import UndesirableMolecule
from mol_gen.preprocessing.preprocessor import MoleculePreprocessor


class TestMoleculePreprocessor:
    @pytest.fixture
    def smiles(self):
        return pd.Series(
            [
                "CCOC(=O)C(C)(C)C1=CC(=C(C=C1)I)C",
                "CC.CC1([C@@H]2[C@@H]3CC[C@H](C3)[C@@H]2C4=C([N+]1(C)C)C=CC5=C4C(=O)NC5)C6=CC(=C(C=C6)N)C(=N)N",
                "CC1=CN(C(=O)NC1=O)[C@H]2[C@H]3[C@@H]([C@@](O2)(CN3C4=NOC(=N4)C)COC(C)(C)C)OC(C)(C)C",
                "CC(C)CN=C1NC(C2=C(N1)N(C=N2)[C@H]3[C@H](C([C@H](O3)CO)OC(=O)C)OC(=O)C)O",
                "CC1=NC=C(C=C1)C(C)(C)N2CCC(C2)(CCC3=CC=C(S3)F)C4=NC5=C(N4)C=C(C=C5)F",
                "CCCC#CC1=C(C2=CC=CC=C2[N+](=C1)[O-])CCNC(=O)OC(C)(C)C",
                "CCOC(=O)C1=CC(=O)NC2=C1C=CC(=C2)F",
                "CCCN(CCC)C=O.CC1=CC2=C(C=C(C=C2)C(=O)NC3=CN=CC(=C3)CNC(=O)CC(C4=CC=CC=C4)N)N=C(C1)N",
                "CC#CC(=O)NC1=[C-]C2=C(C=C1)N=CN=C2NC3=CC=C(C=C3)OC4CCCCCCC4.[Y]",
                "CCN1CC(OC1=O)C2(CCN(C2)C(C)(C)C3=CN=C(C=C3)C)CCC4=CC=C(S4)F",
            ],
            name="SMILES",
        )

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
        }

    @pytest.fixture
    def config(self, valid_config_section):
        return PreprocessingConfig.parse_config(valid_config_section)

    @pytest.fixture
    def preprocessor(self, config):
        return MoleculePreprocessor(config)

    def test_process_molecules_completes_given_valid_smiles(self, preprocessor, smiles):
        preprocessor.process_molecules(smiles)

    def test_process_molecules_returns_series(self, preprocessor, smiles):
        actual = preprocessor.process_molecules(smiles)

        assert isinstance(actual, pd.Series)

    def test_process_molecules_returns_series_with_no_missing_values(
        self, preprocessor, smiles
    ):
        smiles[0] = "invalid smiles"
        actual = preprocessor.process_molecules(smiles)

        assert all(actual.notna())

    def test_process_molecules_removes_invalid_smiles_strings(
        self, preprocessor, smiles
    ):
        smiles[0] = "invalid smiles"
        actual = preprocessor.process_molecules(smiles)

        assert not any(actual.str.contains("invalid smiles"))

    def test_process_molecules_replaces_each_value_using_process_molecule(
        self, mocker, preprocessor, smiles
    ):
        mocker.patch.object(
            preprocessor, "process_molecule", autospec=True, side_effect=range(10)
        )

        actual = preprocessor.process_molecules(smiles)

        assert_series_equal(
            actual, pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], name="SMILES")
        )

    def test_process_molecules_calls_process_molecule_with_smiles_strings(
        self, mocker, preprocessor, smiles
    ):
        mock_process_mol = mocker.spy(preprocessor, "process_molecule")

        preprocessor.process_molecules(smiles)

        assert mock_process_mol.call_count == 10

        mock_process_mol.assert_any_call("CCOC(=O)C(C)(C)C1=CC(=C(C=C1)I)C")
        mock_process_mol.assert_any_call("CCOC(=O)C1=CC(=O)NC2=C1C=CC(=C2)F")

    def test_process_molecule_completes_given_valid_smiles(self, preprocessor):
        preprocessor.process_molecule("CCC")

    def test_process_molecule_passes_converted_molecule_to_filter(
        self, mocker, preprocessor
    ):
        mocker.patch.object(
            preprocessor, "_apply_conversions", return_value="converted_molecule"
        )
        mock_filter = mocker.patch.object(
            preprocessor, "_apply_filters", side_effect=UndesirableMolecule
        )

        preprocessor.process_molecule("CCC")

        mock_filter.assert_called_once_with("converted_molecule")

    def test_process_molecule_returns_molecule_if_filters_passed(
        self, mocker, preprocessor
    ):
        mocker.patch.object(
            preprocessor,
            "_apply_conversions",
            return_value=MolFromSmiles("CCC"),
        )
        mocker.patch.object(preprocessor, "_apply_filters", return_value=None)

        actual = preprocessor.process_molecule("CCC")

        assert actual is not None

    def test_process_molecule_returns_converted_molecule_if_filters_passed(
        self, mocker, preprocessor
    ):
        mocker.patch.object(
            preprocessor,
            "_apply_conversions",
            return_value=MolFromSmiles("CCC"),
        )
        mocker.patch.object(preprocessor, "_apply_filters", return_value=None)

        actual = preprocessor.process_molecule("CCC")

        assert actual == "CCC"

    def test_process_molecule_returns_none_if_molecule_fails_filter(
        self, mocker, preprocessor
    ):
        mocker.patch.object(
            preprocessor,
            "_apply_conversions",
            return_value=MolFromSmiles("CCC"),
        )
        mocker.patch.object(
            preprocessor, "_apply_filters", side_effect=UndesirableMolecule
        )
        actual = preprocessor.process_molecule("CCC")

        assert actual is None

    def test__apply_conversions_completes_given_valid_molecule(self, preprocessor):
        mol = MolFromSmiles("CCC")
        preprocessor._apply_conversions(mol)

    def test__apply_conversions_makes_expected_changes_to_molecule(
        self,
        mocker,
        valid_config_section,
    ):
        mocker.patch.dict(
            CONVERT_METHODS,
            {
                "neutralise_salts": lambda x: MolToSmiles(x) + "_neutral",
                "remove_stereochemistry": lambda x: x + "_achiral",
            },
        )
        config = PreprocessingConfig.parse_config(valid_config_section)
        preprocessor = MoleculePreprocessor(config)
        mol = MolFromSmiles("CCC")

        converted_mol = preprocessor._apply_conversions(mol)

        assert converted_mol == "CCC_neutral_achiral"

    def test__apply_filters_makes_expected_calls_to_filter_functions(
        self, mocker, preprocessor
    ):
        spy_allowed_elements = mocker.spy(
            filter,
            "check_only_allowed_elements_present",
        )
        spy_molecular_weight = mocker.spy(
            filter,
            "check_descriptor_within_range",
        )

        mol = MolFromSmiles("CCC")

        with pytest.raises(UndesirableMolecule):
            preprocessor._apply_filters(mol)

        spy_allowed_elements.assert_called_once_with(
            mol, allowed_elements=["H", "C", "N", "O", "F", "S", "Cl", "Br"]
        )
        spy_molecular_weight.assert_called_once_with(mol, "molecular_weight", 180, 480)
