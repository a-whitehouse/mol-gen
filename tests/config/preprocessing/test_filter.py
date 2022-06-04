from unittest.mock import call

import pytest
from rdkit.Chem import AllChem, MolFromSmiles
from rdkit.DataStructs.cDataStructs import UIntSparseIntVect

from mol_gen.config.preprocessing import filter
from mol_gen.config.preprocessing.filter import (
    ElementsFilter,
    FilterConfig,
    RangeFilter,
    StructureFilter,
)
from mol_gen.exceptions import ConfigException


@pytest.fixture
def mol():
    return MolFromSmiles(
        "NC1=NNC(C2=CC(N(CC3=CC=C(CN4CCN(C)CC4)C=C3)C=C5)=C5C=C2)=C1C#N"
    )


class TestFilterConfig:
    @pytest.fixture
    def valid_config_section(self):
        return {
            "allowed_elements": ["H", "C", "N", "O", "F", "S", "Cl", "Br"],
            "range_filters": {
                "molecular_weight": {"min": 180, "max": 480},
                "partition_coefficient": {"min": -0.4},
                "rotatable_bonds": {"max": 10},
            },
            "structure_filters": [
                {"smiles": "NC1=NNC(C2=CC(NC=C5)=C5C=C2)=C1C#N", "min": 0.35},
                {"smiles": "CN1CCN(Cc2ccccc2)CC1", "min": 0.3},
            ],
        }

    def test_parse_config_completes_given_valid_config_section(
        self, valid_config_section
    ):
        FilterConfig.parse_config(valid_config_section)

    def test_parse_config_returns_expected_config_given_valid_config_section(
        self, valid_config_section
    ):
        config = FilterConfig.parse_config(valid_config_section)

        assert isinstance(config, FilterConfig)

    def test_parse_config_sets_expected_allowed_elements_given_valid_config_section(
        self, valid_config_section
    ):
        config = FilterConfig.parse_config(valid_config_section)
        allowed_elements = config.elements_filter.allowed_elements

        assert isinstance(config.elements_filter, ElementsFilter)
        assert allowed_elements == ["H", "C", "N", "O", "F", "S", "Cl", "Br"]

    @pytest.mark.parametrize(
        "expected_descriptor, expected_min, expected_max",
        [
            ("molecular_weight", 180, 480),
            (
                "partition_coefficient",
                -0.4,
                None,
            ),
            ("rotatable_bonds", None, 10),
        ],
    )
    def test_parse_config_sets_expected_range_filters_given_valid_config_section(
        self, valid_config_section, expected_descriptor, expected_min, expected_max
    ):
        config = FilterConfig.parse_config(valid_config_section)
        range_filter = config.range_filters[expected_descriptor]

        assert isinstance(range_filter, RangeFilter)
        assert range_filter.descriptor == expected_descriptor
        assert range_filter.min == expected_min
        assert range_filter.max == expected_max

    def test_parse_config_sets_range_filters_as_empty_given_none_requested(
        self,
    ):
        config = FilterConfig.parse_config(
            {
                "allowed_elements": ["H", "C", "N", "O", "F", "S", "Cl", "Br"],
            }
        )

        assert config.range_filters == {}

    def test_parse_config_sets_expected_number_of_structure_filters_given_valid_config_section(
        self, valid_config_section
    ):
        config = FilterConfig.parse_config(valid_config_section)
        structure_filters = config.structure_filters

        assert len(structure_filters) == 2

    @pytest.mark.parametrize(
        "element, expected_smiles, expected_min",
        [
            (0, "NC1=NNC(C2=CC(NC=C5)=C5C=C2)=C1C#N", 0.35),
            (1, "CN1CCN(Cc2ccccc2)CC1", 0.3),
        ],
    )
    def test_parse_config_sets_expected_structure_filter_given_valid_config_section(
        self, valid_config_section, element, expected_smiles, expected_min
    ):
        config = FilterConfig.parse_config(valid_config_section)
        structure_filter = config.structure_filters[element]

        assert isinstance(structure_filter, StructureFilter)
        assert structure_filter.smiles == expected_smiles
        assert structure_filter.min == expected_min

    def test_parse_config_sets_structure_filters_as_empty_given_none_requested(
        self,
    ):
        config = FilterConfig.parse_config(
            {
                "allowed_elements": ["H", "C", "N", "O", "F", "S", "Cl", "Br"],
            }
        )

        assert config.structure_filters == []

    def test_apply_completes_given_valid_config_section(
        self, valid_config_section, mol
    ):
        config = FilterConfig.parse_config(valid_config_section)

        config.apply(mol)

    def test_apply_calls_elements_filter_function_as_expected(
        self, mocker, valid_config_section, mol
    ):
        mock_filter_func = mocker.patch.object(
            filter, "check_only_allowed_elements_present"
        )
        config = FilterConfig.parse_config(valid_config_section)

        config.apply(mol)

        mock_filter_func.assert_called_once_with(
            mol, allowed_elements=["H", "C", "N", "O", "F", "S", "Cl", "Br"]
        )

    def test_apply_calls_range_filter_functions_as_expected(
        self, mocker, valid_config_section, mol
    ):
        mock_filter_func = mocker.patch.object(filter, "check_descriptor_within_range")
        config = FilterConfig.parse_config(valid_config_section)

        config.apply(mol)

        mock_filter_func.assert_has_calls(
            [
                call(mol, "molecular_weight", min=180, max=480),
                call(mol, "partition_coefficient", min=-0.4, max=None),
                call(mol, "rotatable_bonds", min=None, max=10),
            ]
        )

    def test_apply_calls_structure_filter_functions_as_expected(
        self, mocker, valid_config_section, mol
    ):
        mock_filter_func = mocker.patch.object(
            filter, "check_tanimoto_score_above_threshold"
        )
        config = FilterConfig.parse_config(valid_config_section)

        config.apply(mol)

        mock_filter_func.assert_has_calls(
            [
                call(
                    mol,
                    AllChem.GetMorganFingerprint(
                        MolFromSmiles("NC1=NNC(C2=CC(NC=C5)=C5C=C2)=C1C#N"), 2
                    ),
                    min=0.35,
                ),
                call(
                    mol,
                    AllChem.GetMorganFingerprint(
                        MolFromSmiles("CN1CCN(Cc2ccccc2)CC1"), 2
                    ),
                    min=0.3,
                ),
            ]
        )


class TestElementsFilter:
    @pytest.fixture
    def valid_config_section(self):
        return ["H", "C", "N", "O", "F", "S", "Cl", "Br"]

    def test_parse_config_completes_given_valid_config_section(
        self, valid_config_section
    ):
        ElementsFilter.parse_config(valid_config_section)

    def test_parse_config_returns_expected_config_given_valid_config_section(
        self, valid_config_section
    ):
        config = ElementsFilter.parse_config(valid_config_section)

        assert isinstance(config, ElementsFilter)

    def test_parse_config_sets_expected_attributes_given_valid_config_section(
        self, valid_config_section
    ):
        config = ElementsFilter.parse_config(valid_config_section)

        assert config.allowed_elements == ["H", "C", "N", "O", "F", "S", "Cl", "Br"]

    @pytest.mark.parametrize(
        "allowed_elements, expected",
        [
            (["D", "C", "N"], "Element D is not recognised."),
            (["H", "c", "N"], "Element c is not recognised."),
        ],
    )
    def test_parse_config_raises_exception_given_invalid_allowed_elements(
        self, allowed_elements, expected
    ):
        with pytest.raises(ConfigException) as excinfo:
            ElementsFilter.parse_config(allowed_elements)

        assert str(excinfo.value) == expected

    def test_apply_completes_given_valid_config_section(
        self, valid_config_section, mol
    ):
        config = ElementsFilter.parse_config(valid_config_section)

        config.apply(mol)

    def test_apply_calls_filter_function_as_expected(
        self, mocker, valid_config_section, mol
    ):
        mock_filter_func = mocker.patch.object(
            filter, "check_only_allowed_elements_present"
        )
        config = ElementsFilter.parse_config(valid_config_section)

        config.apply(mol)

        mock_filter_func.assert_called_once_with(
            mol, allowed_elements=["H", "C", "N", "O", "F", "S", "Cl", "Br"]
        )


class TestRangeFilter:
    @pytest.fixture
    def valid_config_section(self):
        return {"molecular_weight": {"min": 180, "max": 480}}

    def test_parse_config_completes_given_valid_config_section(
        self, valid_config_section
    ):
        RangeFilter.parse_config(valid_config_section)

    def test_parse_config_returns_expected_config_given_valid_config_section(
        self, valid_config_section
    ):
        config = RangeFilter.parse_config(valid_config_section)

        assert isinstance(config, RangeFilter)

    @pytest.mark.parametrize(
        "valid_config_section, expected_descriptor, expected_min, expected_max",
        [
            (
                {"molecular_weight": {"min": 180, "max": 480}},
                "molecular_weight",
                180,
                480,
            ),
            ({"molecular_weight": {"min": 180}}, "molecular_weight", 180, None),
            ({"molecular_weight": {"max": 480}}, "molecular_weight", None, 480),
            ({"molecular_weight": {}}, "molecular_weight", None, None),
        ],
    )
    def test_parse_config_sets_expected_attributes_given_valid_config_section(
        self, valid_config_section, expected_descriptor, expected_min, expected_max
    ):
        config = RangeFilter.parse_config(valid_config_section)

        assert config.descriptor == expected_descriptor
        assert config.min == expected_min
        assert config.max == expected_max

    def test_parse_config_raises_exception_given_invalid_descriptor(self):
        with pytest.raises(ConfigException) as excinfo:
            RangeFilter.parse_config({"unrecognised": {"min": 180, "max": 480}})

        assert str(excinfo.value) == "Descriptor unrecognised is not recognised."

    def test_parse_config_raises_exception_given_invalid_min_value(self):
        with pytest.raises(ConfigException) as excinfo:
            RangeFilter.parse_config({"molecular_weight": {"min": "180", "max": 480}})

        assert (
            str(excinfo.value)
            == "Minimum value 180 for molecular_weight is not a number."
        )

    def test_parse_config_raises_exception_given_invalid_max_value(self):
        with pytest.raises(ConfigException) as excinfo:
            RangeFilter.parse_config({"molecular_weight": {"min": 180, "max": "480"}})

        assert (
            str(excinfo.value)
            == "Maximum value 480 for molecular_weight is not a number."
        )

    def test_apply_completes_given_valid_config_section(
        self, valid_config_section, mol
    ):
        config = RangeFilter.parse_config(valid_config_section)

        config.apply(mol)

    def test_apply_calls_filter_function_as_expected(
        self, mocker, valid_config_section, mol
    ):
        mock_filter_func = mocker.patch.object(filter, "check_descriptor_within_range")
        config = RangeFilter.parse_config(valid_config_section)

        config.apply(mol)

        mock_filter_func.assert_called_once_with(
            mol, "molecular_weight", min=180, max=480
        )


class TestStructureFilter:
    @pytest.fixture
    def valid_config_section(self):
        return {"smiles": "CN1CCN(Cc2ccccc2)CC1", "min": 0.3}

    def test_parse_config_completes_given_valid_config_section(
        self, valid_config_section
    ):
        StructureFilter.parse_config(valid_config_section)

    def test_parse_config_returns_expected_config_given_valid_config_section(
        self, valid_config_section
    ):
        config = StructureFilter.parse_config(valid_config_section)

        assert isinstance(config, StructureFilter)

    def test_parse_config_sets_expected_attributes_given_valid_config_section(
        self, valid_config_section
    ):
        config = StructureFilter.parse_config(valid_config_section)

        assert config.smiles == "CN1CCN(Cc2ccccc2)CC1"
        assert config.min == 0.3
        assert isinstance(config.fingerprint, UIntSparseIntVect)

    @pytest.mark.parametrize("value", [0, 1.1])
    def test_parse_config_raises_exception_given_min_value_outside_range(self, value):
        with pytest.raises(ConfigException) as excinfo:
            StructureFilter.parse_config(
                {"smiles": "CN1CCN(Cc2ccccc2)CC1", "min": value}
            )
        assert str(excinfo.value).endswith("is not within the interval (0, 1].")

    @pytest.mark.parametrize("value", [None, "0.3"])
    def test_parse_config_raises_exception_given_min_value_not_numeric(self, value):
        with pytest.raises(ConfigException) as excinfo:
            StructureFilter.parse_config(
                {"smiles": "CN1CCN(Cc2ccccc2)CC1", "min": value}
            )

        assert str(excinfo.value).endswith("is not a number.")

    @pytest.mark.parametrize("smiles", [None, "unrecognised"])
    def test_parse_config_raises_exception_given_invalid_smiles(self, smiles):
        with pytest.raises(ConfigException):
            StructureFilter.parse_config({"smiles": smiles, "min": 0.3})

    def test_apply_completes_given_valid_config_section(
        self, valid_config_section, mol
    ):
        config = StructureFilter.parse_config(valid_config_section)

        config.apply(mol)

    def test_apply_calls_filter_function_as_expected(
        self, mocker, valid_config_section, mol
    ):
        mock_filter_func = mocker.patch.object(
            filter, "check_tanimoto_score_above_threshold"
        )
        config = StructureFilter.parse_config(valid_config_section)

        config.apply(mol)

        mock_filter_func.assert_called_once_with(
            mol,
            AllChem.GetMorganFingerprint(MolFromSmiles("CN1CCN(Cc2ccccc2)CC1"), 2),
            min=0.3,
        )
