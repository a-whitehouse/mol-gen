import pytest
from mol_gen.exceptions import FilterException, UndesirableMolecule
from mol_gen import filter_molecules
from mol_gen.filter_molecules import (
    check_property_within_range,
    check_value_within_range,
)
from rdkit.Chem import MolFromSmiles


class TestCheckPropertyWithinRange:
    @pytest.fixture
    def mol(self):
        return MolFromSmiles(
            "NC1=NNC(C2=CC(N(CC3=CC=C(CN4CCN(C)CC4)C=C3)C=C5)=C5C=C2)=C1C#N"
        )

    def test_raises_exception_with_unrecognised_property(self, mol):
        with pytest.raises(FilterException):
            check_property_within_range("unrecognised", mol)

    @pytest.mark.parametrize(
        "property",
        [
            "hydrogen_bond_acceptors",
            "hydrogen_bond_donors",
            "molar_refractivity",
            "molecular_weight",
            "partition_coefficient",
            "rotatable_bonds",
            "topological_polar_surface_area",
        ],
    )
    def test_completes_with_recognised_property(self, property, mol):
        check_property_within_range(property, mol)

    def test_calls_check_value_within_range_as_expected_given_min_and_max_undefined(
        self, mocker, mol
    ):
        mock_check_val = mocker.patch.object(
            filter_molecules, "check_value_within_range"
        )
        filter_molecules.PROPERTY_TO_FUNCTION["test_func"] = lambda x: 5

        check_property_within_range("test_func", mol)

        mock_check_val.assert_called_once_with(5, min=None, max=None)

    def test_calls_check_value_within_range_as_expected_given_min_and_max_defined(
        self, mocker, mol
    ):
        mock_check_val = mocker.patch.object(
            filter_molecules, "check_value_within_range"
        )
        filter_molecules.PROPERTY_TO_FUNCTION["test_func"] = lambda x: 5

        check_property_within_range("test_func", mol, min=4, max=6)

        mock_check_val.assert_called_once_with(5, min=4, max=6)

    def test_calls_check_value_within_range_as_expected_given_only_min_defined(
        self, mocker, mol
    ):
        mock_check_val = mocker.patch.object(
            filter_molecules, "check_value_within_range"
        )
        filter_molecules.PROPERTY_TO_FUNCTION["test_func"] = lambda x: 5

        check_property_within_range("test_func", mol, min=4)

        mock_check_val.assert_called_once_with(5, min=4, max=None)

    def test_calls_check_value_within_range_as_expected_given_only_max_defined(
        self, mocker, mol
    ):
        mock_check_val = mocker.patch.object(
            filter_molecules, "check_value_within_range"
        )
        filter_molecules.PROPERTY_TO_FUNCTION["test_func"] = lambda x: 5

        check_property_within_range("test_func", mol, max=6)

        mock_check_val.assert_called_once_with(5, min=None, max=6)


class TestCheckValueWithinRange:
    def test_allows_value_given_min_and_max_undefined(self):
        check_value_within_range(5)

    @pytest.mark.parametrize("value", [4, 4.1])
    def test_allows_value_given_min_not_exceeded(self, value):
        check_value_within_range(value, min=4)

    @pytest.mark.parametrize("value", [5.9, 6])
    def test_allows_value_given_max_not_exceeded(self, value):
        check_value_within_range(value, max=6)

    def test_raises_exception_given_min_exceeded(self):
        with pytest.raises(UndesirableMolecule):
            check_value_within_range(3.9, min=4)

    def test_raises_exception_given_max_exceeded(self):
        with pytest.raises(UndesirableMolecule):
            check_value_within_range(6.1, max=6)
