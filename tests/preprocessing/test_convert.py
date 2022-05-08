import pytest
from rdkit.Chem import Mol, MolFromSmiles

from mol_gen.preprocessing.convert import (
    neutralise_salts,
    remove_fragments,
    remove_isotopes,
    remove_stereochemistry,
)
from mol_gen.utils import check_smiles_equivalent_to_molecule


@pytest.fixture
def mol(smiles: str) -> Mol:
    return MolFromSmiles(smiles)


class TestNeutraliseSalts:
    @pytest.mark.parametrize(
        "smiles",
        ["[Na+].[O-]C(=O)C"],
    )
    def test_completes(self, mol):
        neutralise_salts(mol)

    @pytest.mark.parametrize(
        "smiles",
        ["OC(=O)C"],
    )
    def test_leaves_neutral_molecule_unchanged(self, smiles, mol):
        actual = neutralise_salts(mol)

        check_smiles_equivalent_to_molecule(actual, smiles)

    @pytest.mark.parametrize(
        "smiles, expected",
        [
            ("[Na+].[O-]C(=O)C", "OC(=O)C"),
            ("C(=O)(C(=O)[O-])[O-].[Na+].[Na+]", "C(=O)(C(=O)O)O"),
            ("[NH3+][C@@H](CC1=CC=CC=C1)C([O-])=O", "N[C@@H](CC1=CC=CC=C1)C(O)=O"),
        ],
    )
    def test_converts_molecule_as_expected(self, mol, expected):
        actual = neutralise_salts(mol)

        check_smiles_equivalent_to_molecule(actual, expected)


class TestRemoveFragments:
    @pytest.mark.parametrize(
        "smiles",
        ["CC.CC.CC1(C)CC2=CC=CC=C2C1O"],
    )
    def test_completes(self, mol):
        remove_fragments(mol)

    @pytest.mark.parametrize(
        "smiles",
        ["CC1(C)CC2=CC=CC=C2C1O"],
    )
    def test_leaves_fragment_free_molecule_unchanged(self, smiles, mol):
        actual = remove_fragments(mol)

        check_smiles_equivalent_to_molecule(actual, smiles)

    @pytest.mark.parametrize(
        "smiles, expected",
        [
            ("CC.CC.CC1(C)CC2=CC=CC=C2C1O", "CC1(C)CC2=CC=CC=C2C1O"),
            ("CC1(C)CC2=CC=CC=C2C1O.CC1(C)CC2=CC=CC=C2C1O", "CC1(C)CC2=CC=CC=C2C1O"),
        ],
    )
    def test_converts_molecule_as_expected(self, mol, expected):
        actual = remove_fragments(mol)

        check_smiles_equivalent_to_molecule(actual, expected)


class TestRemoveIsotopes:
    @pytest.mark.parametrize(
        "smiles",
        ["CC(C)C(C(=O)O)[15N]"],
    )
    def test_completes(self, mol):
        remove_isotopes(mol)

    @pytest.mark.parametrize(
        "smiles",
        ["CC(C)C(C(=O)O)N"],
    )
    def test_leaves_label_free_molecule_unchanged(self, smiles, mol):
        actual = remove_isotopes(mol)

        check_smiles_equivalent_to_molecule(actual, smiles)

    @pytest.mark.parametrize(
        "smiles, expected",
        [
            ("CC(C)C(C(=O)O)[15N]", "CC(C)C(C(=O)O)[N]"),
            ("[2H]CC(C)C(C(=O)O)N", "[H]CC(C)C(C(=O)O)N"),
        ],
    )
    def test_converts_molecule_as_expected(self, mol, expected):
        actual = remove_isotopes(mol)

        check_smiles_equivalent_to_molecule(actual, expected)


class TestRemoveStereochemistry:
    @pytest.mark.parametrize(
        "smiles",
        ["CC(C)[C@@H](C(=O)O)N"],
    )
    def test_completes(self, mol):
        remove_stereochemistry(mol)

    @pytest.mark.parametrize(
        "smiles",
        ["CC(C)C(C(=O)O)N"],
    )
    def test_leaves_achiral_molecule_unchanged(self, smiles, mol):
        actual = remove_stereochemistry(mol)

        check_smiles_equivalent_to_molecule(actual, smiles)

    @pytest.mark.parametrize(
        "smiles, expected",
        [
            ("CC(C)[C@@H](C(=O)O)N", "CC(C)C(C(=O)O)N"),
            ("N[C@@H](CC1=CC=CC=C1)C(O)=O", "NC(CC1=CC=CC=C1)C(O)=O"),
        ],
    )
    def test_converts_molecule_as_expected(self, mol, expected):
        actual = remove_stereochemistry(mol)

        check_smiles_equivalent_to_molecule(actual, expected)
