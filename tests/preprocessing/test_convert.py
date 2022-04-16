import pytest
from rdkit.Chem import Mol, MolFromSmiles

from mol_gen.preprocessing.convert import neutralise_salts, remove_stereochemistry
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