import pytest
from rdkit.Chem import Mol, MolFromSmiles

from mol_gen.utils import check_smiles_equivalent_to_molecule


@pytest.fixture
def mol(smiles: str) -> Mol:
    return MolFromSmiles(smiles)


class TestCheckSMILESEquivalentToMolecule:
    @pytest.mark.parametrize(
        "smiles, expected",
        [
            ("CC(C)C(C(=O)O)N", "CC(C(C(=O)O)N)C"),
            ("NC(CC1=CC=CC=C1)C(O)=O", "NC(C(O)=O)CC1=CC=CC=C1"),
        ],
    )
    def test_makes_expected_matches(self, mol, expected):
        check_smiles_equivalent_to_molecule(mol, expected)

    @pytest.mark.parametrize(
        "smiles, expected", [("CC(C)C(C(=O)O)N", "NC(CC1=CC=CC=C1)C(O)=O")]
    )
    def test_raises_exception_for_expected_mismatches(self, mol, expected):
        with pytest.raises(AssertionError):
            check_smiles_equivalent_to_molecule(mol, expected)
