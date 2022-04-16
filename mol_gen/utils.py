from rdkit.Chem import CanonSmiles, Mol, MolToSmiles


def check_smiles_equivalent_to_molecule(mol: Mol, smiles: str) -> None:
    """Checks if the SMILES string and molecule are equivalent.

    Args:
        mol (Mol): Molecule to compare.
        smiles (str): SMILES string to compare

    Raises:
        AssertionError: If SMILES string and molecule are not equivalent.
    """
    smiles_1 = CanonSmiles(MolToSmiles(mol))
    smiles_2 = CanonSmiles(smiles)

    assert smiles_1 == smiles_2
