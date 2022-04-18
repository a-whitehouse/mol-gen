from rdkit.Chem import ChiralType, Mol
from rdkit.Chem.MolStandardize.charge import Uncharger
from rdkit.Chem.SaltRemover import SaltRemover

SALT_REMOVER = SaltRemover()
UNCHARGER = Uncharger()


def neutralise_salts(mol: Mol) -> Mol:
    """Removes salts and neutralises charges of remaining molecule.

    Args:
        mol (Mol): Molecule to neutralise.

    Returns:
        Mol: Neutralised molecule.
    """
    # Remove counterions
    mol = SALT_REMOVER.StripMol(mol, dontRemoveEverything=True)

    # Neutralise charges
    mol = UNCHARGER.uncharge(mol)

    return mol


def remove_stereochemistry(mol: Mol) -> Mol:
    """Removes chiral tags of molecule.

    Args:
        mol (Mol): Molecule to remove chiral tags.

    Returns:
        Mol: Achiral molecule.
    """
    for atom in mol.GetAtoms():
        atom.SetChiralTag(ChiralType.CHI_UNSPECIFIED)

    return mol
