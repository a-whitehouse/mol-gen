from rdkit.Chem import ChiralType, Mol
from rdkit.Chem.MolStandardize.charge import Uncharger
from rdkit.Chem.MolStandardize.rdMolStandardize import FragmentParent
from rdkit.Chem.SaltRemover import SaltRemover

SALT_REMOVER = SaltRemover()
UNCHARGER = Uncharger()


def neutralise_salts(mol: Mol) -> Mol:
    """Remove counterions and neutralise charges of molecule.

    Args:
        mol (Mol): Molecule to convert.

    Example:
        [Na+]C(=O)[O-] -> O=CO

    Returns:
        Mol: Neutralised molecule.
    """
    # Remove counterions
    mol = SALT_REMOVER.StripMol(mol, dontRemoveEverything=True)

    # Neutralise charges
    mol = UNCHARGER.uncharge(mol)

    return mol


def remove_fragments(mol: Mol) -> Mol:
    """Keep only the largest fragment from a molecule.

    Example:
        ClCCl.c1ccccc1 -> c1ccccc1

    Args:
        mol (Mol): Molecule to convert.

    Returns:
        Mol: Largest fragment.
    """
    mol = FragmentParent(mol)

    return mol


def remove_isotopes(mol: Mol) -> Mol:
    """Remove isotopic labels from molecule.

    Example:
        CC(C)C(C(=O)O)[15N] -> CC(C)C(C(=O)O)[N]

    Args:
        mol (Mol): Molecule to convert.

    Returns:
        Mol: Label free molecule.
    """
    for atom in mol.GetAtoms():
        atom.SetIsotope(0)

    return mol


def remove_stereochemistry(mol: Mol) -> Mol:
    """Remove chiral tags from molecule.

    Example:
        CC(C)[C@@H](C(=O)O)N -> CC(C)C(C(=O)O)N

    Args:
        mol (Mol): Molecule to convert.

    Returns:
        Mol: Achiral molecule.
    """
    for atom in mol.GetAtoms():
        atom.SetChiralTag(ChiralType.CHI_UNSPECIFIED)

    return mol
