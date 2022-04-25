import selfies as sf
from rdkit.Chem import ChiralType, Mol
from rdkit.Chem.MolStandardize.charge import Uncharger
from rdkit.Chem.SaltRemover import SaltRemover
from selfies import EncoderError

SALT_REMOVER = SaltRemover()
UNCHARGER = Uncharger()


def neutralise_salts(mol: Mol) -> Mol:
    """Removes counterions and neutralises charges of molecule.

    Args:
        mol (Mol): Molecule to convert.

    Returns:
        Mol: Neutralised molecule.
    """
    # Remove counterions
    mol = SALT_REMOVER.StripMol(mol, dontRemoveEverything=True)

    # Neutralise charges
    mol = UNCHARGER.uncharge(mol)

    return mol


def remove_isotopes(mol: Mol) -> Mol:
    """Removes isotopic labels from molecule.

    Args:
        mol (Mol): Molecule to convert.

    Returns:
        Mol: Label free molecule.
    """
    for atom in mol.GetAtoms():
        atom.SetIsotope(0)

    return mol


def remove_stereochemistry(mol: Mol) -> Mol:
    """Removes chiral tags from molecule.

    Args:
        mol (Mol): Molecule to convert.

    Returns:
        Mol: Achiral molecule.
    """
    for atom in mol.GetAtoms():
        atom.SetChiralTag(ChiralType.CHI_UNSPECIFIED)

    return mol


def encode_smiles_as_selfies(smiles: str) -> str:
    """Attempt encoding of SMILES string as SELFIES.

    If conversion fails, nothing is returned.

    Args:
        smiles (str): SMILES string.

    Returns:
        str: SELFIES.
    """
    try:
        return sf.encoder(smiles)
    except EncoderError:
        pass
