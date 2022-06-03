from typing import Callable

from rdkit.Chem import AllChem, Crippen, Lipinski, Mol, rdMolDescriptors
from rdkit.DataStructs import TanimotoSimilarity
from rdkit.DataStructs.cDataStructs import UIntSparseIntVect

from mol_gen.exceptions import FilterException, UndesirableMolecule

DESCRIPTOR_TO_FUNCTION: dict[str, Callable[[Mol], int | float]] = {
    "hydrogen_bond_acceptors": Lipinski.NumHAcceptors,
    "hydrogen_bond_donors": Lipinski.NumHDonors,
    "molar_refractivity": Crippen.MolMR,
    "molecular_weight": rdMolDescriptors.CalcExactMolWt,
    "partition_coefficient": Crippen.MolLogP,
    "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds,
    "topological_polar_surface_area": rdMolDescriptors.CalcTPSA,
}


def check_only_allowed_elements_present(mol: Mol, allowed_elements: list[str]) -> None:
    """Check if the atoms in a molecule only correspond to allowed elements.

    Args:
        mol (Mol): Molecule to check.
        allowed_elements (list[str]): Allowed elements.

    Raises:
        UndesirableMolecule: If atoms correspond to other elements.
    """
    for atom in mol.GetAtoms():
        element = atom.GetSymbol()
        if element not in allowed_elements:
            raise UndesirableMolecule(f"Element {element} not in allowed_elements.")


def check_descriptor_within_range(
    mol: Mol,
    descriptor: str,
    min: int | float | None = None,
    max: int | float | None = None,
) -> None:
    """Calculate descriptor of molecule and compare to allowed min and max values.

    Implemented descriptor names are defined in DESCRIPTOR_TO_FUNCTION.

    Args:
        descriptor (str): Name of descriptor to calculate.
        mol (Mol): Molecule to calculate descriptor with.
        min (int | float | None, optional): Minimum allowed value. Defaults to None.
        max (int | float | None, optional): Maximum allowed value. Defaults to None.

    Raises:
        FilterException: If descriptor to calculate is unrecognised.
        UndesirableMolecule: If descriptor is outside the allowed range of values.
    """
    try:
        func = DESCRIPTOR_TO_FUNCTION[descriptor]
    except KeyError:
        raise FilterException(f"Descriptor {descriptor} unrecognised.")

    value = func(mol)
    try:
        check_value_within_range(value, min=min, max=max)
    except UndesirableMolecule as e:
        raise UndesirableMolecule(
            f"Descriptor {descriptor} out of allowed range of values: {e}"
        )


def check_value_within_range(
    val: int | float,
    min: int | float | None = None,
    max: int | float | None = None,
):
    """Check if value is within the allowed min and max values.

    Args:
        val (int | float): Value to compare.
        min (int | float | None, optional): Minimum allowed value. Defaults to None.
        max (int | float | None, optional): Maximum allowed value. Defaults to None.

    Raises:
        UndesirableMolecule: If descriptor is outside the allowed range of values.
    """
    if (min is not None) and (min > val):
        raise UndesirableMolecule(f"Value {val} less than minimum allowed value {min}.")

    if (max is not None) and (max < val):
        raise UndesirableMolecule(
            f"Value {val} greater than maximum allowed value {max}."
        )


def check_tanimoto_score_above_threshold(
    mol: Mol, fingerprint: UIntSparseIntVect, min: int | float
):
    """Compare molecule to Morgan fingerprint by Tanimoto scoring.

    Args:
        mol (Mol): Molecule to check.
        fingerprint (UIntSparseIntVect): Morgan fingerprint to compare against.
        min (int | float): Minimum allowed Tanimoto score.

    Raises:
        UndesirableMolecule: If Tanimoto score is less than the minimum allowed value.
    """
    fp2 = AllChem.GetMorganFingerprint(mol, 2)
    score = TanimotoSimilarity(fingerprint, fp2)

    if score < min:
        raise UndesirableMolecule(
            f"Tanimoto score {score} less than minimum allowed value {min}."
        )
