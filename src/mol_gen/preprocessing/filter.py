from typing import Callable, Optional, Union

from rdkit.Chem import Crippen, Lipinski, Mol, rdMolDescriptors

from mol_gen.exceptions import FilterException, UndesirableMolecule

DESCRIPTOR_TO_FUNCTION: dict[str, Callable[[Mol], Union[int, float]]] = {
    "hydrogen_bond_acceptors": Lipinski.NumHAcceptors,
    "hydrogen_bond_donors": Lipinski.NumHDonors,
    "molar_refractivity": Crippen.MolMR,
    "molecular_weight": rdMolDescriptors.CalcExactMolWt,
    "partition_coefficient": Crippen.MolLogP,
    "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds,
    "topological_polar_surface_area": rdMolDescriptors.CalcTPSA,
}


def check_only_allowed_elements_present(mol: Mol, allowed_elements: list[str]) -> None:
    """Checks if the atoms in a molecule only correspond to allowed elements.

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
    min: Optional[Union[int, float]] = None,
    max: Optional[Union[int, float]] = None,
) -> None:
    """Calculates descriptor of molecule and compares to allowed min and max values.

    Implemented descriptor names are defined in DESCRIPTOR_TO_FUNCTION.
    Args:
        descriptor (str): Name of descriptor to calculate.
        mol (Mol): Molecule to calculate descriptor with.
        min (Optional[float], optional): Minimum allowed value. Defaults to None.
        max (Optional[float], optional): Maximum allowed value. Defaults to None.

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
    val: Union[int, float],
    min: Optional[Union[int, float]] = None,
    max: Optional[Union[int, float]] = None,
):
    """Checks if value is within the allowed min and max values.

    Args:
        val (Union[int, float]): Value to compare.
        min (Optional[float], optional): Minimum allowed value. Defaults to None.
        max (Optional[float], optional): Maximum allowed value. Defaults to None.

    Raises:
        UndesirableMolecule: If descriptor is outside the allowed range of values.
    """
    if (min is not None) and (min > val):
        raise UndesirableMolecule(f"Value {val} less than minimum allowed value {min}.")

    if (max is not None) and (max < val):
        raise UndesirableMolecule(
            f"Value {val} greater than maximum allowed value {max}."
        )