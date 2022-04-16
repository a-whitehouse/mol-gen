from typing import Callable, Optional, Union

from rdkit.Chem import Crippen, Lipinski, Mol, rdMolDescriptors

from mol_gen.exceptions import FilterException, UndesirableMolecule

PROPERTY_TO_FUNCTION: dict[
    str, Callable[[Mol], Union[int, float]], Union[int, float]
] = {
    "hydrogen_bond_acceptors": Lipinski.NumHAcceptors,
    "hydrogen_bond_donors": Lipinski.NumHDonors,
    "molar_refractivity": Crippen.MolMR,
    "molecular_weight": rdMolDescriptors.CalcExactMolWt,
    "partition_coefficient": Crippen.MolLogP,
    "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds,
    "topological_polar_surface_area": rdMolDescriptors.CalcTPSA,
}


def check_property_within_range(
    property: str, mol: Mol, min: Optional[float] = None, max: Optional[float] = None
) -> None:
    """Calculates property of molecule and compares to allowed min and max values.

    Implemented property names are defined in PROPERTY_TO_FUNCTION.
    Raises exception if the property is outside the allowed range of values.
    Args:
        property (str): Name of property to calculate.
        mol (Mol): Molecule to calculate property with.
        min (Optional[float], optional): Minimum allowed value. Defaults to None.
        max (Optional[float], optional): Maximum allowed value. Defaults to None.

    Raises:
        FilterException: If property to calculate is unrecognised.
        UndesirableMolecule: If property is outside the allowed range of values.
    """
    try:
        func = PROPERTY_TO_FUNCTION[property]
    except KeyError:
        raise FilterException(f"Unrecognised property: {property}")

    value = func(mol)
    check_value_within_range(value, min=min, max=max)


def check_value_within_range(
    val: Union[int, float],
    min: Optional[Union[int, float]] = None,
    max: Optional[Union[int, float]] = None,
):
    """Checks if value is within the allowed min and max values.

    Raises exception if the property is outside the allowed range of values.
    Args:
        val (Union[int, float]): Value to compare.
        min (Optional[float], optional): Minimum allowed value. Defaults to None.
        max (Optional[float], optional): Maximum allowed value. Defaults to None.

    Raises:
        UndesirableMolecule: If property is outside the allowed range of values.
    """
    if (min is not None) and (min > val):
        raise UndesirableMolecule

    if (max is not None) and (max < val):
        raise UndesirableMolecule