from typing import Optional, Union

from rdkit.Chem import Mol
from rdkit.Chem.Crippen import MolLogP, MolMR
from rdkit.Chem.Lipinski import NumHAcceptors, NumHDonors
from rdkit.Chem.rdMolDescriptors import CalcExactMolWt, CalcNumRotatableBonds, CalcTPSA

from mol_gen.exceptions import UndesirableMolecule


def check_hydrogen_bond_acceptors(
    mol: Mol, min: Optional[float] = None, max: Optional[float] = None
) -> None:
    value = NumHAcceptors(mol)
    check_value_within_range(value, min=min, max=max)


def check_hydrogen_bond_donors(
    mol: Mol, min: Optional[float] = None, max: Optional[float] = None
) -> None:
    value = NumHDonors(mol)
    check_value_within_range(value, min=min, max=max)


def check_molar_refractivity(
    mol: Mol, min: Optional[float] = None, max: Optional[float] = None
) -> None:
    value = MolMR(mol)
    check_value_within_range(value, min=min, max=max)


def check_molecular_weight(
    mol: Mol, min: Optional[float] = None, max: Optional[float] = None
) -> None:
    value = CalcExactMolWt(mol)
    check_value_within_range(value, min=min, max=max)


def check_partition_coefficient(
    mol: Mol, min: Optional[float] = None, max: Optional[float] = None
) -> None:
    value = MolLogP(mol)
    check_value_within_range(value, min=min, max=max)


def check_rotatable_bonds(
    mol: Mol, min: Optional[float] = None, max: Optional[float] = None
) -> None:
    value = CalcNumRotatableBonds(mol)
    check_value_within_range(value, min=min, max=max)


def check_topological_polar_surface_area(
    mol: Mol, min: Optional[float] = None, max: Optional[float] = None
) -> None:
    value = CalcTPSA(mol)
    check_value_within_range(value, min=min, max=max)


def check_value_within_range(
    val: Union[int, float],
    min: Optional[Union[int, float]] = None,
    max: Optional[Union[int, float]] = None,
):
    if (min is not None) and (min > val):
        raise UndesirableMolecule

    if (max is not None) and (max < val):
        raise UndesirableMolecule
