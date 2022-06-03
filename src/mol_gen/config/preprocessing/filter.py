from __future__ import annotations

from typing import Any

from attrs import field, frozen
from rdkit.Chem import AllChem, GetPeriodicTable, Mol, MolFromSmiles

from mol_gen.exceptions import ConfigException
from mol_gen.preprocessing.filter import (
    DESCRIPTOR_TO_FUNCTION,
    check_descriptor_within_range,
    check_only_allowed_elements_present,
    check_tanimoto_score_above_threshold,
)


@frozen
class FilterConfig:
    elements_filter: ElementsFilter
    range_filters: list[RangeFilter]
    structure_filters: list[StructureFilter]

    @classmethod
    def parse_config(cls, config: dict[str, Any]) -> FilterConfig:
        """Parse filter section of preprocessing config.

        Args:
            config (dict[str, Any]): Section of config.

        Raises:
            ConfigException: If a section is invalid.

        Returns:
            FilterConfig: Class representing section of config.
        """
        return cls(
            elements_filter=ElementsFilter.parse_config(
                config.get("allowed_elements", [])
            ),
            range_filters={
                k: RangeFilter.parse_config({k: v})
                for k, v in config.get("range_filters", {}).items()
            },
            structure_filters=[
                StructureFilter.parse_config(i)
                for i in config.get("structure_filters", [])
            ],
        )

    def apply(self, mol: Mol) -> None:
        """Apply elements and range filter methods to molecule.

        Args:
            mol (Mol): Molecule to check.

        Raises:
            UndesirableMolecule: If molecule fails a filter.
        """
        self.elements_filter.apply(mol)

        for filter in self.range_filters.values():
            filter.apply(mol)

        for filter in self.structure_filters:
            filter.apply(mol)


@frozen
class ElementsFilter:
    allowed_elements: list[str] = field()

    @allowed_elements.validator
    def _check_allowed_elements(self, attribute, value):
        periodic_table = GetPeriodicTable()

        for element in value:
            try:
                periodic_table.GetAtomicNumber(element)

            except RuntimeError:
                raise ConfigException(f"Element {element} is not recognised.")

    @classmethod
    def parse_config(cls, config: list[str]) -> ElementsFilter:
        """Parse allowed_elements section of preprocessing config.

        Args:
            config (list[str]): Section of config.

        Raises:
            ConfigException: If an element is unrecognised.

        Returns:
            ElementsFilter: Class representing section of config.
        """
        return cls(config)

    def apply(self, mol: Mol) -> None:
        """Apply filter method to molecule.

        Checks whether atoms of the molecule only correspond to allowed elements.

        Args:
            mol (Mol): Molecule to check.

        Raises:
            UndesirableMolecule: If atoms correspond to other elements.
        """
        check_only_allowed_elements_present(mol, allowed_elements=self.allowed_elements)


@frozen
class RangeFilter:
    descriptor: str = field()
    min: int | float | None = field(default=None)
    max: int | float | None = field(default=None)

    @descriptor.validator
    def _check_descriptor(self, attribute, value):
        if value not in DESCRIPTOR_TO_FUNCTION:
            raise ConfigException(f"Descriptor {value} is not recognised.")

    @min.validator
    def _check_min(self, attribute, value):
        if (value is not None) and not isinstance(value, (float, int)):
            raise ConfigException(
                f"Minimum value {value} for {self.descriptor} is not a number."
            )

    @max.validator
    def _check_max(self, attribute, value):
        if (value is not None) and not isinstance(value, (float, int)):
            raise ConfigException(
                f"Maximum value {value} for {self.descriptor} is not a number."
            )

    @classmethod
    def parse_config(cls, config: dict[str, Any]) -> RangeFilter:
        """Parse range filter in range_filters section of preprocessing config.

        Args:
            config (list[str]): Section of config.

        Raises:
            ConfigException: If a descriptor is unrecognised, or min/max value invalid.

        Returns:
            RangeFilter: Class representing section of config.
        """
        descriptor = list(config.keys())[0]

        return cls(
            descriptor,
            min=config[descriptor].get("min"),
            max=config[descriptor].get("max"),
        )

    def apply(self, mol: Mol) -> None:
        """Apply filter method to molecule.

        Calculates descriptor of molecule and compares to allowed min and max values.

        Args:
            mol (Mol): Molecule to check.

        Raises:
            UndesirableMolecule: If descriptor is outside the allowed range of values.
        """
        check_descriptor_within_range(mol, self.descriptor, min=self.min, max=self.max)


class StructureFilter:
    smiles: str = field()
    min: int | float = field()

    @smiles.validator
    def _check_smiles(self, attribute, value):
        try:
            mol = MolFromSmiles(value)

        except TypeError:
            mol = None

        if mol is None:
            raise ConfigException(f"SMILES string {value} could not be parsed.")

    @min.validator
    def _check_min(self, attribute, value):
        if not isinstance(value, (float, int)):
            raise ConfigException(
                f"Minimum structural similarity score {value} for {self.smiles} "
                "is not a number."
            )
        elif not (0 < value <= 1):
            raise ConfigException(
                f"Minimum structural similarity score {value} for {self.smiles} "
                "is not within the interval (0, 1]."
            )

    def __attrs_post_init__(self):
        mol = MolFromSmiles(self.smiles)
        self._fp = AllChem.GetMorganFingerprint(mol, 2)

    @classmethod
    def parse_config(cls, config: dict[str, Any]) -> StructureFilter:
        """Parse structure filter in structure_filters section of preprocessing config.

        Args:
            config (dict[str, Any]): Section of config.

        Raises:
            ConfigException: If the SMILES string cannot be parsed, or minimum invalid.

        Returns:
            StructureFilter: Class representing section of config.
        """
        return cls(
            smiles=config["smiles"],
            min=config["min"],
        )

    def apply(self, mol: Mol) -> None:
        """Apply structural similarity method to molecule.

        Compares molecule to Morgan fingerprint by Tanimoto scoring.

        Args:
            mol (Mol): Molecule to check.

        Raises:
            UndesirableMolecule: If Tanimoto score less than the minimum allowed value.
        """
        check_tanimoto_score_above_threshold(mol, self._fp, min=self.min)
