from __future__ import annotations

from typing import Any, Optional, Union

from attr import frozen
from rdkit.Chem import GetPeriodicTable, Mol

from mol_gen.exceptions import ConfigException
from mol_gen.preprocessing.filter import (
    DESCRIPTOR_TO_FUNCTION,
    check_descriptor_within_range,
    check_only_allowed_elements_present,
)


@frozen
class FilterConfig:
    elements_filter: ElementsFilter
    range_filters: list[RangeFilter]

    @classmethod
    def parse_config(cls, config: dict[str, Any]) -> FilterConfig:
        """Parse filter section of preprocessing config.

        Args:
            config (list[str]): Section of config.

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
        )

    def apply(self, mol: Mol) -> None:
        """Applies elements and range filter methods to molecule.

        Args:
            mol (Mol): Molecule to check.

        Raises:
            UndesirableMolecule: If molecule fails a filter.
        """
        self.elements_filter.apply(mol)

        for filter in self.range_filters.values():
            filter.apply(mol)


@frozen
class ElementsFilter:
    allowed_elements: list[str]

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
        periodic_table = GetPeriodicTable()
        valid_elements = []

        for element in config:
            try:
                periodic_table.GetAtomicNumber(element)
                valid_elements.append(element)

            except RuntimeError:
                raise ConfigException(f"Element {element} is not recognised.")

        return cls(valid_elements)

    def apply(self, mol: Mol) -> None:
        """Applies filter method to molecule.

        Checks whether atoms of the molecule only correspond to allowed elements.

        Args:
            mol (Mol): Molecule to check.

        Raises:
            UndesirableMolecule: If atoms correspond to other elements.
        """
        check_only_allowed_elements_present(mol, allowed_elements=self.allowed_elements)


@frozen
class RangeFilter:
    descriptor: str
    min: Optional[Union[int, float]] = None
    max: Optional[Union[int, float]] = None

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
        if descriptor not in DESCRIPTOR_TO_FUNCTION:
            raise ConfigException(f"Descriptor {descriptor} is not recognised.")

        min = config[descriptor].get("min")
        if min and not isinstance(min, (float, int)):
            raise ConfigException(
                f"Minimum value {min} for {descriptor} is not a number."
            )

        max = config[descriptor].get("max")
        if max and not isinstance(max, (float, int)):
            raise ConfigException(
                f"Maximum value {max} for {descriptor} is not a number."
            )

        return cls(
            descriptor,
            min=min,
            max=max,
        )

    def apply(self, mol: Mol) -> None:
        """Applies filter method to molecule.

        Calculates descriptor of molecule and compares to allowed min and max values.

        Args:
            mol (Mol): Molecule to check.

        Raises:
            UndesirableMolecule: If descriptor is outside the allowed range of values.
        """
        check_descriptor_within_range(mol, self.descriptor, min=self.min, max=self.max)
