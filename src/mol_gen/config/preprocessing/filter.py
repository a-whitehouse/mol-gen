from __future__ import annotations

from typing import Any, Callable, Optional, Union

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

    def get_method(self) -> Callable[[Mol], None]:
        """Gets filter method that only takes molecule, with other arguments preset.

        Returns:
            Callable[[Mol], None]: Filter method.
        """

        def preset_check_only_allowed_elements_present(mol: Mol):
            return check_only_allowed_elements_present(
                mol, allowed_elements=self.allowed_elements
            )

        return preset_check_only_allowed_elements_present


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

    def get_method(self) -> Callable[[Mol], None]:
        """Gets filter method that only takes molecule, with other arguments preset.

        Returns:
            Callable[[Mol], None]: Filter method.
        """

        def preset_check_descriptor_within_range(mol: Mol):
            return check_descriptor_within_range(
                mol, self.descriptor, min=self.min, max=self.max
            )

        return preset_check_descriptor_within_range
