from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

from rdkit.Chem import Mol

from mol_gen.preprocessing.filter import (
    get_preset_check_descriptor_within_range_function,
)


@dataclass
class FilterConfig:
    allowed_elements: list[str]
    range_filters: list[RangeFilter]

    @classmethod
    def parse_config(cls, config: dict[str, Any]) -> FilterConfig:
        return cls(
            allowed_elements=config.get("allowed_elements", []),
            range_filters=[
                RangeFilter.parse_config({k: v})
                for k, v in config.get("range_filters", {}).items()
            ],
        )


@dataclass
class RangeFilter:
    descriptor: str
    min: Optional[Union[int, float]] = None
    max: Optional[Union[int, float]] = None

    @classmethod
    def parse_config(cls, config: dict[str, Any]) -> RangeFilter:
        descriptor = list(config.keys())[0]
        return cls(
            descriptor,
            min=config[descriptor].get("min"),
            max=config[descriptor].get("max"),
        )

    def get_method(self) -> Callable[[Mol], None]:
        return get_preset_check_descriptor_within_range_function(
            self.descriptor, self.min, self.max
        )
