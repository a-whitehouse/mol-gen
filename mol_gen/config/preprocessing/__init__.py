from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mol_gen.config.preprocessing.convert import ConvertConfig
from mol_gen.config.preprocessing.filter import FilterConfig


@dataclass
class PreprocessingConfig:
    convert: ConvertConfig
    filter: FilterConfig

    @classmethod
    def parse_config(cls, config: dict[str, Any]) -> PreprocessingConfig:
        return cls(
            ConvertConfig.parse_config(config.get("convert")),
            FilterConfig.parse_config(config.get("filter")),
        )
