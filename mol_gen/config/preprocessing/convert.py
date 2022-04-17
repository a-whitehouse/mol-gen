from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from rdkit.Chem import Mol

from mol_gen.exceptions import ConfigException
from mol_gen.preprocessing.convert import neutralise_salts, remove_stereochemistry

CONVERT_METHODS = {
    "neutralise_salts": neutralise_salts,
    "remove_stereochemistry": remove_stereochemistry,
}


@dataclass
class ConvertConfig:
    methods: list[Callable[[Mol], Mol]]

    @classmethod
    def parse_config(cls, config: list[str]) -> ConvertConfig:
        """Parse convert methods section of preprocessing config.

        Args:
            config (list[str]): Section of config.

        Raises:
            ConfigException: If requested convert method unrecognised.

        Returns:
            ConvertConfig: Class representing section of config.
        """
        methods = []

        for method in config:
            try:
                methods.append(CONVERT_METHODS[method])
            except KeyError:
                raise ConfigException(f"Convert method {method} unrecognised.")

        return cls(methods=methods)
