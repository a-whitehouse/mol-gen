from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from rdkit.Chem import Mol

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
        return cls(methods=[CONVERT_METHODS.get(i) for i in config])
