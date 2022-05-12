from __future__ import annotations

from typing import Callable

from attrs import frozen
from rdkit.Chem import Mol

from mol_gen.exceptions import ConfigException, ConvertException
from mol_gen.preprocessing.convert import (
    neutralise_salts,
    remove_fragments,
    remove_isotopes,
    remove_stereochemistry,
)

CONVERT_METHODS = {
    "neutralise_salts": neutralise_salts,
    "remove_fragments": remove_fragments,
    "remove_isotopes": remove_isotopes,
    "remove_stereochemistry": remove_stereochemistry,
}


@frozen
class ConvertConfig:
    methods: list[Callable[[Mol], Mol]]

    @classmethod
    def parse_config(cls, config: list[str]) -> ConvertConfig:
        """Parses convert methods section of preprocessing config.

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

    def apply(self, mol: Mol) -> Mol:
        """Applies convert methods to molecule.

        Args:
            mol (Mol): Molecule to convert.

        Raises:
            ConvertException: If molecule fails a method.

        Returns:
            Mol: Converted molecule.
        """
        for method in self.methods:
            try:
                mol = method(mol)
            except Exception as e:
                raise ConvertException(f"Convert method {method} failed: {e}")

        return mol
