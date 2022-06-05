from __future__ import annotations

from typing import Any

from attrs import frozen

from mol_gen.exceptions import ConfigException


@frozen
class EvaluateConfig:
    n_molecules: int
    subset_size: int

    @classmethod
    def parse_config(cls, config: dict[str, Any]) -> EvaluateConfig:
        """Parse evaluate section of training config.

        Args:
            config (list[str]): Section of config.

        Raises:
            ConfigException: If required section missing.

        Returns:
            EvaluateConfig: Class representing section of config.
        """
        try:
            return cls(
                n_molecules=config["n_molecules"],
                subset_size=config["subset_size"],
            )

        except KeyError as e:
            raise ConfigException("Required section missing:", e)
