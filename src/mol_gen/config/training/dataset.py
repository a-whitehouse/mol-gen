from __future__ import annotations

from typing import Any

from attrs import frozen

from mol_gen.exceptions import ConfigException


@frozen
class DatasetConfig:
    buffer_size: int
    batch_size: int

    @classmethod
    def parse_config(cls, config: dict[str, Any]) -> DatasetConfig:
        """Parse dataset section of training config.

        Args:
            config (list[str]): Section of config.

        Raises:
            ConfigException: If required section missing.

        Returns:
            DatasetConfig: Class representing section of config.
        """
        try:
            return cls(
                buffer_size=config["buffer_size"], batch_size=config["batch_size"]
            )

        except KeyError as e:
            raise ConfigException("Required section missing:", e)
