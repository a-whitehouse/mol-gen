from __future__ import annotations

from typing import Any

from attrs import frozen


@frozen
class DatasetConfig:
    buffer_size: int
    batch_size: int

    @classmethod
    def parse_config(cls, config: dict[str, Any]) -> DatasetConfig:
        """Parses dataset section of training config.

        Args:
            config (list[str]): Section of config.

        Raises:
            ConfigException: If requested convert method unrecognised.

        Returns:
            ConvertConfig: Class representing section of config.
        """
        return cls(buffer_size=config["buffer_size"], batch_size=config["batch_size"])
