from __future__ import annotations

from attrs import frozen


@frozen
class DatasetConfig:
    buffer_size: int
    batch_size: int

    @classmethod
    def parse_config(cls, config: list[str]) -> DatasetConfig:
        """Parses dataset section of training config.

        Args:
            config (list[str]): Section of config.

        Raises:
            ConfigException: If requested convert method unrecognised.

        Returns:
            ConvertConfig: Class representing section of config.
        """
        return cls(
            buffer_size=config.get("buffer_size"), batch_size=config.get("batch_size")
        )
