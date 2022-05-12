from __future__ import annotations

from attrs import frozen


@frozen
class ModelConfig:
    embedding_dim: int
    lstm_units: int

    @classmethod
    def parse_config(cls, config: list[str]) -> ModelConfig:
        """Parses model section of training config.

        Args:
            config (list[str]): Section of config.

        Raises:
            ConfigException: If requested convert method unrecognised.

        Returns:
            ConvertConfig: Class representing section of config.
        """
        return cls(
            embedding_dim=config.get("embedding_dim"),
            lstm_units=config.get("lstm_units"),
        )
