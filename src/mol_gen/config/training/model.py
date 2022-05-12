from __future__ import annotations

from typing import Any

from attrs import frozen

from mol_gen.exceptions import ConfigException


@frozen
class ModelConfig:
    embedding_dim: int
    lstm_units: int
    dropout: float

    @classmethod
    def parse_config(cls, config: dict[str, Any]) -> ModelConfig:
        """Parses model section of training config.

        Args:
            config (list[str]): Section of config.

        Raises:
            ConfigException: If requested convert method unrecognised.

        Returns:
            ConvertConfig: Class representing section of config.
        """
        try:
            return cls(
                embedding_dim=config["embedding_dim"],
                lstm_units=config["lstm_units"],
                dropout=config.get("dropout", 0),
            )

        except KeyError as e:
            raise ConfigException("Required section missing:", e)
