from __future__ import annotations

from typing import Any

from attrs import frozen

from mol_gen.exceptions import ConfigException


@frozen
class ModelConfig:
    embedding_dim: int
    lstm_units: int
    dropout: float
    patience: int
    epochs: int

    @classmethod
    def parse_config(cls, config: dict[str, Any]) -> ModelConfig:
        """Parse model section of training config.

        Args:
            config (list[str]): Section of config.

        Raises:
            ConfigException: If required section missing.

        Returns:
            ModelConfig: Class representing section of config.
        """
        try:
            return cls(
                embedding_dim=config["embedding_dim"],
                lstm_units=config["lstm_units"],
                dropout=config.get("dropout", 0),
                patience=config.get("patience", 5),
                epochs=config.get("epochs", 100),
            )

        except KeyError as e:
            raise ConfigException("Required section missing:", e)
