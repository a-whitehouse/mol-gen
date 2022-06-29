from __future__ import annotations

from typing import Any

from attrs import field, frozen

from mol_gen.exceptions import ConfigException


@frozen
class ModelConfig:
    embedding_dim: int
    lstm_layers: list[LSTMLayerConfig]
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
                lstm_layers=[
                    LSTMLayerConfig.parse_config(i) for i in config["lstm_layers"]
                ],
                patience=config.get("patience", 5),
                epochs=config.get("epochs", 100),
            )

        except KeyError as e:
            raise ConfigException("Required section missing:", e)


@frozen
class LSTMLayerConfig:
    units: int
    dropout: float = field()

    @dropout.validator
    def _check_dropout(self, attribute, value):
        if not (0 <= value < 1):
            raise ConfigException(
                f"Dropout {value} for LSTM layer is not within the interval [0, 1)."
            )

    @classmethod
    def parse_config(cls, config: dict[str, Any]) -> LSTMLayerConfig:
        """Parse LSTM layer section of training config.

        Args:
            config (list[str]): Section of config.

        Raises:
            ConfigException: If required section missing.

        Returns:
            LSTMLayerConfig: Class representing section of config.
        """
        try:
            return cls(
                units=config["units"],
                dropout=config.get("dropout", 0.0),
            )

        except KeyError as e:
            raise ConfigException("Required section missing:", e)
