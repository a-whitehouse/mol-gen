from __future__ import annotations

from typing import Any

from attrs import frozen
from yaml import YAMLError, safe_load

from mol_gen.config.training.dataset import DatasetConfig
from mol_gen.config.training.model import ModelConfig
from mol_gen.exceptions import ConfigException


@frozen
class TrainingConfig:
    dataset: DatasetConfig
    model: ModelConfig

    @classmethod
    def parse_config(cls, config: dict[str, Any]) -> TrainingConfig:
        """Parses training config.

        Args:
            config (dict[str, Any]): Config.

        Raises:
            ConfigException: If a section is invalid.

        Returns:
            ConvertConfig: Class representing config.
        """
        return cls(
            DatasetConfig.parse_config(config["dataset"]),
            ModelConfig.parse_config(config["model"]),
        )

    @classmethod
    def from_file(cls, filepath: str) -> TrainingConfig:
        """Parses preprocessing config from file.

        Args:
            filepath (str): Path to config.

        Raises:
            ConfigException: If the file does not exist or conform to valid yaml.

        Returns:
            PreprocessingConfig: Class representing config.
        """
        try:
            with open(filepath) as f:
                config_dict = safe_load(f)
        except FileNotFoundError:
            raise ConfigException(f"File at {filepath} does not exist.")
        except YAMLError as e:
            raise ConfigException(
                f"File at {filepath} does not contain valid yaml: {e}"
            )

        return cls.parse_config(config_dict)
