from __future__ import annotations

from typing import Any

from attrs import frozen

from mol_gen.config.training.dataset import DatasetConfig
from mol_gen.config.training.model import ModelConfig
from mol_gen.utils import read_yaml_config_file


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
        config_dict = read_yaml_config_file(filepath)

        return cls.parse_config(config_dict)
