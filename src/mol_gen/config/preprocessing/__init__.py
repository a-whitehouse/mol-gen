from __future__ import annotations

from typing import Any

from attrs import frozen
from yaml import YAMLError, safe_load

from mol_gen.config.preprocessing.convert import ConvertConfig
from mol_gen.config.preprocessing.filter import FilterConfig
from mol_gen.exceptions import ConfigException


@frozen
class PreprocessingConfig:
    convert: ConvertConfig
    filter: FilterConfig

    @classmethod
    def parse_config(cls, config: dict[str, Any]) -> PreprocessingConfig:
        """Parse preprocessing config.

        Args:
            config (dict[str, Any]): Config.

        Raises:
            ConfigException: If a section is invalid.

        Returns:
            ConvertConfig: Class representing config.
        """
        return cls(
            ConvertConfig.parse_config(config.get("convert")),
            FilterConfig.parse_config(config.get("filter")),
        )

    @classmethod
    def from_file(cls, filepath: str) -> PreprocessingConfig:
        """Parse preprocessing config from file.

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
