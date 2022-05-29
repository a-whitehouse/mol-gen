from __future__ import annotations

from typing import Any

from attrs import frozen

from mol_gen.config.preprocessing.convert import ConvertConfig
from mol_gen.config.preprocessing.filter import FilterConfig
from mol_gen.config.preprocessing.split import SplitConfig
from mol_gen.utils import read_yaml_config_file


@frozen
class PreprocessingConfig:
    convert: ConvertConfig
    filter: FilterConfig
    split: SplitConfig

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
            SplitConfig.parse_config(config.get("split")),
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
        config_dict = read_yaml_config_file(filepath)

        return cls.parse_config(config_dict)
