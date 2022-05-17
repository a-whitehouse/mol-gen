from __future__ import annotations

from typing import Any

from attrs import field, frozen

from mol_gen.exceptions import ConfigException
from mol_gen.utils import assign_to_split


def validate_set_size(instance, attribute, value):
    if not isinstance(value, (float)) or (value <= 0) or (value >= 1):
        raise ConfigException(
            f"Size for {attribute.name} set should be a number between 0 and 1."
        )


@frozen
class SplitConfig:
    validate: float = field(validator=validate_set_size)
    test: float = field(validator=validate_set_size)

    def __attrs_post_init__(self):
        if (self.validate + self.test) >= 1:
            raise ConfigException(
                "Total size for calibrate and test sets should be less than 1."
            )

    @classmethod
    def parse_config(cls, config: dict[str, Any]) -> SplitConfig:
        """Parses split section of preprocessing config.

        Args:
            config (dict[str, Any]): Section of config.

        Raises:
            ConfigException: If a section is invalid.

        Returns:
            SplitConfig: Class representing section of config.
        """
        return cls(validate=config.get("validate"), test=config.get("test"))

    def apply(self) -> str:
        """Selects set at random from train/validate/test.

        Returns:
            str: Assigned set.
        """
        return assign_to_split(self.validate, self.test)
