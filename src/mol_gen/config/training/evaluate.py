from __future__ import annotations

from typing import Any

from attrs import frozen


@frozen
class EvaluateConfig:
    n_molecules: int
    subset_size: int

    @classmethod
    def parse_config(cls, config: dict[str, Any]) -> EvaluateConfig:
        """Parse evaluate section of training config.

        Args:
            config (list[str]): Section of config.

        Returns:
            EvaluateConfig: Class representing section of config.
        """
        return cls(
            n_molecules=config.get("n_molecules", 1024),
            subset_size=config.get("subset_size", 50),
        )
