import json
from pathlib import Path

from keras.layers import StringLookup


def get_selfies_string_lookup_layer(
    vocabulary: list[str], invert: bool = False
) -> StringLookup:
    """Get string lookup layer from vocabulary for SELFIES.

    The null token '[nop]' is used for the mask_token,
    as it is ignored by the selfies encoder.

    Args:
        vocabulary (list[str]): Vocabulary for layer.
        invert (bool, optional): Whether to map integers to strings. Defaults to False.

    Returns:
        StringLookup: Configured string lookup layer.
    """
    return StringLookup(
        mask_token="[nop]",
        vocabulary=[i for i in vocabulary if i != "[nop]"],
        invert=invert,
    )


def read_string_lookup_from_json(filepath: Path | str, invert: bool = False) -> None:
    """Read string lookup layer from config JSON file.

    Args:
        filepath (Path | str): Path to config.
        invert (bool, optional): Whether to map integers to strings. Defaults to False.
    """
    with open(filepath) as fh:
        config = json.load(fh)

    config[invert] = invert

    return StringLookup.from_config(config)


def write_string_lookup_to_json(
    string_lookup_layer: StringLookup,
    filepath: Path | str,
) -> None:
    """Write string lookup layer to config JSON file.

    Args:
        string_lookup_layer (StringLookup): Configured string lookup layer.
        filepath (Path | str): Path to config.
    """
    config = string_lookup_layer.get_config()

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w") as fh:
        json.dump(config, fh)
