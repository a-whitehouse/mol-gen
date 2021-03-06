from random import random
from typing import Any

from rdkit.Chem import CanonSmiles, Mol, MolToSmiles
from yaml import YAMLError, safe_load

from mol_gen.exceptions import ConfigException


def assign_to_split(validate_size: float, test_size: float) -> str:
    """Select set at random from train/validate/test.

    Args:
        validate_size (float): Validation set proportion.
        test_size (float): Test set proportion.

    Returns:
        str: Assigned set.
    """
    train_size = 1 - (validate_size + test_size)
    value = random()

    if value < train_size:
        return "train"

    elif value < 1 - test_size:
        return "validate"

    else:
        return "test"


def check_smiles_equivalent_to_molecule(mol: Mol, smiles: str) -> None:
    """Check if the SMILES string and molecule are equivalent.

    Args:
        mol (Mol): Molecule to compare.
        smiles (str): SMILES string to compare

    Raises:
        AssertionError: If SMILES string and molecule are not equivalent.
    """
    smiles_1 = CanonSmiles(MolToSmiles(mol))
    smiles_2 = CanonSmiles(smiles)

    assert smiles_1 == smiles_2


def read_yaml_config_file(filepath: str) -> dict[str, Any]:
    """Read yaml config from file.

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
        raise ConfigException(f"File at {filepath} does not contain valid yaml: {e}")

    return config_dict
