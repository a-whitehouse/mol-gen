import selfies as sf
from selfies import EncoderError


def encode_smiles_as_selfies(smiles: str) -> str:
    """Attempt encoding of SMILES string as SELFIES.

    If conversion fails, nothing is returned.

    Args:
        smiles (str): SMILES string.

    Returns:
        str: SELFIES.
    """
    try:
        return sf.encoder(smiles)
    except EncoderError:
        pass
