import re

from mol_gen.preprocessing.selfies import encode_smiles_as_selfies


class TestEncodeSmilesAsSelfies:
    def test_completes_given_valid_smiles(self):
        encode_smiles_as_selfies("CCOC(=O)C(C)(C)C1=CC(=C(C=C1)I)C")

    def test_returns_expected_selfies_given_valid_smiles(self):
        actual = encode_smiles_as_selfies("CCOC(=O)C(C)(C)C1=CC(=C(C=C1)I)C")

        assert re.match(r"^(\[.*?\])*$", actual)

    def test_completes_given_invalid_smiles(self):
        encode_smiles_as_selfies("invalid smiles")

    def test_returns_none_given_invalid_smiles(self):
        actual = encode_smiles_as_selfies("invalid smiles")

        assert actual is None
