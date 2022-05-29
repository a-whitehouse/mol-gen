import pytest
from tensorflow.data import TextLineDataset

from mol_gen.training.string_lookup import get_selfies_string_lookup_layer


@pytest.fixture
def dataset(train_filepath):
    return TextLineDataset(train_filepath)


@pytest.fixture
def vocabulary():
    return [
        "[#Branch1]",
        "[#Branch2]",
        "[#C]",
        "[#N]",
        "[=Branch1]",
        "[=Branch2]",
        "[=C]",
        "[=N]",
        "[=O]",
        "[=Ring1]",
        "[=Ring2]",
        "[Branch1]",
        "[Branch2]",
        "[C]",
        "[Cl]",
        "[NH1]",
        "[N]",
        "[O]",
        "[P]",
        "[Ring1]",
        "[Ring2]",
        "[S]",
    ]


class TestGetSelfiesStringLookupLayer:
    def test_completes_given_valid_input(self, vocabulary):
        get_selfies_string_lookup_layer(vocabulary)

    def test_adds_vocabulary_to_layer_vocabulary(self, vocabulary):
        layer = get_selfies_string_lookup_layer(vocabulary)

        actual = layer.get_vocabulary()

        assert actual[2:] == [
            "[#Branch1]",
            "[#Branch2]",
            "[#C]",
            "[#N]",
            "[=Branch1]",
            "[=Branch2]",
            "[=C]",
            "[=N]",
            "[=O]",
            "[=Ring1]",
            "[=Ring2]",
            "[Branch1]",
            "[Branch2]",
            "[C]",
            "[Cl]",
            "[NH1]",
            "[N]",
            "[O]",
            "[P]",
            "[Ring1]",
            "[Ring2]",
            "[S]",
        ]

    def test_adds_mask_token_to_layer_vocabulary(self, vocabulary):
        layer = get_selfies_string_lookup_layer(vocabulary)

        actual = layer.get_vocabulary()

        assert actual[0] == "[nop]"

    def test_ignores_extra_mask_token_in_vocabulary(self, vocabulary):
        vocabulary.append("[nop]")
        layer = get_selfies_string_lookup_layer(vocabulary)

        actual = layer.get_vocabulary()

        assert "[nop]" not in actual[1:]
