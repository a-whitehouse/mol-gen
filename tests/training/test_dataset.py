import pytest
import tensorflow as tf
from keras.layers import StringLookup
from numpy.testing import assert_array_equal
from tensorflow.data import TextLineDataset

from mol_gen.training.dataset import (
    add_start_and_end_of_sequence_tokens_to_selfies,
    get_selfies_string_lookup_layer,
    process_selfies_dataset,
    split_selfies,
    split_sequence_to_input_and_target,
)


@pytest.fixture
def dataset(tmpdir):
    selfies = [
        "[C][C][Branch1][#Branch2][C][=N][C][=C][N][=C][Ring1][=Branch1][Cl][C][C][=Branch1][C][=O][N][C][C][Branch1][=Branch1][C][Branch1][C][N][=O][C][C][C][Ring1][=Branch2][Ring2][Ring1][Ring2]\n",
        "[C][N][Branch1][=Branch2][C][C][=C][C][=C][O][Ring1][Branch1][C][=Branch1][C][=O][C][C][S][C][N][Ring1][Branch1][S][=Branch1][C][=O][=Branch1][C][=O][C][=C][C][=C][C][=C][Ring1][=Branch1]\n",
        "[C][N][C][=N][C][Branch1][=Branch2][C][=C][C][=C][N][=C][Ring1][=Branch1][=N][C][Branch1][Ring1][N][C][=C][Ring1][=C][C][=Branch1][C][=O][N][C][=C][C][=C][Branch1][C][O][C][=C][Ring1][#Branch1]\n",
        "[C][C][N][C][=Branch2][Ring1][Branch1][=N][C][C][=Branch1][C][=O][N][C][C][C][=C][C][=C][C][=C][Ring1][=Branch1][C][Ring1][#Branch2][N][C][C][S][C][=C][C][=C][C][=C][Ring1][=Branch1]\n",
        "[C][C][N][Branch1][P][C][=Branch1][C][=O][N][C][C][C][Branch1][C][C][C][=Branch1][C][=O][O][C][Branch1][C][C][C][N][Branch1][C][C][C]\n",
        "[C][N][C][Branch2][Ring1][=Branch2][C][=Branch1][C][=O][N][C][C][=C][Branch1][Ring1][O][C][C][=C][Branch1][Ring1][O][C][C][=C][Ring1][#Branch2][O][C][C][C][=N][N][Branch1][C][C][C][=Ring1][=Branch1]\n",
        "[C][C][NH1][C][=C][C][=C][Branch2][Ring1][=Branch1][C][=Branch1][C][=O][N][C][=C][C][=C][C][Branch1][#Branch1][C][N][Branch1][C][C][C][=C][Ring1][#Branch2][C][=C][Ring2][Ring1][Ring1][C][=Ring2][Ring1][=Branch1][C]\n",
        "[C][N][C][C][C][Branch2][Ring1][#Branch1][N][C][=Branch1][C][=O][N][C][Branch1][=Branch2][C][C][=C][N][=C][NH1][Ring1][Branch1][C][=Branch1][C][=O][O][C][C][Ring2][Ring1][Ring2]\n",
        "[N][#C][C][Branch1][Ring1][C][#N][C][Branch1][Ring1][C][#N][Branch1][Ring1][C][#N][N][C][=C][C][=C][C][=C][Ring1][=Branch1][C][C][C][=C][C][=C][Branch1][Branch1][C][=C][Ring1][=Branch1][C][C][Ring1][=C]\n",
        "[C][C][Branch1][C][C][N][C][C][C][C][Branch2][Ring1][Ring2][N][C][C][N][Branch1][#Branch2][C][C][=C][C][=N][N][Ring1][Branch1][C][C][C][Ring1][=N][C][Ring2][Ring1][Ring1][=O]\n",
    ]

    filepath = tmpdir.join("text")

    with open(filepath, "w") as f:
        f.writelines(selfies)

    return TextLineDataset(filepath)


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


@pytest.fixture
def string_to_integer_layer(vocabulary):
    return StringLookup(mask_token="[nop]", vocabulary=vocabulary)


class TestAddStartAndEndOfSequenceTokensToSelfies:
    def test_completes_given_valid_input(self):
        selfies = tf.constant("[C][C][N][Branch1][P][C][=Branch1][C][=O]")

        add_start_and_end_of_sequence_tokens_to_selfies(selfies)

    def test_returns_expected_selfies(self):
        selfies = tf.constant("[C][C][N][Branch1][P][C][=Branch1][C][=O]")

        actual = add_start_and_end_of_sequence_tokens_to_selfies(selfies)

        assert actual == tf.constant(
            "[nop][C][C][N][Branch1][P][C][=Branch1][C][=O][nop]"
        )


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


class TestProcessSelfiesDataset:
    def test_completes_given_valid_input(self, dataset, string_to_integer_layer):
        process_selfies_dataset(dataset, 10_000, 4, string_to_integer_layer)

    def test_returns_expected_number_of_batches(self, dataset, string_to_integer_layer):
        actual = process_selfies_dataset(dataset, 10_000, 4, string_to_integer_layer)

        count = 0
        for _ in actual.take(3):
            count += 1

        assert count == 2

    def test_returns_batches_of_same_size(self, dataset, string_to_integer_layer):
        actual = process_selfies_dataset(dataset, 10_000, 4, string_to_integer_layer)

        for input_tensor, target_tensor in actual.take(2):
            assert input_tensor.shape[1] is not None
            assert target_tensor.shape[1] is not None

    def test_returns_tensor_pairs(self, dataset, string_to_integer_layer):
        actual = process_selfies_dataset(dataset, 10_000, 4, string_to_integer_layer)

        for i in actual.take(2):
            assert len(i) == 2

    def test_returns_tensor_pairs_with_expected_dtype(
        self, dataset, string_to_integer_layer
    ):
        actual = process_selfies_dataset(dataset, 10_000, 4, string_to_integer_layer)

        for input_tensor, target_tensor in actual.take(2):
            assert input_tensor.dtype is tf.dtypes.int64
            assert target_tensor.dtype is tf.dtypes.int64

    def test_returns_tensor_pairs_that_are_shifted_as_expected(
        self, dataset, string_to_integer_layer
    ):
        actual = process_selfies_dataset(dataset, 10_000, 4, string_to_integer_layer)

        for input_tensor, target_tensor in actual.take(2):
            assert_array_equal(input_tensor[..., 1:], target_tensor[..., :-1])

    def test_returns_input_tensor_with_start_of_sequence_token(
        self, dataset, string_to_integer_layer
    ):
        actual = process_selfies_dataset(dataset, 10_000, 4, string_to_integer_layer)

        for input_tensor, _ in actual.take(2):
            assert all(input_tensor[..., 0] == 0)

    def test_returns_target_tensor_with_end_of_sequence_token(
        self, dataset, string_to_integer_layer
    ):
        actual = process_selfies_dataset(dataset, 10_000, 4, string_to_integer_layer)

        for _, target_tensor in actual.take(2):
            assert all(target_tensor[..., -1] == 0)


class TestSplitSelfies:
    pass


class TestSplitSequenceToInputAndTarget:
    pass
