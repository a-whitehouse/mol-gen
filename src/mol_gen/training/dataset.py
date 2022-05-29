from pathlib import Path

import tensorflow as tf
from keras.layers import StringLookup
from tensorflow.data import TextLineDataset
from tensorflow.python.data.ops.dataset_ops import PrefetchDataset

from mol_gen.config.training.dataset import DatasetConfig


def add_start_and_end_of_sequence_tokens_to_selfies(selfies: tf.Tensor) -> tf.Tensor:
    """Add start- and end-of-sequence tokens to SELFIES.

    The null token '[nop]' is used as it is ignored by the selfies encoder.

    Args:
        selfies (tf.Tensor): SELFIES to pad.

    Returns:
        tf.Tensor: SELFIES with added tokens.
    """
    return tf.strings.join(["[nop]", selfies, "[nop]"])


def get_selfies_dataset(
    input_dir: Path, config: DatasetConfig, string_lookup_layer: StringLookup
) -> PrefetchDataset:
    """Read all files from the directory into a TensorFlow dataset with pipeline.

    Args:
        input_dir (Path): Path to directory to read data as text.
        config (DatasetConfig): Config with buffer size and batch size.
        string_lookup_layer (StringLookup): String lookup layer for SELFIES tokens.

    Returns:
        PrefetchDataset: Processed SELFIES dataset.
    """
    filepath_pattern = str(input_dir.joinpath("*"))
    data = TextLineDataset(TextLineDataset.list_files(filepath_pattern))
    return process_selfies_dataset(
        data,
        config.buffer_size,
        config.batch_size,
        string_lookup_layer,
    )


def process_selfies_dataset(
    dataset: TextLineDataset,
    buffer_size: int,
    batch_size: int,
    string_lookup_layer: StringLookup,
) -> PrefetchDataset:
    """Process SELFIES dataset for training of model.

    Args:
        dataset (tf.data.Dataset): Dataset of SELFIES.
        buffer_size (int): Buffer size for shuffling dataset.
        batch_size (int): Batch size for padded batch.
        string_lookup_layer (StringLookup): String lookup layer for SELFIES tokens.

    Returns:
        tf.data.Dataset: Processed SELFIES dataset.
    """
    return (
        dataset.shuffle(buffer_size)
        .map(add_start_and_end_of_sequence_tokens_to_selfies)
        .map(split_selfies)
        .padded_batch(batch_size, padding_values="[nop]")
        .map(string_lookup_layer)
        .map(split_sequence_to_input_and_target)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )


def split_selfies(selfies: tf.Tensor) -> tf.Tensor:
    """Split SELFIES into individual tokens.

    Args:
        selfies (tf.Tensor): SELFIES to split.

    Returns:
        tf.Tensor: Individual tokens.
    """
    selfies = tf.strings.regex_replace(selfies, r"\](\.?)\[", r"] \1 [")
    return tf.strings.split(selfies)


def split_sequence_to_input_and_target(
    sequence: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Split sequence to input and target sequences.

    The target sequence is shifted by one element relative to the input sequence.
    Both sequences are one element shorter than the original sequence.

    Args:
        sequence (tf.Tensor): Sequence to split.

    Returns:
        tuple[tf.Tensor, tf.Tensor]: Input and target sequences.
    """
    return sequence[..., :-1], sequence[..., 1:]
