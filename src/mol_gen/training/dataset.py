import tensorflow as tf
from keras.layers import StringLookup
from tensorflow.data import TextLineDataset


def add_start_and_end_of_sequence_tokens_to_selfies(selfies: tf.Tensor) -> tf.Tensor:
    """Adds start- and end-of-sequence tokens to SELFIES.

    The null token '[nop]' is used as it is ignored by the selfies encoder.

    Args:
        selfies (tf.Tensor): SELFIES to pad.

    Returns:
        tf.Tensor: SELFIES with added tokens.
    """
    return tf.strings.join(["[nop]", selfies, "[nop]"])


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


def process_selfies_dataset(
    dataset: TextLineDataset,
    buffer_size: int,
    batch_size: int,
    string_lookup_layer: StringLookup,
):
    """Processes SELFIES dataset for training of model.

    Args:
        dataset (tf.data.Dataset): Dataset of SELFIES.
        buffer_size (int): Buffer size for shuffling dataset.
        batch_size (int): Batch size for padded batch.
        string_lookup_layer (StringLookup): String lookup layer for SELFIES tokens.

    Returns:
        tf.data.Dataset: Processed SELFIES.
    """
    return (
        dataset.shuffle(buffer_size)
        .map(add_start_and_end_of_sequence_tokens_to_selfies)
        .map(split_selfies)
        .padded_batch(batch_size, drop_remainder=True, padding_values="[nop]")
        .map(string_lookup_layer)
        .map(split_sequence_to_input_and_target)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )


def split_selfies(selfies: tf.Tensor) -> tf.Tensor:
    """Splits SELFIES into individual tokens.

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
    """Splits sequence to input and target sequences.

    The target sequence is shifted by one element relative to the input sequence.
    Both sequences are one element shorter than the original sequence.

    Args:
        sequence (tf.Tensor): Sequence to split.

    Returns:
        tuple[tf.Tensor, tf.Tensor]: Input and target sequences.
    """
    return sequence[..., :-1], sequence[..., 1:]
