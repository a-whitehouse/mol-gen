import tensorflow as tf


def add_start_and_end_of_sequence_tokens_to_selfies(selfies: tf.Tensor) -> tf.Tensor:
    """Adds start- and end-of-sequence tokens to SELFIES.

    The null token '[nop]' is used as it is ignored by the selfies encoder.

    Args:
        selfies (tf.Tensor): SELFIES to pad.

    Returns:
        tf.Tensor: SELFIES with added tokens.
    """
    return tf.strings.join(["[nop]", selfies, "[nop]"])


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
    return sequence[:-1], sequence[1:]
