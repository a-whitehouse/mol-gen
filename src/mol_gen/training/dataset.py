import tensorflow as tf


def split_selfies(selfies: tf.Tensor) -> tf.Tensor:
    """Splits SELFIES into individual tokens.

    Args:
        selfies (tf.Tensor): SELFIES to split.

    Returns:
        tf.Tensor: Individual tokens.
    """
    selfies = tf.strings.regex_replace(selfies, r"\]\[", "] [")
    selfies = tf.strings.regex_replace(selfies, r"\]\.\[", "] . [")
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
