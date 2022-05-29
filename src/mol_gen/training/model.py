from __future__ import annotations

from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.data import TextLineDataset
from tensorflow.keras.layers import LSTM, Dense, Embedding, Input

from mol_gen.config.training.model import ModelConfig


def train_model(
    output_dir: Path,
    model: keras.Model,
    training_data: TextLineDataset,
    validation_data: TextLineDataset,
    config: ModelConfig,
) -> None:
    """Trains model on training data.

    Args:
        output_dir (Path): Path to directory to save trained models and logs.
        model (keras.Model): Model to train.
        training_data (TextLineDataset): Data to train model.
        validation_data (TextLineDataset): Data to determine early stopping.
        config (ModelConfig): Config with training parameters.
    """
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=config.patience,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir.joinpath("checkpoints", "model.{epoch:02d}.h5"))
        ),
        tf.keras.callbacks.TensorBoard(log_dir=str(output_dir.joinpath("logs"))),
    ]
    model.fit(
        training_data,
        validation_data=validation_data,
        epochs=config.epochs,
        callbacks=callbacks,
    )


def get_compiled_model(
    config: ModelConfig,
    vocab_size: int,
) -> keras.Model:
    """Get compiled model.

    Args:
        config (ModelConfig): Config with hyperparameters for model layers.
        vocab_size (int): Total size of vocabulary.

    Returns:
        keras.Model: Compiled model.
    """
    input_layer = Input(shape=(None,))

    embedding_layer = Embedding(
        input_dim=vocab_size,
        output_dim=config.embedding_dim,
        mask_zero=True,
    )(input_layer)

    lstm_layer = LSTM(config.lstm_units, return_sequences=True, dropout=config.dropout)(
        embedding_layer
    )

    dense_layer = Dense(vocab_size)(lstm_layer)

    model = tf.keras.Model(inputs=input_layer, outputs=dense_layer)

    model.compile(
        tf.optimizers.Adam(1e-2),
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    )

    return model
