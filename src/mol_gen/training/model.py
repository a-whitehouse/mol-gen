from __future__ import annotations

from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.data import TextLineDataset
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import LSTM, Dense, Embedding, Input

from mol_gen.config.training import TrainingConfig
from mol_gen.config.training.model import ModelConfig
from mol_gen.training.evaluate import ReportCheckpoint


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
    model = keras.Sequential()
    model.add(Input(shape=(None,)))

    model.add(
        Embedding(
            input_dim=vocab_size,
            output_dim=config.embedding_dim,
            mask_zero=True,
        )
    )

    for layer_config in config.lstm_layers:
        model.add(
            LSTM(
                layer_config.units, return_sequences=True, dropout=layer_config.dropout
            )
        )

    model.add(Dense(vocab_size))

    model.compile(
        tf.optimizers.Adam(1e-2),
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    )

    return model


def get_callbacks(
    checkpoint_dir: Path | str,
    log_dir: Path | str,
    report_dir: Path | str,
    train_dir: Path | str,
    report_template_path: Path | str,
    string_lookup_path: Path | str,
    config: TrainingConfig,
) -> list[Callback]:
    """Get callbacks to add to model training loop.

    Args:
        checkpoint_dir (Path | str): Path to directory to save trained models.
        log_dir (Path | str):  Path to directory to write logs.
        report_dir (Path | str): Path to directory to write output HTML reports.
        train_dir (Path | str): Path to training set directory to read SELFIES as text.
        report_template_path (Path | str): Path to model evaluation report template.
        string_lookup_path (Path | str): Path to string lookup config.
        config (TrainingConfig): Config with training and evaluation parameters.

    Returns:
        list[Callback]: Configured callbacks.
    """
    checkpoint_path = str(Path(checkpoint_dir) / "model.{epoch:02d}.h5")
    report_path = str(Path(report_dir) / "model.{epoch:02d}.html")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=config.model.patience,
        ),
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path),
        ReportCheckpoint(
            train_dir=train_dir,
            checkpoint_path=checkpoint_path,
            report_path=report_path,
            string_lookup_path=string_lookup_path,
            template_path=report_template_path,
            config=config.evaluate,
        ),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir),
    ]

    return callbacks


def train_model(
    model: keras.Model,
    training_data: TextLineDataset,
    validation_data: TextLineDataset,
    callbacks: list[Callback],
    config: ModelConfig,
) -> None:
    """Train model on training data.

    Args:
        model (keras.Model): Model to train.
        training_data (TextLineDataset): Data to train model.
        validation_data (TextLineDataset): Data to determine early stopping.
        callbacks (list[Callback]): Callbacks to add to model training loop.
        config (ModelConfig): Config with training parameters.
    """
    model.fit(
        training_data,
        validation_data=validation_data,
        epochs=config.epochs,
        callbacks=callbacks,
    )
