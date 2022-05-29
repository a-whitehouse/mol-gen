from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.data import TextLineDataset
from tensorflow.keras.layers import LSTM, Dense, Embedding, Input
from tensorflow.python.framework.ops import EagerTensor

from mol_gen.config.training.model import ModelConfig
from mol_gen.training.dataset import get_selfies_string_lookup_layer


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


class MoleculeGenerator:
    def __init__(self, model, string_to_integer_layer, integer_to_string_layer):
        self.model = model

        self.string_to_integer_layer = string_to_integer_layer
        self.integer_to_string_layer = integer_to_string_layer

        self.prediction_mask = self._get_prediction_mask(["UNK"])
        self.prediction_mask_initial = self._get_prediction_mask(
            [string_to_integer_layer.mask_token]
        )

    @classmethod
    def from_files(
        cls, model_filepath: str, token_counts_filepath: str
    ) -> MoleculeGenerator:
        """Load molecule generator from files.

        Args:
            model_filepath (str): Path to model checkpoint.
            token_counts_filepath (str): Path to token counts.

        Returns:
            MoleculeGenerator: Molecule generator.
        """
        model = tf.keras.models.load_model(model_filepath)

        vocabulary = pd.read_csv(token_counts_filepath)["token"].to_list()
        string_to_integer_layer = get_selfies_string_lookup_layer(vocabulary)
        integer_to_string_layer = get_selfies_string_lookup_layer(
            vocabulary, invert=True
        )

        return cls(model, string_to_integer_layer, integer_to_string_layer)

    def _get_prediction_mask(self, mask_tokens: list[str]) -> EagerTensor:
        """Get prediction mask for logits.

        Args:
            mask_tokens (list[str]): Tokens to mask.

        Returns:
            EagerTensor: Prediction Mask.
        """
        sparse_mask = tf.SparseTensor(
            values=[-float("inf")] * len(mask_tokens),
            indices=self.string_to_integer_layer(mask_tokens)[:, None],
            dense_shape=[self.string_to_integer_layer.vocabulary_size()],
        )
        return tf.sparse.to_dense(sparse_mask)

    def generate_molecules(self, n_molecules: int, temperature: float = 1) -> list[str]:
        """Generate molecules encoded as SELFIES.

        Args:
            n_molecules (int): Number of molecules to generate.
            temperature (float, optional): Scaling factor for logits. Defaults to 1.

        Returns:
            list[str]: Generated molecules.
        """
        # Model requires mask token at start of molecules
        mols = np.zeros(n_molecules).reshape(-1, 1)

        # Use separate arrays for complete and incomplete molecules
        complete_mols = np.empty((0, 0))

        while len(mols):
            # Take predicted logits for final characters in sequences
            pred_logits = self.model.predict(mols)[:, -1, :]

            # Don't allow unknown token to be selected
            pred_logits = (pred_logits / temperature) + self.prediction_mask

            # Don't allow mask token to be applied immediately after starting token
            if mols.shape[1] == 1:
                pred_logits += self.prediction_mask_initial

            # Make weighted random selection of tokens using predicted logits
            pred_tokens = tf.random.categorical(pred_logits, 1)

            # Append selected tokens to the end of molecules
            mols = np.hstack((mols, pred_tokens))

            # Remove molecules that have a mask token as the final predicted token
            new_mols = mols[mols[:, -1] == 0]
            mols = mols[mols[:, -1] != 0]

            # Pad completed molecules to the same shape so all can be stored in array
            complete_mols = np.pad(
                complete_mols, ((0, 0), (0, new_mols.shape[1] - complete_mols.shape[1]))
            )

            # Only add to completed molecules if new completed molecules present
            if len(new_mols):
                complete_mols = np.append(complete_mols, new_mols, axis=0)

        # Get corresponding SELFIES tokens from integers
        mols = self.integer_to_string_layer(complete_mols)

        # Concatenate SELFIES token elements for each molecule
        mols = tf.strings.reduce_join(mols, axis=1)

        # Decode to string
        return [i.decode() for i in mols.numpy()[1:]]
