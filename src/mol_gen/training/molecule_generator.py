from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor

from mol_gen.training.string_lookup import read_string_lookup_from_json


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
        cls, model_filepath: Path | str, string_lookup_config_filepath: Path | str
    ) -> MoleculeGenerator:
        """Load molecule generator from files.

        Args:
            model_filepath (Path | str): Path to model checkpoint.
            string_lookup_config_filepath (Path | str): Path to string lookup config.

        Returns:
            MoleculeGenerator: Molecule generator.
        """
        model = tf.keras.models.load_model(model_filepath)

        string_to_integer_layer = read_string_lookup_from_json(
            string_lookup_config_filepath
        )
        integer_to_string_layer = read_string_lookup_from_json(
            string_lookup_config_filepath, invert=True
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
