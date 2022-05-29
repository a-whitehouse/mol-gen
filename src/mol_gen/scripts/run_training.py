import argparse
from pathlib import Path

import pandas as pd

from mol_gen.config.training import TrainingConfig
from mol_gen.training.dataset import get_selfies_dataset
from mol_gen.training.model import get_compiled_model, train_model
from mol_gen.training.string_lookup import (
    get_selfies_string_lookup_layer,
    write_string_lookup_to_json,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Execute training on SELFIES.")
    parser.add_argument("--config", type=str, help="Path to training yaml config file.")
    parser.add_argument(
        "--input",
        type=str,
        help="""Path to directory created by preprocessing script,
        containing 'selfies' directory with SELFIES text files and token counts.""",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to directory to write trained models.",
    )

    args = parser.parse_args()

    return args


def main():
    """Run dataset preprocessing pipeline and model training on input SELFIES."""
    args = parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    train_dir = input_dir / "selfies" / "train"
    validate_dir = input_dir / "selfies" / "validate"
    token_counts_filepath = input_dir / "selfies" / "token_counts.csv"

    string_lookup_filepath = output_dir / "string_lookup.json"
    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"

    config = TrainingConfig.from_file(args.config)

    vocabulary = pd.read_csv(token_counts_filepath)["token"].to_list()
    string_to_integer_layer = get_selfies_string_lookup_layer(vocabulary)
    write_string_lookup_to_json(string_to_integer_layer, string_lookup_filepath)

    training_data = get_selfies_dataset(
        train_dir, config.dataset, string_to_integer_layer
    )
    validation_data = get_selfies_dataset(
        validate_dir, config.dataset, string_to_integer_layer
    )

    vocab_size = string_to_integer_layer.vocabulary_size()
    model = get_compiled_model(config.model, vocab_size)
    train_model(
        checkpoint_dir, log_dir, model, training_data, validation_data, config.model
    )


if __name__ == "__main__":
    main()
