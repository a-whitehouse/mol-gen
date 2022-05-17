import argparse
from pathlib import Path

import pandas as pd
from tensorflow.data import TextLineDataset

from mol_gen.config.training import TrainingConfig
from mol_gen.training.dataset import (
    get_selfies_string_lookup_layer,
    process_selfies_dataset,
)
from mol_gen.training.model import train_selfies_model


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
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
    args = parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    selfies_train_filepath = input_dir.joinpath("selfies", "train", "*.part")
    token_counts_filepath = input_dir.joinpath("selfies", "token_counts.csv")

    config = TrainingConfig.from_file(args.config)

    vocabulary = pd.read_csv(token_counts_filepath)["token"].to_list()
    string_to_integer_layer = get_selfies_string_lookup_layer(vocabulary)

    dataset = TextLineDataset(TextLineDataset.list_files(str(selfies_train_filepath)))
    dataset = process_selfies_dataset(
        dataset,
        config.dataset.buffer_size,
        config.dataset.batch_size,
        string_to_integer_layer,
    )

    vocab_size = string_to_integer_layer.vocabulary_size()
    train_selfies_model(output_dir, dataset, vocab_size)


if __name__ == "__main__":
    main()
