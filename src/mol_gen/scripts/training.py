from pathlib import Path

import click
import pandas as pd
from pyprojroot import here

from mol_gen.config.training import TrainingConfig
from mol_gen.training.dataset import get_selfies_dataset
from mol_gen.training.evaluate import create_model_evaluation_report
from mol_gen.training.model import get_compiled_model, train_model
from mol_gen.training.string_lookup import (
    get_selfies_string_lookup_layer,
    write_string_lookup_to_json,
)


@click.command("train")
@click.option("--config", type=click.STRING, help="Path to training yaml config file.")
@click.option(
    "--input",
    type=click.STRING,
    help="""Path to directory created by preprocessing script,
        containing 'selfies' directory with SELFIES text files and token counts.""",
)
@click.option(
    "--output",
    type=click.STRING,
    help="Path to directory to write trained models.",
)
def run_training(config, input, output):
    """Train and evaluate model checkpoints on preprocessed SELFIES."""
    input_dir = Path(input)
    output_dir = Path(output)

    train_dir = input_dir / "selfies" / "train"
    validate_dir = input_dir / "selfies" / "validate"
    token_counts_filepath = input_dir / "selfies" / "token_counts.csv"

    string_lookup_filepath = output_dir / "string_lookup.json"
    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    html_report_filepath = output_dir / "model_evaluation_report.html"

    report_template_filepath = (
        here() / "notebooks" / "templates" / "model_evaluation_report.ipynb"
    )

    config = TrainingConfig.from_file(config)

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

    create_model_evaluation_report(
        report_template_filepath,
        html_report_filepath,
        checkpoint_dir,
        train_dir,
        string_lookup_filepath,
        config.evaluate,
    )
