from pathlib import Path
from tempfile import TemporaryDirectory

import click

from mol_gen.config.preprocessing import PreprocessingConfig
from mol_gen.preprocessing.dask import (
    apply_molecule_preprocessor_to_parquet,
    create_selfies_from_smiles,
    create_splits_from_parquet,
    drop_duplicates_and_repartition_parquet,
    get_selfies_token_counts_from_parquet,
    run_with_distributed_client,
)


def run_smiles_preprocessing(
    input_dir: Path,
    output_dir: Path,
    config_path: Path,
    column: str,
) -> None:
    """Run preprocessing methods and filters on input SMILES strings.

    Args:
        input_dir (Path): Path to directory to read data as parquet.
        output_dir (Path): Path to directory to write data as parquet.
        config_path (Path): Path to config.
        column (str): Name of column containing SMILES strings.
    """
    with TemporaryDirectory() as temp_dir:
        print("Starting preprocessing of SMILES strings")
        run_with_distributed_client(apply_molecule_preprocessor_to_parquet)(
            input_dir, temp_dir, config_path, column
        )

        print("Removing duplicate molecules and repartitioning files")
        drop_duplicates_and_repartition_parquet(temp_dir, output_dir, "SMILES")


@run_with_distributed_client
def run_selfies_preprocessing(
    input_dir: Path,
    output_dir: Path,
    config_path: Path,
    column: str,
) -> None:
    """Encode SMILES strings as SELFIES, calculate token counts and split data.

    Args:
        input_dir (Path): Path to directory to read data as parquet.
        output_dir (Path): Path to directory to write data as parquet.
        config_path (Path): Path to config.
        column (str): Name of column containing SMILES strings.
    """
    output_dir.mkdir(exist_ok=True)
    config = PreprocessingConfig.from_file(config_path)

    with TemporaryDirectory() as temp_dir:
        print("Starting encoding of SMILES strings as SELFIES")
        create_selfies_from_smiles(input_dir, temp_dir, column)

        print("Counting SELFIES tokens")
        get_selfies_token_counts_from_parquet(temp_dir, output_dir, "SELFIES")

        print("Splitting SELFIES to separate train/validate/test sets")
        create_splits_from_parquet(temp_dir, output_dir, config.split)


@click.command("preprocess")
@click.option(
    "--config", type=click.STRING, help="Path to preprocessing yaml config file."
)
@click.option(
    "--input", type=click.STRING, help="Path to directory containing SMILES strings."
)
@click.option(
    "--output",
    type=click.STRING,
    help="Path to directory to write preprocessed SMILES strings.",
)
@click.option(
    "--column",
    type=click.STRING,
    help="Name of column containing SMILES strings.",
    default="SMILES",
)
def run_preprocessing(config, input, output, column):
    """Run preprocessing methods and filters on input SMILES strings."""
    input_dir = Path(input)
    output_dir = Path(output)
    config_path = Path(config)

    smiles_dir = output_dir / "smiles"
    selfies_dir = output_dir / "selfies"

    if not smiles_dir.exists():
        run_smiles_preprocessing(input_dir, smiles_dir, config_path, column)

    else:
        print(
            "Skipping preprocessing of SMILES strings\n"
            "SMILES directory already exists in output directory"
        )

    if not selfies_dir.exists():
        run_selfies_preprocessing(smiles_dir, selfies_dir, config_path, "SMILES")

    else:
        print(
            "Skipping preprocessing of SELFIES\n"
            "SELFIES directory already exists in output directory"
        )
