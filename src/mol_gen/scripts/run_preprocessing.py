import argparse
from pathlib import Path
from shutil import rmtree

from mol_gen.preprocessing.dask import (
    apply_molecule_preprocessor_to_parquet,
    create_selfies_from_smiles,
    drop_duplicates_and_repartition_parquet,
    get_selfies_token_counts_from_parquet,
    run_with_distributed_client,
    write_parquet_as_text,
)


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Apply preprocessing steps to SMILES strings."
    )
    parser.add_argument(
        "--config", type=str, help="Path to preprocessing yaml config file."
    )
    parser.add_argument(
        "--input", type=str, help="Path to directory containing SMILES strings."
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to directory to write preprocessed SMILES strings.",
    )
    parser.add_argument(
        "--column",
        type=str,
        help="Name of column containing SMILES strings.",
        default="SMILES",
    )

    args = parser.parse_args()

    return args


def run_smiles_preprocessing(
    input_dir: Path,
    intermediate_dir: Path,
    output_dir: Path,
    config_path: Path,
    column: str,
) -> None:
    """Run preprocessing methods and filters to input SMILES strings.

    Args:
        input_dir (Path): Path to directory to read data as parquet.
        intermediate_dir (Path): Path to directory to write temporary files.
        output_dir (Path): Path to directory to write data as parquet.
        config_path (Path): Path to config.
        column (str): Name of column containing SMILES strings.
    """
    print("Starting preprocessing of SMILES strings")
    run_with_distributed_client(apply_molecule_preprocessor_to_parquet)(
        input_dir, intermediate_dir, config_path, column
    )

    print("Removing duplicate molecules and repartitioning files")
    drop_duplicates_and_repartition_parquet(intermediate_dir, output_dir, "SMILES")

    print("Removing intermediate files")
    rmtree(intermediate_dir)


@run_with_distributed_client
def run_selfies_preprocessing(
    input_dir: Path,
    intermediate_dir: Path,
    output_dir: Path,
    config_path: Path,
    column: str,
) -> None:
    """Encode SMILES strings as SELFIES, calculate token counts and split data.

    Args:
        input_dir (Path): Path to directory to read data as parquet.
        intermediate_dir (Path): Path to directory to write temporary files.
        output_dir (Path): Path to directory to write data as parquet.
        config_path (Path): Path to config.
        column (str): Name of column containing SMILES strings.
    """
    selfies_parquet_dir = output_dir / "parquet"
    selfies_text_dir = output_dir / "text"

    print("Starting encoding of SMILES strings as SELFIES")
    create_selfies_from_smiles(input_dir, selfies_parquet_dir, column)

    print("Counting SELFIES tokens")
    get_selfies_token_counts_from_parquet(selfies_parquet_dir, output_dir, "SELFIES")

    print("Writing SELFIES as text files")
    write_parquet_as_text(selfies_parquet_dir, selfies_text_dir, "SELFIES")


def main():
    args = parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    config_path = Path(args.config)

    intermediate_dir = output_dir / "intermediate"
    smiles_dir = output_dir / "smiles"
    selfies_dir = output_dir / "selfies"

    if not smiles_dir.exists():
        run_smiles_preprocessing(
            input_dir, intermediate_dir, smiles_dir, config_path, args.column
        )

    else:
        print(
            "Skipping preprocessing of SMILES strings\n"
            "SMILES directory already exists in output directory"
        )

    if not selfies_dir.exists():
        run_selfies_preprocessing(
            smiles_dir, intermediate_dir, selfies_dir, config_path, "SMILES"
        )

    else:
        print(
            "Skipping preprocessing of SELFIES\n"
            "SELFIES directory already exists in output directory"
        )


if __name__ == "__main__":
    main()
