import argparse
from pathlib import Path
from shutil import rmtree

from mol_gen.preprocessing.dask import (
    apply_molecule_preprocessor_to_parquet,
    create_selfies_from_smiles,
    drop_duplicates_and_repartition_parquet,
    get_selfies_token_counts_from_parquet,
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


def main():
    args = parse_args()

    output_dir = Path(args.output)
    intermediate_dir = output_dir / "intermediate"
    preprocessed_smiles_dir = output_dir / "smiles"

    selfies_dir = output_dir / "selfies"
    selfies_parquet_dir = selfies_dir / "parquet"
    selfies_text_dir = selfies_dir / "text"

    if not preprocessed_smiles_dir.exists():
        print("Starting preprocessing of SMILES strings")
        apply_molecule_preprocessor_to_parquet(
            args.input, intermediate_dir, args.config, args.column
        )
        print("Removing duplicate molecules and repartitioning files")
        drop_duplicates_and_repartition_parquet(
            intermediate_dir, preprocessed_smiles_dir, column="SMILES"
        )
        print("Removing intermediate files")
        rmtree(intermediate_dir)

    else:
        print("Skipping preprocessing of SMILES strings")
        print("SMILES directory already exists in output directory")

    if not selfies_dir.exists():
        print("Starting encoding of SMILES strings as SELFIES")
        create_selfies_from_smiles(
            preprocessed_smiles_dir, selfies_parquet_dir, column="SMILES"
        )
        print("Counting SELFIES tokens")
        get_selfies_token_counts_from_parquet(
            selfies_parquet_dir, selfies_dir, column="SELFIES"
        )
        print("Writing SELFIES as text files")
        write_parquet_as_text(selfies_parquet_dir, selfies_text_dir, column="SELFIES")

    else:
        print("Skipping preprocessing of SELFIES")
        print("SELFIES directory already exists in output directory")


if __name__ == "__main__":
    main()
