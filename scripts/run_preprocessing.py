import argparse
from pathlib import Path
from shutil import rmtree

from mol_gen.preprocessing.dask import (
    convert_and_filter_smiles_strings,
    create_selfies_from_smiles,
    drop_duplicates_and_repartition,
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

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    output_dir = Path(args.output)
    intermediate_dir = output_dir / "intermediate"
    preprocessed_smiles_dir = output_dir / "smiles"
    selfies_dir = output_dir / "selfies"

    if not preprocessed_smiles_dir.exists():
        print("Starting preprocessing of SMILES strings")
        convert_and_filter_smiles_strings(args.input, intermediate_dir, args.config)
        drop_duplicates_and_repartition(intermediate_dir, preprocessed_smiles_dir)
        rmtree(intermediate_dir)

    else:
        print("Skipping preprocessing of SMILES strings")
        print("SMILES directory already exists in output directory")

    if not selfies_dir.exists():
        print("Starting encoding of SMILES strings as SELFIES")
        create_selfies_from_smiles(preprocessed_smiles_dir, selfies_dir)


if __name__ == "__main__":
    main()
