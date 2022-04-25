import argparse
from pathlib import Path
from shutil import rmtree
from typing import Union

import dask.dataframe as dd
from dask.distributed import Client

from mol_gen.preprocessing.convert import encode_smiles_as_selfies
from mol_gen.preprocessing.preprocessor import preprocess_smiles_dataframe


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


def convert_and_filter_smiles_strings(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    config_path: Union[str, Path],
) -> None:
    print("Setting up dask client")
    with Client() as client:
        print(client.dashboard_link)
        print("Executing preprocessing steps on input molecules")

        df = dd.read_parquet(input_path)
        df.repartition(partition_size="25MB").map_partitions(
            preprocess_smiles_dataframe, config_path, meta=df
        ).to_parquet(output_path)


def drop_duplicates_and_repartition(input_path: str, output_path: str) -> None:
    print("Removing duplicate molecules and repartitioning files")
    df = dd.read_parquet(input_path)

    df.drop_duplicates(subset="SMILES", split_out=df.npartitions).repartition(
        partition_size="100MB"
    ).to_parquet(output_path)


def create_selfies_from_smiles(input_path: str, output_path: str):
    print("Setting up dask client")
    with Client() as client:
        print(client.dashboard_link)
        print("Creating SELFIES from SMILES strings")

        df = dd.read_parquet(input_path)

        df["SELFIES"] = df["SMILES"].apply(encode_smiles_as_selfies, meta=(None, str))
        df[["SELFIES"]].dropna().to_parquet(output_path)


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
