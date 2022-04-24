import argparse
import logging
from pathlib import Path
from shutil import rmtree

import dask.dataframe as dd
from dask.distributed import Client

from mol_gen.preprocessing.preprocessor import process_dataframe

logger = logging.getLogger("preprocessing")
logger.setLevel = logging.INFO
handler = logging.FileHandler(filename="example.log")
logger.addHandler(handler)


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
    "--output", type=str, help="Path to directory to write preprocessed SMILES strings."
)

args = parser.parse_args()


def main():

    print("Setting up dask client")

    with Client():
        df = dd.read_parquet(args.input)

        # Persist preprocessed results in intermediate directory
        intermediate_dir = Path(args.output).joinpath("intermediate")

        print("Executing preprocessing steps on input molecules")
        df.repartition(partition_size="25MB").map_partitions(
            process_dataframe, args.config, meta=df
        ).to_parquet(intermediate_dir)

    # Drop duplicates and repartition intermediate results
    # Persist results in output directory and wipe intermediate directory
    print("Removing duplicate molecules and repartitioning files")
    df = dd.read_parquet(intermediate_dir)

    df.drop_duplicates(subset="SMILES", split_out=df.npartitions).repartition(
        partition_size="100MB"
    ).to_parquet(args.output)

    rmtree(intermediate_dir)


if __name__ == "__main__":
    main()
