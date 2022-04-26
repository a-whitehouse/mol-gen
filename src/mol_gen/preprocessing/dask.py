from functools import wraps
from pathlib import Path
from typing import Union

import dask.dataframe as dd
from dask.distributed import Client

from mol_gen.preprocessing.preprocessor import preprocess_smiles_dataframe
from mol_gen.preprocessing.selfies import encode_smiles_as_selfies


def run_with_distributed_client(func):
    @wraps(func)
    def wrapped_func(*args, **kwargs):
        print("Setting up dask client")
        with Client() as client:
            print(client.dashboard_link)
            func(*args, **kwargs)
            print("Closing dask client")

    return wrapped_func


@run_with_distributed_client
def convert_and_filter_smiles_strings(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    config_path: Union[str, Path],
) -> None:
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


@run_with_distributed_client
def create_selfies_from_smiles(input_path: str, output_path: str):
    print("Creating SELFIES from SMILES strings")

    df = dd.read_parquet(input_path)

    df["SELFIES"] = df["SMILES"].apply(encode_smiles_as_selfies, meta=(None, str))
    df[["SELFIES"]].dropna().to_parquet(output_path)
