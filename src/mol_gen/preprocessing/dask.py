from functools import wraps
from pathlib import Path

import dask.dataframe as dd
import pandas as pd
from dask.distributed import Client
from selfies import split_selfies

from mol_gen.config.preprocessing import PreprocessingConfig
from mol_gen.config.preprocessing.split import SplitConfig
from mol_gen.preprocessing.preprocessor import MoleculePreprocessor
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


def apply_molecule_preprocessor_to_parquet(
    input_dir: Path, output_dir: Path, config_path: Path, column: str
) -> None:
    """Apply preprocessing methods and filters to molecules in dataframe.

    Args:
        input_dir (Path): Path to directory to read data as parquet.
        output_dir (Path): Path to directory to write data as parquet.
        config_path (Path): Path to config.
        column (str): Name of column containing SMILES strings.
    """
    df = dd.read_parquet(input_dir)

    df.repartition(partition_size="25MB").map_partitions(
        apply_molecule_preprocessor_to_partition,
        config_path,
        column,
        meta={"SMILES": str},
    ).to_parquet(output_dir)


def apply_molecule_preprocessor_to_partition(
    df: pd.DataFrame, config_path: str, column: str
) -> pd.DataFrame:
    """Apply preprocessing methods and filters to molecules in dataframe.

    Molecules must be present as SMILES strings.

    Args:
        df (pd.DataFrame): SMILES string of molecules to preprocess.
        config_path (Path): Path to config.
        column (str): Name of column containing SMILES strings.

    Returns:
        pd.DataFrame: Filtered dataframe with converted molecules in same column.
    """
    config = PreprocessingConfig.from_file(config_path)
    preprocessor = MoleculePreprocessor(config)

    return preprocessor.process_molecules(df[column]).rename("SMILES").to_frame()


def drop_duplicates_and_repartition_parquet(
    input_dir: Path, output_dir: Path, column: str
) -> None:
    """Drops rows from dataframe with repeated values in given column and repartitions.

    Args:
        input_dir (Path): Path to directory to read data as parquet.
        output_dir (Path): Path to directory to write data as parquet.
        column (str): Name of column to use for dropping duplicate rows.
    """
    df = dd.read_parquet(input_dir)

    df.drop_duplicates(subset=column, split_out=df.npartitions).repartition(
        partition_size="100MB"
    ).to_parquet(output_dir)


def create_selfies_from_smiles(input_dir: Path, output_dir: Path, column: str) -> None:
    """Encodes SMILES strings as SELFIES.

    Args:
        input_dir (Path): Path to directory to read data as parquet.
        output_dir (Path): Path to directory to write data as parquet.
        column (str): Name of column containing SMILES strings.
    """
    df = dd.read_parquet(input_dir)

    df["SELFIES"] = df[column].apply(encode_smiles_as_selfies, meta=(None, str))
    df[["SELFIES"]].dropna().to_parquet(output_dir)


def get_selfies_token_counts_from_parquet(
    input_dir: Path, output_dir: Path, column: str
) -> None:
    """Gets counts of SELFIES tokens from strings in dataframe column.

    Args:
        input_dir (Path): Path to directory to read data as parquet.
        output_dir (Path): Path to directory to write token counts as csv.
        column (str): Name of column containing SELFIES.
    """
    df = dd.read_parquet(input_dir)

    counts = (
        df.repartition(partition_size="25MB")
        .map_partitions(
            get_selfies_tokens_from_partition, column, meta=("SELFIES", str)
        )
        .value_counts()
        .rename("count")
        .compute()
    )

    counts.to_csv(output_dir / "token_counts.csv", index_label="token")


def get_selfies_tokens_from_partition(df: pd.DataFrame, column: str) -> pd.Series:
    """Gets individual SELFIES tokens from strings in dataframe column.

    Args:
        df (pd.DataFrame): SMILES string of molecules to preprocess.
        column (str): Name of column containing SELFIES.

    Returns:
        pd.Series: SELFIES tokens.
    """
    return df[column].apply(split_selfies).apply(list).explode(ignore_index=True)


def create_splits_from_parquet(
    input_dir: Path, output_dir: Path, config: SplitConfig
) -> None:
    """Splits dataframe by row to separate train/validate/test sets.

    Created sets are written in the corresponding subdirectory as text files.

    Args:
        input_dir (Path): Path to directory to read data as parquet.
        output_dir (Path): Path to directory to write split data.
        config (SplitConfig): Config with validate and test set sizes.
    """
    df = dd.read_parquet(input_dir)
    columns = df.columns

    df["split"] = df.apply(lambda _: config.assign(), axis=1, meta=(None, str))

    for set_name in ("train", "validate", "test"):
        df.loc[df["split"] == set_name, columns].to_csv(
            output_dir.joinpath(set_name), index=False, header=False
        )
