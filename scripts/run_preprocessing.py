import argparse
from pathlib import Path

import dask.dataframe as dd
import yaml

from mol_gen.config.preprocessing import PreprocessingConfig
from mol_gen.preprocessing.preprocessor import MoleculePreprocessor

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

with open(args.config) as f:
    config_dict = yaml.safe_load(f)

config = PreprocessingConfig.parse_config(config_dict)
preprocessor = MoleculePreprocessor(config)

input_filepaths = Path(args.input).glob("*.csv")

for fp in input_filepaths:
    print(f"Preprocessing {fp}")
    all_smiles = dd.read_csv(fp)

    all_smiles["SMILES"] = all_smiles["SMILES"].apply(
        preprocessor.process_molecule, meta=("SMILES", "object")
    )
    all_smiles = all_smiles.dropna(subset="SMILES")

    output_filepath = Path(args.output).joinpath(fp.name)
    all_smiles["SMILES"].to_csv(output_filepath, single_file=True)
    out = all_smiles.compute()
