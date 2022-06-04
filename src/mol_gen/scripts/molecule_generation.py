from pathlib import Path

import click

from mol_gen.molecule_generation.molecule_generator import MoleculeGenerator


@click.command("generate")
@click.option("--model", type=click.STRING, help="Path to trained model.")
@click.option("--vocab", type=click.STRING, help="Path to string lookup config.")
@click.option(
    "--output",
    type=click.STRING,
    help="Path to file to write generated molecules.",
)
@click.option(
    "--n-mols",
    type=click.IntRange(min=1),
    default=1000,
    help="Number of molecules to generate.",
)
def run_molecule_generation(
    model_path: str,
    vocab_path: str,
    output_path: str,
    n_mols: int,
) -> None:
    """Generate molecules with trained model.

    Args:
        model_path (str): Path to trained model.
        vocab_path (str): Path to string lookup config.
        output_dir (str): Path to file to write generated molecules.
        n_mols (int): Number of molecules to generate.
    """
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)

    mol_generator = MoleculeGenerator.from_files(model_path, vocab_path)
    mols = mol_generator.generate_molecules(n_mols)

    with open(output_path, "w") as fh:
        for mol in mols:
            fh.write(f"{mol}\n")
