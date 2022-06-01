import click

from mol_gen.scripts.run_molecule_generation import run_molecule_generation
from mol_gen.scripts.run_preprocessing import run_preprocessing
from mol_gen.scripts.run_training import run_training


@click.group()
def cli():
    """Molecule generation project."""
    pass


cli.add_command(run_molecule_generation)
cli.add_command(run_training)
cli.add_command(run_preprocessing)
