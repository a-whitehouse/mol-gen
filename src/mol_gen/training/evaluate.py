from pathlib import Path
from tempfile import TemporaryDirectory

import dask.dataframe as dd
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import papermill as pm
from nbconvert import HTMLExporter
from rdkit.Chem import Mol, MolFromSmiles
from rdkit.Chem.Draw import MolsToGridImage
from selfies import decoder
from tensorflow.keras.callbacks import Callback

from mol_gen.config.training.evaluate import EvaluateConfig
from mol_gen.preprocessing.filter import DESCRIPTOR_TO_FUNCTION


class ReportCheckpoint(Callback):
    def __init__(
        self,
        train_dir: Path | str,
        checkpoint_path: Path | str,
        report_path: Path | str,
        string_lookup_path: Path | str,
        template_path: Path | str,
        config: EvaluateConfig,
    ):
        self.train_dir = train_dir
        self.checkpoint_path = checkpoint_path
        self.report_path = report_path
        self.string_lookup_path = string_lookup_path
        self.template_path = template_path
        self.evaluate_config = config

        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        """Execute on end of epoch."""
        report_path = self.report_path.format(epoch)
        checkpoint_path = self.checkpoint_path.format(epoch)

        create_model_evaluation_report(
            train_dir=self.train_dir,
            checkpoint_path=checkpoint_path,
            report_path=report_path,
            template_path=self.template_path,
            string_lookup_path=self.string_lookup_path,
            config=self.evaluate_config,
        )


def create_model_evaluation_report(
    train_dir: Path | str,
    checkpoint_path: Path | str,
    report_path: Path | str,
    template_path: Path | str,
    string_lookup_path: Path | str,
    config: EvaluateConfig,
):
    """Create HTML report from notebook template for evaluating model checkpoint.

    Args:
        train_dir (Path | str): Path to training set directory to read SELFIES as text.
        checkpoint_path (Path | str): Path to trained model checkpoint.
        report_path (Path | str): Path to write output HTML report.
        template_path (Path | str): Path to model evaluation report notebook template.
        string_lookup_path (Path | str): Path to string lookup config.
        config (EvaluateConfig): Config with number of molecules to generate and draw.
    """
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory() as temp_dir:
        notebook_path = Path(temp_dir) / "notebook.ipynb"
        pm.execute_notebook(
            template_path,
            notebook_path,
            parameters={
                "train_dir": str(train_dir),
                "checkpoint_path": str(checkpoint_path),
                "string_lookup_config_path": str(string_lookup_path),
                "n_molecules": config.n_molecules,
                "subset_size": config.subset_size,
            },
        )

        write_notebook_as_html(notebook_path, report_path)


def write_notebook_as_html(input_path: Path | str, output_path: Path | str) -> None:
    """Write Jupyter notebook as HTML.

    Args:
        input_path (Path | str): Path to notebook.
        output_path (Path | str): Path to write HTML file.
    """
    html_exporter = HTMLExporter(template_name="classic")

    with open(input_path) as fh:
        nb = nbformat.read(fh, as_version=4)

    body, _ = html_exporter.from_notebook_node(nb)

    with open(output_path, "w") as fh:
        fh.write(body)


def calculate_percent_selfies_novel(selfies: list[str], train_dir: Path | str) -> float:
    """Calculate percent of SELFIES that are novel.

    The training directory is read into a dask dataframe,
    with each molecule in the SELFIES column compared against the input SELFIES.

    Args:
        selfies (list[str]): SELFIES to count.
        train_dir (Path | str): Path to training set directory to read SELFIES as text.

    Returns:
        float: Percentage to a precision of 1 decimal place.
    """
    training_data = dd.read_csv(
        Path(train_dir) / "*",
        names=["SELFIES"],
    )
    repeated_selfies = training_data.loc[
        training_data["SELFIES"].isin(selfies)
    ].compute()
    novel_selfies = [i for i in selfies if i not in repeated_selfies]
    return round(100 * (len(novel_selfies) / len(selfies)))


def calculate_percent_selfies_valid(selfies: list[str]) -> float:
    """Calculate percent of SELFIES that are valid.

    Count SELFIES that can be successfully decoded as SMILES strings.

    Args:
        selfies (list[str]): SELFIES to count.

    Returns:
        float: Percentage to a precision of 1 decimal place.
    """
    valid_selfies = get_valid_molecules_from_selfies(selfies)
    return round(100 * (len(valid_selfies) / len(selfies)))


def calculate_percent_unique(values: list[str]) -> float:
    """Calculate percent of values that are unique.

    Args:
        values (list[str]): Values to count.

    Returns:
        float: Percentage to a precision of 1 decimal place.
    """
    unique_values = set(values)
    return round(100 * (len(unique_values) / len(values)))


def draw_subset_selfies(selfies: list[str], subset_size: int):
    """Draw subset of SELFIES as molecule grid image.

    Args:
        selfies (list[str]): SELFIES to draw.
        subset_size (int, optional): Total molecules to draw. Defaults to 100.
    """
    mols = get_valid_molecules_from_selfies(selfies)
    subset_mols = np.random.choice(mols, subset_size, replace=False)
    return MolsToGridImage(subset_mols, subImgSize=(500, 500), molsPerRow=2, maxMols=50)


def plot_descriptor_distributions(selfies: list[str], bins: int = 10):
    """Plot distributions of descriptors for SELFIES as histograms.

    Args:
        selfies (list[str]): SELFIES to plot.
        bins (int, optional): Total bins in each histogram. Defaults to 10.
    """
    valid_mols = get_valid_molecules_from_selfies(selfies)
    for descriptor, func in DESCRIPTOR_TO_FUNCTION.items():
        values = [func(i) for i in valid_mols]
        plt.hist(values, bins=bins)
        plt.xlabel(descriptor)
        plt.ylabel("count")
        plt.show()


def get_valid_molecules_from_selfies(selfies: list[str]) -> list[Mol]:
    """Get RDKit molecules from SELFIES.

    Only successfully parsed SELFIES are returned.

    Args:
        selfies (list[str]): SELFIES to parse.

    Returns:
        list[Mol]: Molecules.
    """
    smiles = [decoder(i) for i in selfies]
    return [MolFromSmiles(i) for i in smiles if MolFromSmiles(i)]
