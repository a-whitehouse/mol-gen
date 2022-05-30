from pathlib import Path
from tempfile import TemporaryDirectory

import nbformat
import papermill as pm
from nbconvert import HTMLExporter


def create_model_evaluation_report(
    template_path: Path | str,
    output_path: Path | str,
    checkpoint_dir: Path | str,
    train_dir: Path | str,
    string_lookup_config_filepath: Path | str,
):
    """Create HTML report from notebook template for evaluating model checkpoints.

    Args:
        template_path (Path | str): Path to model evaluation report notebook template.
        output_path (Path | str): Path to write output HTML report.
        checkpoint_dir (Path | str): Path to directory to save trained models.
        train_dir (Path | str): Path to training set directory to read SELFIES as text.
        string_lookup_config_filepath (Path | str): Path to string lookup config.
    """
    with TemporaryDirectory() as temp_dir:
        notebook_path = Path(temp_dir) / "notebook.ipynb"
        pm.execute_notebook(
            template_path,
            notebook_path,
            parameters={
                "train_dir": str(train_dir),
                "checkpoint_dir": str(checkpoint_dir),
                "string_lookup_config_filepath": str(string_lookup_config_filepath),
            },
        )
        write_notebook_as_html(notebook_path, output_path)


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
