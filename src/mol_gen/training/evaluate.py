from pathlib import Path

import papermill as pm


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
    pm.execute_notebook(
        template_path,
        output_path,
        parameters={
            "train_dir": train_dir,
            "checkpoint_dir": checkpoint_dir,
            "string_lookup_config_filepath": string_lookup_config_filepath,
        },
    )
