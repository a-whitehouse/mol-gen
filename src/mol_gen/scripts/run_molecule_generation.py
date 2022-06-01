import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate molecules with trained model."
    )
    parser.add_argument("--model", type=str, help="Path to trained model.")
    parser.add_argument("--vocab", type=str, help="Path to string lookup config.")
    parser.add_argument("--n-mols", type=int, help="Number of molecules to generate.")
    parser.add_argument(
        "--output",
        type=str,
        help="Path to file to write generated molecules.",
    )

    args = parser.parse_args()

    return args


def run_molecule_generation(
    model_path: Path,
    vocab_path: Path,
    output_path: Path,
    n_mols: int,
) -> None:
    """Generate molecules with trained model.

    Args:
        model_path (Path): Path to trained model.
        vocab_path (Path): Path to string lookup config.
        output_dir (Path): Path to file to write generated molecules.
        n_mols (int): Number of molecules to generate.
    """
    pass


def main():
    """Generate molecules with trained model."""
    args = parse_args()

    model_path = Path(args.model)
    vocab_path = Path(args.vocab)
    output_path = Path(args.output)

    run_molecule_generation(model_path, vocab_path, output_path, args.n_mols)


if __name__ == "__main__":
    main()
