import pandas as pd
from attrs import frozen
from rdkit.Chem import Mol, MolFromSmiles, MolToSmiles

from mol_gen.config.preprocessing import PreprocessingConfig
from mol_gen.exceptions import PreprocessingException


def process_dataframe(df: pd.DataFrame, config_path: str) -> pd.Series:
    """Apply preprocessing methods and filters to molecules in dataframe.

    Molecules must be present as SMILES strings in column "SMILES".

    Args:
        df (pd.DataFrame): Dataframe containing SMILES strings in column "SMILES".
        config_path (Path): Path to config.
    """
    config = PreprocessingConfig.from_file(config_path)
    preprocessor = MoleculePreprocessor(config)

    df["SMILES"] = df["SMILES"].apply(preprocessor.process_molecule)
    df.dropna(subset=["SMILES"], inplace=True)

    return df


@frozen
class MoleculePreprocessor:
    config: PreprocessingConfig

    def process_molecule(self, smiles: str) -> str:
        """Apply conversion and filter methods to molecule.

        If a preprocessing step fails, nothing is returned.

        Args:
            mol (Mol): SMILES string of molecule to preprocess.

        Returns:
            str: SMILES string of preprocessed molecule.
        """
        try:
            mol = self.parse_smiles(smiles)
            mol = self.apply_conversions(mol)
            self.apply_filters(mol)

        except PreprocessingException:
            return

        return MolToSmiles(mol)

    def parse_smiles(self, smiles: str) -> Mol:
        """Parses SMILES string to RDKit molecule.

        Args:
            smiles (str): SMILES string

        Returns:
            Mol: Molecule.
        """
        mol = MolFromSmiles(smiles)

        if mol is None:
            raise PreprocessingException(f"SMILES string {smiles} could not be parsed.")

        return mol

    def apply_conversions(self, mol: Mol) -> Mol:
        """Apply conversion methods to molecule.

        Args:
            mol (Mol): Molecule to convert.

        Returns:
            str: Converted molecule.
        """
        return self.config.convert.apply(mol)

    def apply_filters(self, mol: Mol) -> None:
        """Check whether molecule passes filters defined in preprocessing config.

        Args:
            mol (Mol): Molecule to test.

        Raises:
            UndesirableMolecule: If molecule fails a filter method.
        """
        self.config.filter.apply(mol)
