from dataclasses import dataclass

from rdkit.Chem import Mol, MolFromSmiles, MolToSmiles

from mol_gen.config.preprocessing import PreprocessingConfig
from mol_gen.exceptions import ConvertException, FilterException


@dataclass
class MoleculePreprocessor:
    config: PreprocessingConfig

    def process_molecule(self, smiles: str) -> str:
        """Apply conversion and filter methods to molecule.

        Args:
            mol (Mol): SMILES string of molecule to preprocess.

        Raises:
            UndesirableMolecule: If molecule fails a filter method.

        Returns:
            str: SMILES string of preprocessed molecule.
        """
        mol = MolFromSmiles(smiles)

        try:
            mol = self.apply_conversions(mol)
        except ConvertException:
            return

        try:
            self.apply_filters(mol)
        except FilterException:
            return

        smiles = MolToSmiles(mol)
        return smiles

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
