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
        convert_methods = self.config.convert.methods
        for method in convert_methods:
            try:
                mol = method(mol)
            except Exception as e:
                raise ConvertException(f"Convert method {method} failed: {e}")

        return mol

    def apply_filters(self, mol: Mol) -> None:
        """Check whether molecule passes filters defined in preprocessing config.

        Args:
            mol (Mol): Molecule to test.

        Raises:
            UndesirableMolecule: If molecule fails a filter method.
        """
        # Apply allowed elements filter
        method = self.config.filter.elements_filter.get_method()
        method(mol)

        # Apply range filters
        for filter in self.config.filter.range_filters.values():
            method = filter.get_method()
            method(mol)
