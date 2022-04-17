from dataclasses import dataclass

from rdkit.Chem import Mol, MolFromSmiles, MolToSmiles

from mol_gen.config.preprocessing import PreprocessingConfig
from mol_gen.exceptions import ConvertException, FilterException
from mol_gen.preprocessing.filter import check_only_allowed_elements_present


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
            mol = self.convert_molecule(mol)
        except ConvertException:
            return

        try:
            self.apply_filters(mol)
        except FilterException:
            return

        smiles = MolToSmiles(mol)
        return smiles

    def convert_molecule(self, mol: Mol) -> Mol:
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
        self.apply_allowed_elements_filter(mol)
        self.apply_range_filters(mol)

    def apply_allowed_elements_filter(self, mol: Mol) -> None:
        """Check whether molecular descriptors are within allowed values.

        Args:
            mol (Mol): Molecule to test.

        Raises:
            UndesirableMolecule: If molecule fails the filter method.
        """
        allowed_elements = self.config.filter.allowed_elements
        check_only_allowed_elements_present(mol, allowed_elements)

    def apply_range_filters(self, mol: Mol) -> None:
        """Check whether molecular descriptors are within allowed values.

        Args:
            mol (Mol): Molecule to test.

        Raises:
            UndesirableMolecule: If molecule fails a filter method.
        """
        for filter in self.config.filter.range_filters:
            filter_method = filter.get_method()
            filter_method(mol)
