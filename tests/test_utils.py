import pytest
import yaml
from rdkit.Chem import Mol, MolFromSmiles

from mol_gen.exceptions import ConfigException
from mol_gen.utils import check_smiles_equivalent_to_molecule, read_yaml_config_file


@pytest.fixture
def mol(smiles: str) -> Mol:
    return MolFromSmiles(smiles)


class TestCheckSMILESEquivalentToMolecule:
    @pytest.mark.parametrize(
        "smiles, expected",
        [
            ("CC(C)C(C(=O)O)N", "CC(C(C(=O)O)N)C"),
            ("NC(CC1=CC=CC=C1)C(O)=O", "NC(C(O)=O)CC1=CC=CC=C1"),
        ],
    )
    def test_makes_expected_matches(self, mol, expected):
        check_smiles_equivalent_to_molecule(mol, expected)

    @pytest.mark.parametrize(
        "smiles, expected", [("CC(C)C(C(=O)O)N", "NC(CC1=CC=CC=C1)C(O)=O")]
    )
    def test_raises_exception_for_expected_mismatches(self, mol, expected):
        with pytest.raises(AssertionError):
            check_smiles_equivalent_to_molecule(mol, expected)


class TestReadYamlConfigFile:
    @pytest.fixture
    def valid_config_section(self):
        return {
            "convert": ["neutralise_salts", "remove_stereochemistry"],
            "filter": {
                "allowed_elements": ["H", "C", "N", "O", "F", "S", "Cl", "Br"],
                "range_filters": {
                    "molecular_weight": {"min": 180, "max": 480},
                },
            },
        }

    @pytest.fixture
    def valid_config_file(self, tmpdir, valid_config_section):
        fp = tmpdir.join("preprocessing.yml")

        with open(fp, "w") as fh:
            yaml.dump(valid_config_section, fh)

        return fp

    def test_from_file_completes_given_valid_file(
        self,
        valid_config_file,
    ):
        read_yaml_config_file(valid_config_file)

    def test_from_file_returns_expected_config_given_valid_file(
        self,
        valid_config_section,
        valid_config_file,
    ):
        config = read_yaml_config_file(valid_config_file)

        assert config == valid_config_section

    def test_from_file_raises_exception_given_file_not_found(self, tmpdir):
        fp = tmpdir.join("preprocessing.yml")

        with pytest.raises(ConfigException) as excinfo:
            read_yaml_config_file(fp)

        assert "does not exist" in str(excinfo.value)

    def test_from_file_raises_exception_given_file_not_valid_yaml(self, tmpdir):
        fp = tmpdir.join("preprocessing.yml")
        with open(fp, "w") as fh:
            fh.write("convert: [")

        with pytest.raises(ConfigException) as excinfo:
            read_yaml_config_file(fp)

        assert "does not contain valid yaml" in str(excinfo.value)
