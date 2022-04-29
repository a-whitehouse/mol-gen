import pandas as pd
import pytest
import yaml
from pandas.testing import assert_frame_equal, assert_index_equal

from mol_gen.preprocessing.dask import (
    apply_molecule_preprocessor_to_parquet,
    apply_molecule_preprocessor_to_partition,
    drop_duplicates_and_repartition_parquet,
)


@pytest.fixture
def smiles():
    return pd.DataFrame(
        [
            "CCOC(=O)C(C)(C)C1=CC(=C(C=C1)I)C",
            "CC.CC1([C@@H]2[C@@H]3CC[C@H](C3)[C@@H]2C4=C([N+]1(C)C)C=CC5=C4C(=O)NC5)C6=CC(=C(C=C6)N)C(=N)N",
            "CC1=CN(C(=O)NC1=O)[C@H]2[C@H]3[C@@H]([C@@](O2)(CN3C4=NOC(=N4)C)COC(C)(C)C)OC(C)(C)C",
            "CC(C)CN=C1NC(C2=C(N1)N(C=N2)[C@H]3[C@H](C([C@H](O3)CO)OC(=O)C)OC(=O)C)O",
            "CC1=NC=C(C=C1)C(C)(C)N2CCC(C2)(CCC3=CC=C(S3)F)C4=NC5=C(N4)C=C(C=C5)F",
            "CCCC#CC1=C(C2=CC=CC=C2[N+](=C1)[O-])CCNC(=O)OC(C)(C)C",
            "CCOC(=O)C1=CC(=O)NC2=C1C=CC(=C2)F",
            "CCCN(CCC)C=O.CC1=CC2=C(C=C(C=C2)C(=O)NC3=CN=CC(=C3)CNC(=O)CC(C4=CC=CC=C4)N)N=C(C1)N",
            "CC#CC(=O)NC1=[C-]C2=C(C=C1)N=CN=C2NC3=CC=C(C=C3)OC4CCCCCCC4.[Y]",
            "CCN1CC(OC1=O)C2(CCN(C2)C(C)(C)C3=CN=C(C=C3)C)CCC4=CC=C(S4)F",
        ],
        columns=["SMILES"],
        index=[0, 3, 4, 5, 6, 7, 8, 10, 11, 12],
    )


@pytest.fixture
def valid_config_section():
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
def config_path(tmpdir, valid_config_section):
    config_path = tmpdir.join("preprocessing.yml")

    with open(config_path, "w") as f:
        yaml.dump(valid_config_section, f)

    return config_path


class TestApplyMoleculePreprocessorToParquet:
    @pytest.fixture
    def input_dir(self, tmpdir, smiles):
        input_dir = tmpdir.join("smiles")

        smiles.to_parquet(input_dir)

        return input_dir

    @pytest.fixture
    def output_dir(self, tmpdir):
        return tmpdir.join("output")

    def test_completes_given_valid_smiles(self, input_dir, output_dir, config_path):
        apply_molecule_preprocessor_to_parquet(
            input_dir, output_dir, config_path, "SMILES"
        )

    def test_writes_dataframe_with_smiles_column(
        self, input_dir, output_dir, config_path
    ):
        apply_molecule_preprocessor_to_parquet(
            input_dir, output_dir, config_path, "SMILES"
        )

        actual = pd.read_parquet(output_dir)

        assert isinstance(actual, pd.DataFrame)
        assert_index_equal(actual.columns, pd.Index(["SMILES"]))

    def test_raises_exception_given_incorrect_column_name(
        self, input_dir, output_dir, config_path
    ):
        with pytest.raises(KeyError):
            apply_molecule_preprocessor_to_parquet(
                input_dir, output_dir, config_path, "smiles"
            )


class TestApplyMoleculePreprocessorToPartition:
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
    def config_path(self, tmpdir, valid_config_section):
        config_path = tmpdir.join("preprocessing.yml")

        with open(config_path, "w") as f:
            yaml.dump(valid_config_section, f)

        return config_path

    def test_completes_given_valid_smiles(self, smiles, config_path):
        apply_molecule_preprocessor_to_partition(smiles, config_path, "SMILES")

    def test_returns_dataframe_with_smiles_column(self, smiles, config_path):
        actual = apply_molecule_preprocessor_to_partition(smiles, config_path, "SMILES")

        assert isinstance(actual, pd.DataFrame)
        assert_index_equal(actual.columns, pd.Index(["SMILES"]))

    def test_returns_dataframe_with_no_missing_values(self, smiles, config_path):
        smiles[0] = "invalid smiles"

        actual = apply_molecule_preprocessor_to_partition(smiles, config_path, "SMILES")

        assert all(actual.notna())

    def test_removes_invalid_smiles_strings(self, smiles, config_path):
        smiles[0] = "invalid smiles"

        actual = apply_molecule_preprocessor_to_partition(smiles, config_path, "SMILES")

        assert not any(actual["SMILES"].str.contains("invalid smiles"))

    def test_raises_exception_given_incorrect_column_name(self, smiles, config_path):
        with pytest.raises(KeyError):
            apply_molecule_preprocessor_to_partition(smiles, config_path, "smiles")

    def test_renames_column_to_smiles(self, smiles, config_path):
        smiles = smiles.rename(columns={"SMILES": "smiles"})

        actual = apply_molecule_preprocessor_to_partition(smiles, config_path, "smiles")

        assert_index_equal(actual.columns, pd.Index(["SMILES"]))


class TestDropDuplicatesAndRepartitionParquet:
    @pytest.fixture
    def duplicate_smiles(self):
        return pd.DataFrame(
            [
                "CCOC(=O)C(C)(C)C1=CC(=C(C=C1)I)C",
                "CCOC(=O)C(C)(C)C1=CC(=C(C=C1)I)C",
                "CCOC(=O)C(C)(C)C1=CC(=C(C=C1)I)C",
                "CC.CC1([C@@H]2[C@@H]3CC[C@H](C3)[C@@H]2C4=C([N+]1(C)C)C=CC5=C4C(=O)NC5)C6=CC(=C(C=C6)N)C(=N)N",
                "CC1=CN(C(=O)NC1=O)[C@H]2[C@H]3[C@@H]([C@@](O2)(CN3C4=NOC(=N4)C)COC(C)(C)C)OC(C)(C)C",
                "CC(C)CN=C1NC(C2=C(N1)N(C=N2)[C@H]3[C@H](C([C@H](O3)CO)OC(=O)C)OC(=O)C)O",
                "CC1=NC=C(C=C1)C(C)(C)N2CCC(C2)(CCC3=CC=C(S3)F)C4=NC5=C(N4)C=C(C=C5)F",
                "CCCC#CC1=C(C2=CC=CC=C2[N+](=C1)[O-])CCNC(=O)OC(C)(C)C",
                "CCOC(=O)C1=CC(=O)NC2=C1C=CC(=C2)F",
                "CCOC(=O)C1=CC(=O)NC2=C1C=CC(=C2)F",
                "CCCN(CCC)C=O.CC1=CC2=C(C=C(C=C2)C(=O)NC3=CN=CC(=C3)CNC(=O)CC(C4=CC=CC=C4)N)N=C(C1)N",
                "CC#CC(=O)NC1=[C-]C2=C(C=C1)N=CN=C2NC3=CC=C(C=C3)OC4CCCCCCC4.[Y]",
                "CCN1CC(OC1=O)C2(CCN(C2)C(C)(C)C3=CN=C(C=C3)C)CCC4=CC=C(S4)F",
            ],
            columns=["SMILES"],
        )

    @pytest.fixture
    def input_dir(self, tmpdir, duplicate_smiles):
        input_dir = tmpdir.join("input")
        duplicate_smiles.to_parquet(input_dir)
        return input_dir

    @pytest.fixture
    def output_dir(self, tmpdir):
        return tmpdir.join("output")

    def test_completes(self, input_dir, output_dir):
        drop_duplicates_and_repartition_parquet(input_dir, output_dir, column="SMILES")

    def test_raises_exception_given_incorrect_column_name(self, input_dir, output_dir):
        with pytest.raises(KeyError):
            drop_duplicates_and_repartition_parquet(
                input_dir, output_dir, column="smiles"
            )

    def test_raises_exception_given_missing_parquet(self, tmpdir, output_dir):
        with pytest.raises(Exception):
            drop_duplicates_and_repartition_parquet(tmpdir, output_dir, column="SMILES")

    def test_writes_expected_parquet(self, input_dir, output_dir, smiles):
        drop_duplicates_and_repartition_parquet(input_dir, output_dir, column="SMILES")

        output_df = pd.read_parquet(output_dir)
        assert_frame_equal(output_df, smiles, check_names=False)
