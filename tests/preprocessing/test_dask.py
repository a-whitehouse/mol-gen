from pathlib import Path

import dask.dataframe as dd
import pandas as pd
import pytest
import yaml
from pandas.testing import assert_frame_equal, assert_index_equal, assert_series_equal
from pyprojroot import here

from mol_gen.config.preprocessing import PreprocessingConfig
from mol_gen.preprocessing.dask import (
    apply_molecule_preprocessor_to_parquet,
    apply_molecule_preprocessor_to_partition,
    create_selfies_from_smiles,
    create_splits_from_parquet,
    drop_duplicates_and_repartition_parquet,
    get_selfies_token_counts_from_parquet,
    get_selfies_tokens_from_partition,
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
        "split": {"validate": 0.1, "test": 0.2},
    }


@pytest.fixture
def config_path(tmpdir, valid_config_section):
    config_path = tmpdir.join("preprocessing.yml")

    with open(config_path, "w") as f:
        yaml.dump(valid_config_section, f)

    return config_path


@pytest.fixture
def input_dir(tmpdir):
    return tmpdir.join("input")


@pytest.fixture
def output_dir(tmpdir):
    return tmpdir.join("output")


class TestApplyMoleculePreprocessorToParquet:
    def test_completes_given_valid_smiles(
        self, smiles, input_dir, output_dir, config_path
    ):
        smiles.to_parquet(input_dir)

        apply_molecule_preprocessor_to_parquet(
            input_dir, output_dir, config_path, "SMILES"
        )

    def test_writes_dataframe_with_smiles_column(
        self, smiles, input_dir, output_dir, config_path
    ):
        smiles.to_parquet(input_dir)

        apply_molecule_preprocessor_to_parquet(
            input_dir, output_dir, config_path, "SMILES"
        )

        actual = pd.read_parquet(output_dir)

        assert isinstance(actual, pd.DataFrame)
        assert_index_equal(actual.columns, pd.Index(["SMILES"]))

    def test_raises_exception_given_incorrect_column_name(
        self, smiles, input_dir, output_dir, config_path
    ):
        smiles.to_parquet(input_dir)

        with pytest.raises(KeyError):
            apply_molecule_preprocessor_to_parquet(
                input_dir, output_dir, config_path, "smiles"
            )


class TestApplyMoleculePreprocessorToPartition:
    def test_completes_given_valid_smiles(self, smiles, config_path):
        apply_molecule_preprocessor_to_partition(smiles, config_path, "SMILES")

    def test_returns_dataframe_with_smiles_column(self, smiles, config_path):
        actual = apply_molecule_preprocessor_to_partition(smiles, config_path, "SMILES")

        assert isinstance(actual, pd.DataFrame)
        assert_index_equal(actual.columns, pd.Index(["SMILES"]))

    def test_returns_dataframe_with_no_missing_values(self, smiles, config_path):
        smiles["SMILES"][0] = "invalid smiles"

        actual = apply_molecule_preprocessor_to_partition(smiles, config_path, "SMILES")

        assert all(actual.notna())

    def test_removes_invalid_smiles_strings(self, smiles, config_path):
        smiles["SMILES"][0] = "invalid smiles"

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

    def test_completes(self, duplicate_smiles, input_dir, output_dir):
        duplicate_smiles.to_parquet(input_dir)

        drop_duplicates_and_repartition_parquet(input_dir, output_dir, column="SMILES")

    def test_raises_exception_given_incorrect_column_name(
        self, duplicate_smiles, input_dir, output_dir
    ):
        duplicate_smiles.to_parquet(input_dir)

        with pytest.raises(KeyError):
            drop_duplicates_and_repartition_parquet(
                input_dir, output_dir, column="smiles"
            )

    def test_raises_exception_given_missing_parquet(self, input_dir, output_dir):
        with pytest.raises(Exception):
            drop_duplicates_and_repartition_parquet(
                input_dir, output_dir, column="SMILES"
            )

    def test_writes_expected_parquet(
        self, duplicate_smiles, input_dir, output_dir, smiles
    ):
        duplicate_smiles.to_parquet(input_dir)

        drop_duplicates_and_repartition_parquet(input_dir, output_dir, column="SMILES")

        output_df = pd.read_parquet(output_dir)
        assert_frame_equal(output_df, smiles, check_names=False)


class TestCreateSelfiesFromSmiles:
    def test_completes_given_valid_smiles(self, smiles, input_dir, output_dir):
        smiles.to_parquet(input_dir)

        create_selfies_from_smiles(input_dir, output_dir, "SMILES")

    def test_writes_dataframe_with_expected_column(self, smiles, input_dir, output_dir):
        smiles.to_parquet(input_dir)

        create_selfies_from_smiles(input_dir, output_dir, "SMILES")

        actual = pd.read_parquet(output_dir)

        assert isinstance(actual, pd.DataFrame)
        assert_index_equal(actual.columns, pd.Index(["SELFIES"]))

    def test_writes_dataframe_with_selfies(self, smiles, input_dir, output_dir):
        smiles.to_parquet(input_dir)

        create_selfies_from_smiles(input_dir, output_dir, "SMILES")

        actual = pd.read_parquet(output_dir)

        assert all(actual["SELFIES"].str.contains(r"^(\[.+?\])*$", regex=True))

    def test_writes_dataframe_with_no_missing_values(
        self, smiles, input_dir, output_dir
    ):
        smiles["SMILES"][0] = "invalid smiles"
        smiles.to_parquet(input_dir)

        create_selfies_from_smiles(input_dir, output_dir, "SMILES")

        actual = pd.read_parquet(output_dir)

        assert all(actual["SELFIES"].notna())

    def test_removes_invalid_smiles_strings(self, smiles, input_dir, output_dir):
        smiles["SMILES"][0] = "invalid smiles"
        smiles.to_parquet(input_dir)

        create_selfies_from_smiles(input_dir, output_dir, "SMILES")

        actual = pd.read_parquet(output_dir)

        assert not any(actual["SELFIES"].str.contains("invalid smiles"))

    def test_raises_exception_given_incorrect_column_name(
        self, smiles, input_dir, output_dir
    ):
        smiles.to_parquet(input_dir)

        with pytest.raises(KeyError):
            create_selfies_from_smiles(input_dir, output_dir, "smiles")


@pytest.fixture
def selfies():
    return pd.DataFrame(
        [
            "[Br][C][Branch1][C][Br][=C][C][=N][C][=C][NH1][Ring1][Branch1]",
            "[O][N][=C][C][C][C][Ring1][Ring2][C][=C][C][=C][C][=C][Ring1][=Branch1][Cl]",
        ],
        columns=["SELFIES"],
    )


class TestGetSelfiesTokenCountsFromParquet:
    def test_completes_given_valid_selfies(self, selfies, input_dir, output_dir):
        selfies.to_parquet(input_dir)
        output_dir.mkdir()

        get_selfies_token_counts_from_parquet(input_dir, output_dir, "SELFIES")

    def test_writes_dataframe_with_expected_column(
        self, selfies, input_dir, output_dir
    ):
        selfies.to_parquet(input_dir)
        output_dir.mkdir()

        get_selfies_token_counts_from_parquet(input_dir, output_dir, "SELFIES")

        actual = pd.read_csv(output_dir / "token_counts.csv", index_col=0)

        assert isinstance(actual, pd.DataFrame)
        assert_index_equal(actual.columns, pd.Index(["count"]))
        assert actual.index.name == "token"

    def test_writes_dataframe_with_expected_number_of_tokens(
        self, selfies, input_dir, output_dir
    ):
        selfies.to_parquet(input_dir)
        output_dir.mkdir()

        get_selfies_token_counts_from_parquet(input_dir, output_dir, "SELFIES")

        actual = pd.read_csv(output_dir / "token_counts.csv", index_col=0)

        assert len(actual) == 12

    def test_writes_dataframe_with_expected_token_counts(
        self, selfies, input_dir, output_dir
    ):
        selfies.to_parquet(input_dir)
        output_dir.mkdir()

        get_selfies_token_counts_from_parquet(input_dir, output_dir, "SELFIES")

        actual = pd.read_csv(output_dir / "token_counts.csv", index_col=0)

        assert actual.loc["[C]", "count"] == 10
        assert actual.loc["[=C]", "count"] == 6
        assert actual.loc["[Ring1]", "count"] == 3
        assert actual.loc["[Br]", "count"] == 2
        assert actual.loc["[Branch1]", "count"] == 2
        assert actual.loc["[=N]", "count"] == 1
        assert actual.loc["[NH1]", "count"] == 1
        assert actual.loc["[Ring2]", "count"] == 1
        assert actual.loc["[=Branch1]", "count"] == 1
        assert actual.loc["[O]", "count"] == 1
        assert actual.loc["[N]", "count"] == 1
        assert actual.loc["[Cl]", "count"] == 1


class TestGetSelfiesTokensFromPartition:
    def test_completes_given_valid_selfies(self, selfies):
        get_selfies_tokens_from_partition(selfies, "SELFIES")

    def test_returns_series_with_expected_name(self, selfies):
        actual = get_selfies_tokens_from_partition(selfies, "SELFIES")

        assert isinstance(actual, pd.Series)
        assert actual.name == "SELFIES"

    def test_returns_series_with_expected_tokens(self, selfies):
        expected = pd.Series(
            [
                "[Br]",
                "[C]",
                "[Branch1]",
                "[C]",
                "[Br]",
                "[=C]",
                "[C]",
                "[=N]",
                "[C]",
                "[=C]",
                "[NH1]",
                "[Ring1]",
                "[Branch1]",
                "[O]",
                "[N]",
                "[=C]",
                "[C]",
                "[C]",
                "[C]",
                "[Ring1]",
                "[Ring2]",
                "[C]",
                "[=C]",
                "[C]",
                "[=C]",
                "[C]",
                "[=C]",
                "[Ring1]",
                "[=Branch1]",
                "[Cl]",
            ],
            name="SELFIES",
        )

        actual = get_selfies_tokens_from_partition(selfies, "SELFIES")

        assert_series_equal(actual, expected)

    def test_raises_exception_given_incorrect_column_name(self, selfies):
        with pytest.raises(KeyError):
            get_selfies_tokens_from_partition(selfies, "selfies")


class TestCreateSplitsFromParquet:
    @pytest.fixture
    def input_dir(self):
        return here().joinpath("tests", "data", "preprocessed", "smiles")

    @pytest.fixture
    def config(self, valid_config_section):
        return PreprocessingConfig.parse_config(valid_config_section).split

    def test_completes(self, tmpdir, input_dir, config):
        create_splits_from_parquet(input_dir, Path(tmpdir), config)

    def test_creates_splits(self, tmpdir, input_dir, config):
        create_splits_from_parquet(input_dir, Path(tmpdir), config)

        train_split = dd.read_csv(tmpdir.join("train", "*"), names=["SMILES"]).compute()
        assert isinstance(train_split, pd.DataFrame)
        assert len(train_split)

        validate_split = dd.read_csv(
            tmpdir.join("validate", "*"), names=["SMILES"]
        ).compute()
        assert isinstance(validate_split, pd.DataFrame)
        assert len(validate_split)

        test_split = dd.read_csv(tmpdir.join("test", "*"), names=["SMILES"]).compute()
        assert isinstance(test_split, pd.DataFrame)
        assert len(test_split)

    def test_creates_splits_with_expected_total_number_of_rows(
        self, tmpdir, input_dir, config
    ):
        create_splits_from_parquet(input_dir, Path(tmpdir), config)

        whole_dataset = pd.read_parquet(input_dir)

        train_split = dd.read_csv(tmpdir.join("train", "*"), names=["SMILES"]).compute()
        validate_split = dd.read_csv(
            tmpdir.join("validate", "*"), names=["SMILES"]
        ).compute()
        test_split = dd.read_csv(tmpdir.join("test", "*"), names=["SMILES"]).compute()

        assert len(train_split) + len(validate_split) + len(test_split) == len(
            whole_dataset
        )

    def test_creates_splits_that_do_not_overlap(self, tmpdir, input_dir, config):
        create_splits_from_parquet(input_dir, Path(tmpdir), config)

        train_split = dd.read_csv(tmpdir.join("train", "*"), names=["SMILES"]).compute()
        validate_split = dd.read_csv(
            tmpdir.join("validate", "*"), names=["SMILES"]
        ).compute()
        test_split = dd.read_csv(tmpdir.join("test", "*"), names=["SMILES"]).compute()

        assert not train_split["SMILES"].isin(validate_split["SMILES"]).sum()
        assert not train_split["SMILES"].isin(test_split["SMILES"]).sum()
        assert not validate_split["SMILES"].isin(test_split["SMILES"]).sum()
