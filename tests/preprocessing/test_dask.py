import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from mol_gen.preprocessing.dask import drop_duplicates_and_repartition_parquet


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
