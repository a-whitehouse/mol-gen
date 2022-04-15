import pytest
from mol_gen.filter_molecules import check_value_within_range
from mol_gen.exceptions import UndesirableMolecule


class TestCheckValueWithinRange:
    def test_allows_value_given_min_and_max_undefined(self):
        check_value_within_range(5)

    @pytest.mark.parametrize("value", [4, 4.1])
    def test_allows_value_given_min_not_exceeded(self, value):
        check_value_within_range(value, min=4)

    @pytest.mark.parametrize("value", [5.9, 6])
    def test_allows_value_given_max_not_exceeded(self, value):
        check_value_within_range(value, max=6)

    def test_raises_exception_given_min_exceeded(self):
        with pytest.raises(UndesirableMolecule):
            check_value_within_range(3.9, min=4)

    def test_raises_exception_given_max_exceeded(self):
        with pytest.raises(UndesirableMolecule):
            check_value_within_range(6.1, max=6)
