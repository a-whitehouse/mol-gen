import pytest

from mol_gen.config.preprocessing.filter import (
    ElementsFilter,
    FilterConfig,
    RangeFilter,
)
from mol_gen.exceptions import ConfigException


class TestFilterConfig:
    @pytest.fixture
    def valid_config_section(self):
        return {
            "allowed_elements": ["H", "C", "N", "O", "F", "S", "Cl", "Br"],
            "range_filters": {
                "molecular_weight": {"min": 180, "max": 480},
                "partition_coefficient": {"min": -0.4},
                "rotatable_bonds": {"max": 10},
            },
        }

    def test_parse_config_completes_given_valid_config_section(
        self, valid_config_section
    ):
        FilterConfig.parse_config(valid_config_section)

    def test_parse_config_returns_expected_config_given_valid_config_section(
        self, valid_config_section
    ):
        config = FilterConfig.parse_config(valid_config_section)

        assert isinstance(config, FilterConfig)

    def test_parse_config_sets_expected_allowed_elements_given_valid_config_section(
        self, valid_config_section
    ):
        config = FilterConfig.parse_config(valid_config_section)
        allowed_elements = config.elements_filter.allowed_elements

        assert isinstance(config.elements_filter, ElementsFilter)
        assert allowed_elements == ["H", "C", "N", "O", "F", "S", "Cl", "Br"]

    @pytest.mark.parametrize(
        "expected_descriptor, expected_min, expected_max",
        [
            ("molecular_weight", 180, 480),
            (
                "partition_coefficient",
                -0.4,
                None,
            ),
            ("rotatable_bonds", None, 10),
        ],
    )
    def test_parse_config_sets_expected_range_filters_given_valid_config_section(
        self, valid_config_section, expected_descriptor, expected_min, expected_max
    ):
        config = FilterConfig.parse_config(valid_config_section)
        range_filter = config.range_filters[expected_descriptor]

        assert isinstance(range_filter, RangeFilter)
        assert range_filter.descriptor == expected_descriptor
        assert range_filter.min == expected_min
        assert range_filter.max == expected_max

    def test_parse_config_sets_expected_range_filters_as_empty_given_none_requested(
        self,
    ):
        config = FilterConfig.parse_config(
            {
                "allowed_elements": ["H", "C", "N", "O", "F", "S", "Cl", "Br"],
            }
        )

        assert config.range_filters == {}


class TestElementsFilter:
    @pytest.fixture
    def valid_config_section(self):
        return ["H", "C", "N", "O", "F", "S", "Cl", "Br"]

    def test_parse_config_completes_given_valid_config_section(
        self, valid_config_section
    ):
        ElementsFilter.parse_config(valid_config_section)

    def test_parse_config_returns_expected_config_given_valid_config_section(
        self, valid_config_section
    ):
        config = ElementsFilter.parse_config(valid_config_section)

        assert isinstance(config, ElementsFilter)

    def test_parse_config_sets_expected_attributes_given_valid_config_section(
        self, valid_config_section
    ):
        config = ElementsFilter.parse_config(valid_config_section)

        assert config.allowed_elements == ["H", "C", "N", "O", "F", "S", "Cl", "Br"]

    @pytest.mark.parametrize(
        "allowed_elements, expected",
        [
            (["D", "C", "N"], "Element D is not recognised."),
            (["H", "c", "N"], "Element c is not recognised."),
        ],
    )
    def test_parse_config_raises_exception_given_invalid_allowed_elements(
        self, allowed_elements, expected
    ):
        with pytest.raises(ConfigException) as excinfo:
            ElementsFilter.parse_config(allowed_elements)

        assert str(excinfo.value) == expected


class TestRangeFilter:
    @pytest.fixture
    def valid_config_section(self):
        return {"molecular_weight": {"min": 180, "max": 480}}

    def test_parse_config_completes_given_valid_config_section(
        self, valid_config_section
    ):
        RangeFilter.parse_config(valid_config_section)

    def test_parse_config_returns_expected_config_given_valid_config_section(
        self, valid_config_section
    ):
        config = RangeFilter.parse_config(valid_config_section)

        assert isinstance(config, RangeFilter)

    @pytest.mark.parametrize(
        "valid_config_section, expected_descriptor, expected_min, expected_max",
        [
            (
                {"molecular_weight": {"min": 180, "max": 480}},
                "molecular_weight",
                180,
                480,
            ),
            ({"molecular_weight": {"min": 180}}, "molecular_weight", 180, None),
            ({"molecular_weight": {"max": 480}}, "molecular_weight", None, 480),
            ({"molecular_weight": {}}, "molecular_weight", None, None),
        ],
    )
    def test_parse_config_sets_expected_attributes_given_valid_config_section(
        self, valid_config_section, expected_descriptor, expected_min, expected_max
    ):
        config = RangeFilter.parse_config(valid_config_section)

        assert config.descriptor == expected_descriptor
        assert config.min == expected_min
        assert config.max == expected_max

    def test_parse_config_raises_exception_given_invalid_descriptor(self):
        with pytest.raises(ConfigException) as excinfo:
            RangeFilter.parse_config({"unrecognised": {"min": 180, "max": 480}})

        assert str(excinfo.value) == "Descriptor unrecognised is not recognised."

    def test_parse_config_raises_exception_given_invalid_min_value(self):
        with pytest.raises(ConfigException) as excinfo:
            RangeFilter.parse_config({"molecular_weight": {"min": "180", "max": 480}})

        assert (
            str(excinfo.value)
            == "Minimum value 180 for molecular_weight is not a number."
        )

    def test_parse_config_raises_exception_given_invalid_max_value(self):
        with pytest.raises(ConfigException) as excinfo:
            RangeFilter.parse_config({"molecular_weight": {"min": 180, "max": "480"}})

        assert (
            str(excinfo.value)
            == "Maximum value 480 for molecular_weight is not a number."
        )
