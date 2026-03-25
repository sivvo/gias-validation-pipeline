"""Tests for gias_pipeline.filter."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from gias_pipeline.filter import (
    TYPES_MAINSTREAM,
    TYPES_INDEPENDENT,
    TYPES_NURSERY,
    TYPES_FE,
    FilterResult,
    filter_schools,
    _COL_TYPE,
    _COL_STATUS,
    _SCOPE_DEFAULTS,
)

FIXTURES_CSV = Path(__file__).parent.parent / "fixtures" / "test_schools.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _fixture_df() -> pd.DataFrame:
    return pd.read_csv(FIXTURES_CSV, dtype=str)


# ---------------------------------------------------------------------------
# Unit tests — default scope (all optional groups included, closed excluded)
# ---------------------------------------------------------------------------

class TestFilterBasic:
    def test_open_mainstream_type_is_active(self):
        df = _make_df([
            {_COL_TYPE: "Community school", _COL_STATUS: "Open"},
        ])
        result = filter_schools(df)
        assert len(result.active) == 1
        assert len(result.excluded) == 0

    def test_closed_school_excluded_by_default(self):
        df = _make_df([
            {_COL_TYPE: "Community school", _COL_STATUS: "Closed"},
        ])
        result = filter_schools(df)
        assert len(result.active) == 0
        assert len(result.excluded) == 1

    def test_proposed_school_is_excluded(self):
        df = _make_df([
            {_COL_TYPE: "Academy converter", _COL_STATUS: "Proposed to open"},
        ])
        result = filter_schools(df)
        assert len(result.active) == 0

    def test_always_excluded_type_is_excluded(self):
        # Welsh establishment is never in any group
        df = _make_df([
            {_COL_TYPE: "Welsh establishment", _COL_STATUS: "Open"},
        ])
        result = filter_schools(df)
        assert len(result.active) == 0
        assert len(result.excluded) == 1

    def test_all_mainstream_types_included(self):
        rows = [{_COL_TYPE: t, _COL_STATUS: "Open"} for t in TYPES_MAINSTREAM]
        df = _make_df(rows)
        result = filter_schools(df)
        assert len(result.active) == len(TYPES_MAINSTREAM)
        assert len(result.excluded) == 0

    def test_independent_included_by_default(self):
        df = _make_df([
            {_COL_TYPE: "Other independent school", _COL_STATUS: "Open"},
        ])
        result = filter_schools(df)
        assert len(result.active) == 1

    def test_nursery_included_by_default(self):
        df = _make_df([
            {_COL_TYPE: "Local authority nursery school", _COL_STATUS: "Open"},
        ])
        result = filter_schools(df)
        assert len(result.active) == 1

    def test_fe_included_by_default(self):
        df = _make_df([
            {_COL_TYPE: "Further education", _COL_STATUS: "Open"},
        ])
        result = filter_schools(df)
        assert len(result.active) == 1

    def test_mixed_df_counts(self):
        df = _make_df([
            {_COL_TYPE: "Community school", _COL_STATUS: "Open"},        # active
            {_COL_TYPE: "Community school", _COL_STATUS: "Closed"},       # excluded
            {_COL_TYPE: "Welsh establishment", _COL_STATUS: "Open"},      # excluded
            {_COL_TYPE: "Academy converter", _COL_STATUS: "Open"},        # active
        ])
        result = filter_schools(df)
        assert len(result.active) == 2
        assert len(result.excluded) == 2

    def test_totals_match_input(self):
        df = _make_df([
            {_COL_TYPE: "Community school", _COL_STATUS: "Open"},
            {_COL_TYPE: "Welsh establishment", _COL_STATUS: "Open"},
            {_COL_TYPE: "Community school", _COL_STATUS: "Closed"},
        ])
        result = filter_schools(df)
        assert len(result.active) + len(result.excluded) == len(df)

    def test_missing_type_column_raises(self):
        df = _make_df([{_COL_STATUS: "Open"}])
        with pytest.raises(ValueError, match="TypeOfEstablishment"):
            filter_schools(df)

    def test_missing_status_column_raises(self):
        df = _make_df([{_COL_TYPE: "Community school"}])
        with pytest.raises(ValueError, match="EstablishmentStatus"):
            filter_schools(df)

    def test_nan_type_treated_as_excluded(self):
        df = _make_df([
            {_COL_TYPE: None, _COL_STATUS: "Open"},
        ])
        result = filter_schools(df)
        assert len(result.active) == 0

    def test_nan_status_treated_as_excluded(self):
        df = _make_df([
            {_COL_TYPE: "Community school", _COL_STATUS: None},
        ])
        result = filter_schools(df)
        assert len(result.active) == 0

    def test_result_is_filterresult(self):
        df = _make_df([{_COL_TYPE: "Community school", _COL_STATUS: "Open"}])
        result = filter_schools(df)
        assert isinstance(result, FilterResult)
        assert isinstance(result.active, pd.DataFrame)
        assert isinstance(result.excluded, pd.DataFrame)

    def test_none_scope_uses_defaults(self):
        df = _make_df([{_COL_TYPE: "Community school", _COL_STATUS: "Open"}])
        assert filter_schools(df, scope=None).active.equals(
            filter_schools(df, scope={}).active
        )


# ---------------------------------------------------------------------------
# Unit tests — scope flags
# ---------------------------------------------------------------------------

class TestScopeFlags:
    def test_include_closed_true_includes_closed(self):
        df = _make_df([
            {_COL_TYPE: "Community school", _COL_STATUS: "Closed"},
        ])
        result = filter_schools(df, scope={"include_closed": True})
        assert len(result.active) == 1

    def test_include_closed_false_excludes_closed(self):
        df = _make_df([
            {_COL_TYPE: "Community school", _COL_STATUS: "Closed"},
        ])
        result = filter_schools(df, scope={"include_closed": False})
        assert len(result.active) == 0

    def test_include_independent_false_excludes_independent(self):
        df = _make_df([
            {_COL_TYPE: "Other independent school", _COL_STATUS: "Open"},
        ])
        result = filter_schools(df, scope={"include_independent": False})
        assert len(result.active) == 0

    def test_include_independent_true_includes_independent(self):
        df = _make_df([
            {_COL_TYPE: "Other independent school", _COL_STATUS: "Open"},
        ])
        result = filter_schools(df, scope={"include_independent": True})
        assert len(result.active) == 1

    def test_include_nursery_false_excludes_nursery(self):
        df = _make_df([
            {_COL_TYPE: "Local authority nursery school", _COL_STATUS: "Open"},
        ])
        result = filter_schools(df, scope={"include_nursery": False})
        assert len(result.active) == 0

    def test_include_fe_false_excludes_fe(self):
        df = _make_df([
            {_COL_TYPE: "Further education", _COL_STATUS: "Open"},
        ])
        result = filter_schools(df, scope={"include_fe": False})
        assert len(result.active) == 0

    def test_mainstream_unaffected_by_optional_flags_off(self):
        df = _make_df([
            {_COL_TYPE: "Community school", _COL_STATUS: "Open"},
        ])
        scope = {"include_independent": False, "include_nursery": False, "include_fe": False}
        result = filter_schools(df, scope=scope)
        assert len(result.active) == 1


# ---------------------------------------------------------------------------
# Integration tests — fixture CSV
# ---------------------------------------------------------------------------

class TestFilterFixtures:
    def test_fixture_csv_loads(self):
        df = _fixture_df()
        assert len(df) == 30

    def test_closed_school_excluded(self):
        df = _fixture_df()
        result = filter_schools(df)
        assert "100015" not in result.active["URN"].astype(str).values

    def test_independent_school_included_by_default(self):
        # URN 100016 is "Other independent school", Open
        df = _fixture_df()
        result = filter_schools(df)
        assert "100016" in result.active["URN"].astype(str).values

    def test_independent_school_excluded_when_flag_off(self):
        df = _fixture_df()
        result = filter_schools(df, scope={"include_independent": False})
        assert "100016" not in result.active["URN"].astype(str).values

    def test_proposed_school_excluded(self):
        df = _fixture_df()
        result = filter_schools(df)
        assert "100021" not in result.active["URN"].astype(str).values

    def test_special_school_included(self):
        df = _fixture_df()
        result = filter_schools(df)
        assert "100012" in result.active["URN"].astype(str).values

    def test_pru_included(self):
        df = _fixture_df()
        result = filter_schools(df)
        assert "100013" in result.active["URN"].astype(str).values

    def test_16_19_included(self):
        df = _fixture_df()
        result = filter_schools(df)
        assert "100014" in result.active["URN"].astype(str).values

    def test_academy_converter_included(self):
        df = _fixture_df()
        result = filter_schools(df)
        assert "100005" in result.active["URN"].astype(str).values

    def test_free_school_included(self):
        df = _fixture_df()
        result = filter_schools(df)
        assert "100017" in result.active["URN"].astype(str).values

    def test_active_count_with_defaults(self):
        # 30 total; excluded: 100015 (Closed), 100021 (Proposed to open) = 2
        # 100016 (independent, Open) is now included by default
        df = _fixture_df()
        result = filter_schools(df)
        assert "100015" not in result.active["URN"].astype(str).values
        assert "100021" not in result.active["URN"].astype(str).values
        assert len(result.active) + len(result.excluded) == 30
