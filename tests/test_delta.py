"""Tests for gias_pipeline.delta."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from gias_pipeline.delta import DeltaResult, compute_delta
from gias_pipeline.models import SchoolRecord


def _make_record(
    urn: str,
    url_canonical: str | None = None,
) -> SchoolRecord:
    return SchoolRecord(
        urn=urn,
        name=f"School {urn}",
        phase="Primary",
        establishment_type="Community school",
        status="Open",
        la_code="001",
        la_name="Test LA",
        url_original="https://example.com",
        url_canonical=url_canonical,
        url_confidence="high",
        url_source="gias_direct",
        email_domain=None,
        email_domain_confidence="low",
        http_status=200,
        is_reachable=True,
        redirect_chain=[],
        flags=[],
        gias_last_updated="2024-01-01",
        pipeline_run_id="test-run",
    )


def _write_previous_csv(output_dir: Path, rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "schools.csv", index=False)


class TestDeltaFirstRun:
    def test_no_previous_csv_returns_empty_delta(self, tmp_path):
        records = [_make_record("100001", "https://a.sch.uk")]
        result = compute_delta(records, output_dir=tmp_path)
        assert isinstance(result, DeltaResult)
        assert len(result.added) == 0
        assert len(result.removed) == 0
        assert len(result.changed) == 0

    def test_empty_output_dir_no_error(self, tmp_path):
        result = compute_delta([], output_dir=tmp_path)
        assert len(result.added) == 0


class TestDeltaAdded:
    def test_new_urn_appears_in_added(self, tmp_path):
        _write_previous_csv(tmp_path, [
            {"urn": "100001", "name": "Old School", "url_canonical": "https://old.sch.uk"},
        ])
        records = [
            _make_record("100001", "https://old.sch.uk"),
            _make_record("100002", "https://new.sch.uk"),
        ]
        result = compute_delta(records, output_dir=tmp_path)
        assert "100002" in result.added["urn"].values
        assert "100001" not in result.added["urn"].values

    def test_added_count_correct(self, tmp_path):
        # Write a previous CSV with headers but no data rows
        _write_previous_csv(tmp_path, [
            {"urn": "999999", "name": "Old", "url_canonical": ""},
        ])
        records = [
            _make_record("100001"),
            _make_record("100002"),
        ]
        result = compute_delta(records, output_dir=tmp_path)
        assert len(result.added) == 2


class TestDeltaRemoved:
    def test_missing_urn_appears_in_removed(self, tmp_path):
        _write_previous_csv(tmp_path, [
            {"urn": "100001", "name": "Still Here", "url_canonical": "https://a.sch.uk"},
            {"urn": "100002", "name": "Gone School", "url_canonical": "https://b.sch.uk"},
        ])
        records = [_make_record("100001", "https://a.sch.uk")]
        result = compute_delta(records, output_dir=tmp_path)
        assert "100002" in result.removed["urn"].values
        assert "100001" not in result.removed["urn"].values

    def test_removed_count_correct(self, tmp_path):
        _write_previous_csv(tmp_path, [
            {"urn": "100001", "name": "A", "url_canonical": ""},
            {"urn": "100002", "name": "B", "url_canonical": ""},
            {"urn": "100003", "name": "C", "url_canonical": ""},
        ])
        records = [_make_record("100001")]
        result = compute_delta(records, output_dir=tmp_path)
        assert len(result.removed) == 2


class TestDeltaChanged:
    def test_url_change_detected(self, tmp_path):
        _write_previous_csv(tmp_path, [
            {"urn": "100001", "name": "School", "url_canonical": "https://old.sch.uk"},
        ])
        records = [_make_record("100001", "https://new.sch.uk")]
        result = compute_delta(records, output_dir=tmp_path)
        assert len(result.changed) == 1
        row = result.changed.iloc[0]
        assert row["old_url_canonical"] == "https://old.sch.uk"
        assert row["new_url_canonical"] == "https://new.sch.uk"

    def test_no_change_not_in_changed(self, tmp_path):
        _write_previous_csv(tmp_path, [
            {"urn": "100001", "name": "School", "url_canonical": "https://same.sch.uk"},
        ])
        records = [_make_record("100001", "https://same.sch.uk")]
        result = compute_delta(records, output_dir=tmp_path)
        assert len(result.changed) == 0

    def test_combined_added_removed_changed(self, tmp_path):
        _write_previous_csv(tmp_path, [
            {"urn": "100001", "name": "Unchanged", "url_canonical": "https://a.sch.uk"},
            {"urn": "100002", "name": "Changed", "url_canonical": "https://old.sch.uk"},
            {"urn": "100003", "name": "Removed", "url_canonical": "https://c.sch.uk"},
        ])
        records = [
            _make_record("100001", "https://a.sch.uk"),
            _make_record("100002", "https://new.sch.uk"),  # changed
            _make_record("100004", "https://d.sch.uk"),    # added
        ]
        result = compute_delta(records, output_dir=tmp_path)
        assert "100004" in result.added["urn"].values
        assert "100003" in result.removed["urn"].values
        assert "100002" in result.changed["urn"].values
        assert len(result.changed) == 1
