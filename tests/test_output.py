"""Tests for gias_pipeline.output."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from gias_pipeline.delta import DeltaResult
from gias_pipeline.models import SchoolRecord
from gias_pipeline.output import write_outputs


def _make_record(
    urn: str,
    url_original: str | None = "https://school.sch.uk",
    url_canonical: str | None = "https://school.sch.uk",
    email_domain: str | None = "school.sch.uk",
    is_reachable: bool | None = True,
    flags: list[str] | None = None,
) -> SchoolRecord:
    return SchoolRecord(
        urn=urn,
        name=f"School {urn}",
        phase="Primary",
        establishment_type="Community school",
        status="Open",
        la_code="001",
        la_name="Test LA",
        url_original=url_original,
        url_canonical=url_canonical,
        url_confidence="high",
        url_source="gias_direct",
        email_domain=email_domain,
        email_domain_confidence="high",
        http_status=200 if is_reachable else 404,
        is_reachable=is_reachable,
        redirect_chain=[],
        flags=flags or [],
        gias_last_updated="2024-01-01",
        pipeline_run_id="test-run",
    )


def _empty_delta() -> DeltaResult:
    return DeltaResult(
        added=pd.DataFrame(columns=["urn", "name", "url_canonical"]),
        removed=pd.DataFrame(columns=["urn", "name", "url_canonical"]),
        changed=pd.DataFrame(
            columns=["urn", "name", "old_url_canonical", "new_url_canonical"]
        ),
    )


def _base_summary() -> dict:
    return {
        "run_id": "test-uuid",
        "run_at": "2024-01-01T00:00:00Z",
        "gias_date": "20240101",
        "runtime_seconds": 1.0,
    }


class TestAllFilesCreated:
    def test_all_output_files_exist(self, tmp_path):
        records = [
            _make_record("100001"),
            _make_record("100002", flags=["la_hosted"]),
            _make_record("100003", url_original=None, url_canonical=None,
                         email_domain=None, is_reachable=None,
                         flags=["domain_missing"]),
            _make_record("100004", is_reachable=False, url_canonical=None,
                         flags=["unreachable"]),
        ]
        write_outputs(records, _empty_delta(), tmp_path, _base_summary())

        expected_files = [
            "schools.csv",
            "email_domains.txt",
            "urls_canonical.txt",
            "urls_original.txt",
            "unreachable.csv",
            "manual_review.csv",
            "la_hosted.csv",
            "delta_added.csv",
            "delta_removed.csv",
            "delta_changed.csv",
            "run_summary.json",
        ]
        for fname in expected_files:
            assert (tmp_path / fname).exists(), f"Missing: {fname}"


class TestSchoolsCsv:
    def test_all_records_written(self, tmp_path):
        records = [_make_record(str(i)) for i in range(5)]
        write_outputs(records, _empty_delta(), tmp_path, _base_summary())
        df = pd.read_csv(tmp_path / "schools.csv", dtype=str)
        assert len(df) == 5

    def test_column_order(self, tmp_path):
        records = [_make_record("100001")]
        write_outputs(records, _empty_delta(), tmp_path, _base_summary())
        df = pd.read_csv(tmp_path / "schools.csv")
        expected_cols = [
            "urn", "name", "phase", "establishment_type", "la_code", "la_name",
            "url_original", "url_canonical",
            "url_confidence", "url_source", "is_reachable", "http_status",
            "email_domain", "email_domain_confidence", "flags",
            "gias_last_updated", "pipeline_run_id",
        ]
        assert list(df.columns) == expected_cols

    def test_flags_pipe_separated(self, tmp_path):
        records = [_make_record("100001", flags=["la_hosted", "redirect_followed"])]
        write_outputs(records, _empty_delta(), tmp_path, _base_summary())
        df = pd.read_csv(tmp_path / "schools.csv", dtype=str)
        assert df.iloc[0]["flags"] == "la_hosted|redirect_followed"


class TestEmailDomainsTxt:
    def test_deduplicated_and_sorted(self, tmp_path):
        records = [
            _make_record("100001", email_domain="z.sch.uk"),
            _make_record("100002", email_domain="a.sch.uk"),
            _make_record("100003", email_domain="z.sch.uk"),  # duplicate
        ]
        write_outputs(records, _empty_delta(), tmp_path, _base_summary())
        lines = (tmp_path / "email_domains.txt").read_text().splitlines()
        non_empty = [l for l in lines if l]
        assert non_empty == sorted(set(non_empty))
        assert non_empty.count("z.sch.uk") == 1

    def test_none_email_domains_excluded(self, tmp_path):
        records = [
            _make_record("100001", email_domain=None,
                         flags=["domain_missing"], url_original=None,
                         url_canonical=None, is_reachable=None),
        ]
        write_outputs(records, _empty_delta(), tmp_path, _base_summary())
        content = (tmp_path / "email_domains.txt").read_text()
        assert content.strip() == ""


class TestRunSummaryJson:
    def test_valid_json(self, tmp_path):
        records = [_make_record("100001")]
        write_outputs(records, _empty_delta(), tmp_path, _base_summary())
        content = (tmp_path / "run_summary.json").read_text()
        data = json.loads(content)
        assert isinstance(data, dict)

    def test_all_expected_keys_present(self, tmp_path):
        records = [_make_record("100001")]
        write_outputs(records, _empty_delta(), tmp_path, _base_summary())
        data = json.loads((tmp_path / "run_summary.json").read_text())
        required_keys = [
            "run_id", "run_at", "gias_date", "total_processed",
            "reachable", "unreachable", "no_url", "url_confidence",
            "flags", "delta", "runtime_seconds",
        ]
        for key in required_keys:
            assert key in data, f"Missing key: {key}"

    def test_counts_correct(self, tmp_path):
        records = [
            _make_record("100001", is_reachable=True),
            _make_record("100002", is_reachable=False, url_canonical=None,
                         flags=["unreachable"]),
            _make_record("100003", url_original=None, url_canonical=None,
                         email_domain=None, is_reachable=None,
                         flags=["domain_missing"]),
        ]
        write_outputs(records, _empty_delta(), tmp_path, _base_summary())
        data = json.loads((tmp_path / "run_summary.json").read_text())
        assert data["total_processed"] == 3
        assert data["reachable"] == 1
        assert data["unreachable"] == 1
        assert data["no_url"] == 1


class TestSubsetFiles:
    def test_unreachable_csv_contains_only_unreachable(self, tmp_path):
        records = [
            _make_record("100001", is_reachable=True),
            _make_record("100002", is_reachable=False, url_canonical=None,
                         flags=["unreachable"]),
        ]
        write_outputs(records, _empty_delta(), tmp_path, _base_summary())
        df = pd.read_csv(tmp_path / "unreachable.csv", dtype=str)
        assert len(df) == 1
        assert df.iloc[0]["urn"] == "100002"

    def test_manual_review_contains_domain_missing(self, tmp_path):
        records = [
            _make_record("100001"),
            _make_record("100002", url_original=None, url_canonical=None,
                         email_domain=None, is_reachable=None,
                         flags=["domain_missing"]),
        ]
        write_outputs(records, _empty_delta(), tmp_path, _base_summary())
        df = pd.read_csv(tmp_path / "manual_review.csv", dtype=str)
        assert "100002" in df["urn"].values

    def test_la_hosted_csv_contains_la_hosted(self, tmp_path):
        records = [
            _make_record("100001"),
            _make_record("100002", flags=["la_hosted"]),
        ]
        write_outputs(records, _empty_delta(), tmp_path, _base_summary())
        df = pd.read_csv(tmp_path / "la_hosted.csv", dtype=str)
        assert len(df) == 1
        assert df.iloc[0]["urn"] == "100002"

