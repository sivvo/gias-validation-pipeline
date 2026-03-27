"""Tests for gias_pipeline.enrich."""

from __future__ import annotations

import pytest

from gias_pipeline.enrich import (
    _apex_from_url,
    _email_confidence,
    _refine_email_domain,
)
from gias_pipeline.models import SchoolRecord


def _make_record(
    urn: str = "100001",
    url_original: str | None = "https://school.sch.uk",
    url_canonical: str | None = None,
    email_domain: str | None = None,
    email_domain_confidence: str = "low",
    is_reachable: bool | None = None,
    flags: list[str] | None = None,
) -> SchoolRecord:
    return SchoolRecord(
        urn=urn,
        name="Test School",
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
        email_domain_confidence=email_domain_confidence,
        http_status=None,
        is_reachable=is_reachable,
        redirect_chain=[],
        flags=flags or [],
        gias_last_updated="2024-01-01",
        pipeline_run_id="test-run",
    )


# ---------------------------------------------------------------------------
# Email domain refinement (no DNS)
# ---------------------------------------------------------------------------

class TestRefineEmailDomain:
    def test_sch_uk_canonical_gives_high_confidence(self):
        record = _make_record(
            url_canonical="https://myschool.sch.uk",
            is_reachable=True,
        )
        _refine_email_domain(record)
        assert record.email_domain == "myschool.sch.uk"
        assert record.email_domain_confidence == "high"

    def test_org_uk_canonical_gives_medium_confidence(self):
        record = _make_record(
            url_canonical="https://school.org.uk",
            is_reachable=True,
        )
        _refine_email_domain(record)
        assert record.email_domain == "school.org.uk"
        assert record.email_domain_confidence == "medium"

    def test_co_uk_canonical_gives_medium_confidence(self):
        record = _make_record(
            url_canonical="https://school.co.uk",
            is_reachable=True,
        )
        _refine_email_domain(record)
        assert record.email_domain_confidence == "medium"

    def test_ac_uk_canonical_gives_medium_confidence(self):
        record = _make_record(
            url_canonical="https://college.ac.uk",
            is_reachable=True,
        )
        _refine_email_domain(record)
        assert record.email_domain_confidence == "medium"

    def test_other_tld_gives_low_confidence(self):
        record = _make_record(
            url_canonical="https://school.example.com",
            is_reachable=True,
        )
        _refine_email_domain(record)
        assert record.email_domain == "school.example.com"
        assert record.email_domain_confidence == "low"

    def test_unreachable_uses_original_url_with_unverified_flag(self):
        record = _make_record(
            url_original="https://gone.example.com",
            url_canonical=None,
            email_domain="gone.example.com",
            is_reachable=False,
        )
        _refine_email_domain(record)
        assert record.email_domain == "gone.example.com"
        assert record.email_domain_confidence == "low"
        assert "email_domain_unverified" in record.flags

    def test_no_url_gives_none_email_domain(self):
        record = _make_record(
            url_original=None,
            url_canonical=None,
            is_reachable=None,
            flags=["domain_missing"],
        )
        _refine_email_domain(record)
        assert record.email_domain is None

    def test_no_dns_calls_made(self):
        """Refinement must never make network calls (DNS or HTTP)."""
        import socket
        original_getaddrinfo = socket.getaddrinfo

        def fail_if_called(*args, **kwargs):
            raise AssertionError("DNS lookup called — should not happen")

        socket.getaddrinfo = fail_if_called
        try:
            record = _make_record(
                url_canonical="https://school.example.com",
                is_reachable=True,
            )
            _refine_email_domain(record)
        finally:
            socket.getaddrinfo = original_getaddrinfo


class TestEmailConfidence:
    def test_sch_uk_is_high(self):
        assert _email_confidence("school.sch.uk") == "high"

    def test_ac_uk_is_medium(self):
        assert _email_confidence("college.ac.uk") == "medium"

    def test_org_uk_is_medium(self):
        assert _email_confidence("trust.org.uk") == "medium"

    def test_co_uk_is_medium(self):
        assert _email_confidence("school.co.uk") == "medium"

    def test_com_is_low(self):
        assert _email_confidence("school.com") == "low"

    def test_gov_uk_is_low(self):
        assert _email_confidence("la.gov.uk") == "low"

class TestApexFromUrl:
    def test_strips_www(self):
        assert _apex_from_url("https://www.school.sch.uk") == "school.sch.uk"

    def test_no_www(self):
        assert _apex_from_url("https://school.sch.uk/home") == "school.sch.uk"

    def test_adds_scheme(self):
        assert _apex_from_url("school.sch.uk") == "school.sch.uk"

    def test_none_returns_none(self):
        assert _apex_from_url(None) is None

    def test_empty_returns_none(self):
        assert _apex_from_url("") is None


