"""Tests for gias_pipeline.extract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from gias_pipeline.extract import (
    _classify_row,
    _ends_with_la_domain,
    _finalise,
    _is_social,
    _parse_url,
    _PartialRecord,
    extract,
)
from gias_pipeline.filter import filter_schools
from gias_pipeline.models import SchoolRecord

FIXTURES_CSV = Path(__file__).parent.parent / "fixtures" / "test_schools.csv"
DATA_DIR = Path(__file__).parent.parent / "data"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fixture_df() -> pd.DataFrame:
    return pd.read_csv(FIXTURES_CSV, dtype=str)


@pytest.fixture(scope="module")
def active_df(fixture_df) -> pd.DataFrame:
    return filter_schools(fixture_df).active


@pytest.fixture(scope="module")
def records(active_df) -> dict[str, SchoolRecord]:
    return {r.urn: r for r in extract(active_df, data_dir=DATA_DIR, run_id="test-run")}


# ---------------------------------------------------------------------------
# Unit tests — pure helpers
# ---------------------------------------------------------------------------

class TestParseUrl:
    def test_full_https_url(self):
        fqdn, apex, canonical = _parse_url("https://www.school.sch.uk")
        assert fqdn == "www.school.sch.uk"
        assert apex == "school.sch.uk"
        assert canonical == "https://www.school.sch.uk"

    def test_no_protocol_prepends_https(self):
        _, _, canonical = _parse_url("www.school.sch.uk")
        assert canonical == "https://www.school.sch.uk"

    def test_http_url_netloc_extracted(self):
        fqdn, _, _ = _parse_url("http://www.school.sch.uk")
        assert fqdn == "www.school.sch.uk"

    def test_with_path_returns_fqdn_only(self):
        fqdn, _, _ = _parse_url("https://www.school.sch.uk/home/")
        assert fqdn == "www.school.sch.uk"

    def test_empty_string_returns_none(self):
        assert _parse_url("") is None

    def test_strip_www_from_apex(self):
        _, apex, _ = _parse_url("https://www.example.sch.uk")
        assert apex == "example.sch.uk"

    def test_no_www_apex_unchanged(self):
        _, apex, _ = _parse_url("https://example.sch.uk")
        assert apex == "example.sch.uk"


class TestIsSocial:
    def test_facebook_detected(self):
        assert _is_social("facebook.com")

    def test_x_com_detected(self):
        assert _is_social("x.com")

    def test_twitter_detected(self):
        assert _is_social("twitter.com")

    def test_school_sch_uk_not_social(self):
        assert not _is_social("myschool.sch.uk")

    def test_subdomain_of_facebook_detected(self):
        assert _is_social("m.facebook.com")


class TestEndsWithLaDomain:
    LA = frozenset({"sandwell.sch.uk", "kent.sch.uk", "northyorks.gov.uk"})

    def test_sch_uk_namespace_match(self):
        # school.sandwell.sch.uk → apex = school.sandwell.sch.uk
        assert _ends_with_la_domain("eatonvalley.sandwell.sch.uk", self.LA)

    def test_gov_uk_match(self):
        assert _ends_with_la_domain("schools.northyorks.gov.uk", self.LA)

    def test_exact_apex_match(self):
        assert _ends_with_la_domain("sandwell.sch.uk", self.LA)

    def test_direct_sch_uk_not_matched(self):
        # A school with its own .sch.uk should NOT be flagged la_hosted
        assert not _ends_with_la_domain("myschool.sch.uk", self.LA)

    def test_no_match(self):
        assert not _ends_with_la_domain("myschool.co.uk", self.LA)

    def test_case_insensitive(self):
        assert _ends_with_la_domain("EATONVALLEY.SANDWELL.SCH.UK", self.LA)


# ---------------------------------------------------------------------------
# Unit tests — _classify_row
# ---------------------------------------------------------------------------

LA_DOMAINS = frozenset({"sandwell.sch.uk", "kent.sch.uk", "northyorks.gov.uk"})
SOCIAL_DOMAINS = frozenset({"facebook.com", "twitter.com", "x.com", "instagram.com",
                            "linkedin.com", "youtube.com", "tiktok.com"})


def _make_row(website: str = "", name: str = "Test School", urn: str = "999001") -> pd.Series:
    return pd.Series({
        "URN": urn,
        "EstablishmentName": name,
        "PhaseOfEducation (name)": "Primary",
        "TypeOfEstablishment (name)": "Community school",
        "EstablishmentStatus (name)": "Open",
        "LA (code)": "810",
        "LA (name)": "North Yorkshire",
        "Trusts (UID)": "",
        "Trusts (name)": "",
        "SchoolWebsite": website,
        "LastChangedDate": "01/09/2023",
    })


class TestClassifyRow:
    def test_junk_url_manual_review(self):
        p = _classify_row(_make_row("N/A"), LA_DOMAINS, SOCIAL_DOMAINS, "r1")
        assert "domain_missing" in p.flags
        assert p.url_source == "manual"
        assert p.url_original is None
        assert p.email_domain_direct is None

    def test_blank_url_manual_review(self):
        p = _classify_row(_make_row(""), LA_DOMAINS, SOCIAL_DOMAINS, "r1")
        assert "domain_missing" in p.flags
        assert p.url_source == "manual"

    def test_dash_manual_review(self):
        p = _classify_row(_make_row("-"), LA_DOMAINS, SOCIAL_DOMAINS, "r1")
        assert "domain_missing" in p.flags

    def test_none_string_manual_review(self):
        p = _classify_row(_make_row("none"), LA_DOMAINS, SOCIAL_DOMAINS, "r1")
        assert "domain_missing" in p.flags

    def test_see_above_manual_review(self):
        p = _classify_row(_make_row("see above"), LA_DOMAINS, SOCIAL_DOMAINS, "r1")
        assert "domain_missing" in p.flags

    def test_social_url_manual_review(self):
        p = _classify_row(_make_row("https://www.facebook.com/school"), LA_DOMAINS, SOCIAL_DOMAINS, "r1")
        assert "social_media_only" in p.flags
        assert p.url_source == "manual"
        assert p.url_original is None

    def test_la_sch_uk_namespace_flagged(self):
        # school.sandwell.sch.uk → la_hosted
        p = _classify_row(_make_row("www.myschool.sandwell.sch.uk"), LA_DOMAINS, SOCIAL_DOMAINS, "r1")
        assert "la_hosted" in p.flags
        assert p.url_confidence == "medium"
        assert p.url_source == "gias_direct"
        assert p.url_original is not None

    def test_la_gov_uk_flagged(self):
        p = _classify_row(_make_row("northyorks.gov.uk/schools/test"), LA_DOMAINS, SOCIAL_DOMAINS, "r1")
        assert "la_hosted" in p.flags
        assert p.url_confidence == "medium"

    def test_high_confidence_own_sch_uk(self):
        p = _classify_row(_make_row("www.test-school.sch.uk"), LA_DOMAINS, SOCIAL_DOMAINS, "r1")
        assert not p.flags
        assert p.url_confidence == "high"
        assert p.url_source == "gias_direct"
        assert p.email_domain_direct == "test-school.sch.uk"
        assert p.email_domain_confidence_direct == "high"

    def test_high_confidence_co_uk_email_fallback(self):
        p = _classify_row(_make_row("www.myschool.co.uk"), LA_DOMAINS, SOCIAL_DOMAINS, "r1")
        assert p.url_confidence == "high"
        assert p.email_domain_direct == "myschool.co.uk"   # apex fallback
        assert p.email_domain_confidence_direct == "low"

    def test_urn_propagated(self):
        p = _classify_row(_make_row("www.x.sch.uk", urn="123456"), LA_DOMAINS, SOCIAL_DOMAINS, "r1")
        assert p.urn == "123456"

    def test_run_id_propagated(self):
        p = _classify_row(_make_row("www.x.sch.uk"), LA_DOMAINS, SOCIAL_DOMAINS, "my-run")
        assert p.pipeline_run_id == "my-run"


# ---------------------------------------------------------------------------
# Unit tests — _finalise
# ---------------------------------------------------------------------------

class TestFinalise:
    def _base_partial(self, **kwargs) -> _PartialRecord:
        defaults = dict(
            urn="999001", name="Test School", phase="Primary",
            establishment_type="Community school", status="Open",
            la_code="810", la_name="North Yorkshire",
            gias_last_updated="01/09/2023", pipeline_run_id="run1",
            url_original=None, url_source="manual",
            url_confidence="low", apex=None,
            flags=["domain_missing"],
        )
        defaults.update(kwargs)
        return _PartialRecord(**defaults)

    def test_manual_review_record_has_no_url(self):
        p = self._base_partial()
        r = _finalise(p)
        assert r.url_original is None
        assert r.url_source == "manual"
        assert "domain_missing" in r.flags

    def test_gias_direct_high_confidence(self):
        p = self._base_partial(
            url_original="www.school.sch.uk",
            url_source="gias_direct",
            url_confidence="high",
            apex="school.sch.uk",
            flags=[],
            email_domain_direct="school.sch.uk",
            email_domain_confidence_direct="high",
        )
        r = _finalise(p)
        assert r.url_original == "www.school.sch.uk"
        assert r.url_confidence == "high"
        assert r.email_domain == "school.sch.uk"
        assert r.email_domain_confidence == "high"
        assert not r.flags

    def test_la_hosted_record(self):
        p = self._base_partial(
            url_original="www.school.sandwell.sch.uk",
            url_source="gias_direct",
            url_confidence="medium",
            apex="school.sandwell.sch.uk",
            flags=["la_hosted"],
            email_domain_direct="school.sandwell.sch.uk",
            email_domain_confidence_direct="high",
        )
        r = _finalise(p)
        assert "la_hosted" in r.flags
        assert r.url_confidence == "medium"

    def test_session2_fields_are_none(self):
        p = self._base_partial()
        r = _finalise(p)
        assert r.url_canonical is None
        assert r.http_status is None
        assert r.is_reachable is None
        assert r.redirect_chain == []


# ---------------------------------------------------------------------------
# Integration tests — extract() against fixture data
# ---------------------------------------------------------------------------

class TestExtractFixtureCases:
    """Test each of the 16 fixture cases end-to-end (no mocking needed)."""

    # Case 1: Clean .sch.uk URL (happy path)
    def test_case1_clean_sch_uk(self, records):
        r = records["100001"]
        assert r.url_confidence == "high"
        assert r.url_source == "gias_direct"
        assert r.url_original is not None
        assert "sch.uk" in r.url_original
        assert not r.flags
        assert r.email_domain_confidence == "high"

    # Case 2: URL with www prefix
    def test_case2_www_prefix(self, records):
        r = records["100002"]
        assert r.url_confidence == "high"
        assert r.url_source == "gias_direct"
        assert "www" in (r.url_original or "")

    # Case 3: HTTP (not HTTPS) URL — still classified correctly
    def test_case3_http_url(self, records):
        r = records["100003"]
        assert r.url_confidence == "high"
        assert r.url_source == "gias_direct"

    # Case 4: LA-hosted URL (northyorks.gov.uk)
    def test_case4_la_hosted(self, records):
        r = records["100004"]
        assert "la_hosted" in r.flags
        assert r.url_confidence == "medium"
        assert r.url_source == "gias_direct"

    # Case 5: Facebook URL only → manual review
    def test_case5_facebook_only(self, records):
        r = records["100005"]
        assert "social_media_only" in r.flags
        assert r.url_source == "manual"
        assert r.url_original is None

    # Case 6: Blank SchoolWebsite → manual review
    def test_case6_blank_website(self, records):
        r = records["100006"]
        assert "domain_missing" in r.flags
        assert r.url_source == "manual"
        assert r.url_original is None

    # Case 7: "N/A" in SchoolWebsite → manual review
    def test_case7_na_website(self, records):
        r = records["100007"]
        assert "domain_missing" in r.flags
        assert r.url_source == "manual"

    # Case 8: URL with trailing slash and path
    def test_case8_url_with_path(self, records):
        r = records["100008"]
        assert r.url_confidence == "high"
        assert r.url_source == "gias_direct"
        assert r.url_original is not None

    # Case 9: .academy.org.uk domain
    def test_case9_academy_domain(self, records):
        r = records["100009"]
        assert r.url_source == "gias_direct"
        assert r.url_confidence == "high"

    # Case 10: .co.uk domain — high confidence, email fallback to apex
    def test_case10_co_uk_domain(self, records):
        r = records["100010"]
        assert r.url_source == "gias_direct"
        assert r.url_confidence == "high"
        assert r.email_domain_confidence == "low"

    # Case 11: Junk domain — parses fine, gias_direct (liveness in session 2)
    def test_case11_junk_domain_parses(self, records):
        r = records["100011"]
        assert r.url_source == "gias_direct"

    # Case 12: Special school
    def test_case12_special_school(self, records):
        r = records["100012"]
        assert r.establishment_type == "Community special school"
        assert r.url_original is not None

    # Case 13: PRU
    def test_case13_pru(self, records):
        r = records["100013"]
        assert r.establishment_type == "Pupil referral unit"

    # Case 14: 16-19 institution
    def test_case14_16_19(self, records):
        r = records["100014"]
        assert r.establishment_type == "Free schools 16 to 19"

    # Case 15 (Closed) is filtered out before extract
    def test_case15_closed_not_extracted(self, records):
        assert "100015" not in records

    # Case 16 (independent, Open) is included by default scope
    def test_case16_independent_extracted_by_default(self, records):
        assert "100016" in records

    # Manual review schools have no URL
    def test_manual_review_schools_have_no_url(self, records):
        manual_urns = ["100006", "100007", "100028", "100029"]  # blank/na/none/see above
        for urn in manual_urns:
            if urn in records:
                assert records[urn].url_original is None, f"URN {urn} should have no URL"

    # LA-hosted schools have medium confidence
    def test_la_hosted_medium_confidence(self, records):
        r = records["100004"]
        assert r.url_confidence == "medium"
        assert "la_hosted" in r.flags


# ---------------------------------------------------------------------------
# Email domain tests
# ---------------------------------------------------------------------------

class TestEmailDomain:

    def _extract_single(self, website: str, name: str = "Test School") -> SchoolRecord:
        df = pd.DataFrame([{
            "URN": "888001",
            "EstablishmentName": name,
            "PhaseOfEducation (name)": "Primary",
            "TypeOfEstablishment (name)": "Community school",
            "EstablishmentStatus (name)": "Open",
            "LA (code)": "810",
            "LA (name)": "North Yorkshire",
            "Trusts (UID)": "",
            "Trusts (name)": "",
            "SchoolWebsite": website,
            "LastChangedDate": "01/09/2023",
        }])
        return extract(df, data_dir=DATA_DIR, run_id="test")[0]

    def test_own_sch_uk_high_confidence(self):
        r = self._extract_single("www.myschool.sch.uk")
        assert r.email_domain == "myschool.sch.uk"
        assert r.email_domain_confidence == "high"

    def test_la_sch_uk_namespace_high_confidence(self):
        # school.sandwell.sch.uk — apex ends in .sch.uk
        r = self._extract_single("www.myschool.sandwell.sch.uk")
        assert r.email_domain is not None
        assert r.email_domain.endswith(".sch.uk")
        assert r.email_domain_confidence == "high"

    def test_co_uk_falls_back_to_apex_low_confidence(self):
        r = self._extract_single("www.myschool.co.uk")
        assert r.email_domain == "myschool.co.uk"
        assert r.email_domain_confidence == "low"

    def test_missing_url_email_domain_is_none(self):
        r = self._extract_single("")
        assert r.email_domain is None
        assert "domain_missing" in r.flags
