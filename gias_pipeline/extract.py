"""Extract and classify school URLs from filtered GIAS data.

Execution model (two phases):
  1. Classify  — iterate all rows, run the URL decision tree synchronously,
                 produce a _PartialRecord per school. No I/O.
  2. Finalise  — convert each _PartialRecord to a SchoolRecord.

Schools whose URL cannot be extracted (domain_missing, social_media_only,
parse_failed) are written directly to manual review — no DNS inference is
attempted.
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import pandas as pd

from .models import SchoolRecord

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_JUNK_PATTERNS: frozenset[str] = frozenset(
    {"n/a", "na", "none", "-", "tbc", "tba", "see above", "see website", ""}
)

_SOCIAL_DOMAINS: frozenset[str] = frozenset(
    {
        "facebook.com",
        "twitter.com",
        "x.com",
        "instagram.com",
        "linkedin.com",
        "youtube.com",
        "tiktok.com",
    }
)

# GIAS column names
_COL_URN = "URN"
_COL_NAME = "EstablishmentName"
_COL_PHASE = "PhaseOfEducation (name)"
_COL_TYPE = "TypeOfEstablishment (name)"
_COL_STATUS = "EstablishmentStatus (name)"
_COL_LA_CODE = "LA (code)"
_COL_LA_NAME = "LA (name)"
_COL_WEBSITE = "SchoolWebsite"
_COL_LAST_CHANGED = "LastChangedDate"


# ---------------------------------------------------------------------------
# Pure helpers (no I/O)
# ---------------------------------------------------------------------------

def _load_domain_list(path: Path) -> frozenset[str]:
    """Return a frozenset of lowercased, stripped domain strings from *path*."""
    if not path.exists():
        logger.warning("Domain list not found: %s", path)
        return frozenset()
    lines = path.read_text(encoding="utf-8").splitlines()
    return frozenset(
        line.strip().lower()
        for line in lines
        if line.strip() and not line.startswith("#")
    )


def _is_social(domain: str, social_domains: frozenset[str] = _SOCIAL_DOMAINS) -> bool:
    """Return True if *domain* (apex) matches a known social platform."""
    domain = domain.lower()
    return any(domain == s or domain.endswith("." + s) for s in social_domains)


def _ends_with_la_domain(domain: str, la_domains: frozenset[str]) -> bool:
    domain = domain.lower()
    return any(domain == la or domain.endswith("." + la) for la in la_domains)


def _parse_url(raw: str) -> Optional[tuple[str, str, str]]:
    """Parse *raw* into (fqdn, apex_domain, canonical_url) or return None."""
    raw = raw.strip()
    if "://" not in raw:
        raw = "https://" + raw
    try:
        parsed = urlparse(raw)
        fqdn = parsed.netloc.lower()
        if not fqdn:
            return None
        fqdn = fqdn.split(":")[0]  # strip port
        apex = fqdn[4:] if fqdn.startswith("www.") else fqdn
        canonical = f"https://{fqdn}"
        return fqdn, apex, canonical
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Intermediate record (phase 1 → phase 2)
# ---------------------------------------------------------------------------

@dataclass
class _PartialRecord:
    # Scalar fields
    urn: str
    name: str
    phase: str
    establishment_type: str
    status: str
    la_code: str
    la_name: str
    gias_last_updated: str
    pipeline_run_id: str

    # URL classification
    url_original: Optional[str]   # set for gias_direct; None for manual review
    url_source: str               # "gias_direct" | "manual"
    url_confidence: str           # "high" | "medium" | "low"
    apex: Optional[str]           # parsed apex domain; None for manual-review records
    flags: list[str] = field(default_factory=list)

    # Email domain (always resolved inline, no DNS)
    email_domain_direct: Optional[str] = None
    email_domain_confidence_direct: str = "low"


# ---------------------------------------------------------------------------
# Phase 1 — classify each row (pure, no I/O)
# ---------------------------------------------------------------------------

def _classify_row(
    row: pd.Series,
    la_domains: frozenset[str],
    social_domains: frozenset[str],
    run_id: str,
) -> _PartialRecord:
    """Build a _PartialRecord from one GIAS row. No DNS calls."""

    def col(name: str) -> str:
        val = row.get(name, "")
        return "" if pd.isna(val) else str(val).strip()

    urn = col(_COL_URN)
    name = col(_COL_NAME)
    raw_url = col(_COL_WEBSITE)
    raw_lower = raw_url.lower()

    base = dict(
        urn=urn,
        name=name,
        phase=col(_COL_PHASE),
        establishment_type=col(_COL_TYPE),
        status=col(_COL_STATUS),
        la_code=col(_COL_LA_CODE),
        la_name=col(_COL_LA_NAME),
        gias_last_updated=col(_COL_LAST_CHANGED),
        pipeline_run_id=run_id,
    )

    # ------------------------------------------------------------------
    # Helper: school goes straight to manual review with a given flag
    # ------------------------------------------------------------------
    def _manual(flag: str) -> _PartialRecord:
        return _PartialRecord(
            **base,
            url_original=None,
            url_source="manual",
            url_confidence="low",
            apex=None,
            flags=[flag],
            email_domain_direct=None,
            email_domain_confidence_direct="low",
        )

    if raw_lower in _JUNK_PATTERNS:
        return _manual("domain_missing")

    parsed = _parse_url(raw_url)

    if parsed is None:
        return _manual("parse_failed")

    fqdn, apex, _ = parsed

    if _is_social(apex, social_domains):
        return _manual("social_media_only")

    # ------------------------------------------------------------------
    # Usable URL from GIAS
    # ------------------------------------------------------------------
    flags: list[str] = []

    if _ends_with_la_domain(apex, la_domains):
        flags.append("la_hosted")
        url_confidence = "medium"
    else:
        url_confidence = "high"

    # Email domain — inline, no DNS
    if apex.endswith(".sch.uk"):
        email_domain_direct: Optional[str] = apex
        email_domain_confidence_direct = "high"
    else:
        # Best-guess fallback: use the apex domain directly
        email_domain_direct = apex
        email_domain_confidence_direct = "low"

    return _PartialRecord(
        **base,
        url_original=raw_url.strip(),
        url_source="gias_direct",
        url_confidence=url_confidence,
        apex=apex,
        flags=flags,
        email_domain_direct=email_domain_direct,
        email_domain_confidence_direct=email_domain_confidence_direct,
    )


# ---------------------------------------------------------------------------
# Phase 2 — produce final SchoolRecord from each _PartialRecord
# ---------------------------------------------------------------------------

def _finalise(partial: _PartialRecord) -> SchoolRecord:
    """Convert a _PartialRecord to a SchoolRecord."""
    return SchoolRecord(
        urn=partial.urn,
        name=partial.name,
        phase=partial.phase,
        establishment_type=partial.establishment_type,
        status=partial.status,
        la_code=partial.la_code,
        la_name=partial.la_name,
        url_original=partial.url_original,
        url_canonical=None,             # populated in session 2
        url_confidence=partial.url_confidence,
        url_source=partial.url_source,
        email_domain=partial.email_domain_direct,
        email_domain_confidence=partial.email_domain_confidence_direct,
        http_status=None,
        is_reachable=None,
        redirect_chain=[],
        flags=list(partial.flags),
        gias_last_updated=partial.gias_last_updated,
        pipeline_run_id=partial.pipeline_run_id,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract(
    active: pd.DataFrame,
    data_dir: Path = Path("./data"),
    run_id: Optional[str] = None,
) -> list[SchoolRecord]:
    """Extract a SchoolRecord for every row in *active*.

    Parameters
    ----------
    active:
        Filtered DataFrame from :func:`filter.filter_schools`.
    data_dir:
        Directory containing ``la_domains.txt`` and ``social_domains.txt``.
    run_id:
        Unique identifier for this pipeline run. Auto-generated if not given.

    Returns
    -------
    list[SchoolRecord]
    """
    if run_id is None:
        run_id = str(uuid.uuid4())

    la_domains = _load_domain_list(data_dir / "la_domains.txt")
    extra_social = _load_domain_list(data_dir / "social_domains.txt")
    social_domains = _SOCIAL_DOMAINS | extra_social

    total = len(active)
    logger.info("Extracting records for %d schools (run_id=%s)", total, run_id)
    logger.debug(
        "Loaded %d LA domains, %d social domains", len(la_domains), len(social_domains)
    )

    records: list[SchoolRecord] = []
    for i, (_, row) in enumerate(active.iterrows()):
        records.append(_finalise(_classify_row(row, la_domains, social_domains, run_id)))
        if (i + 1) % 1000 == 0:
            logger.info("Processed %d / %d schools...", i + 1, total)

    logger.info("Extraction complete: %d records", total)
    return records
