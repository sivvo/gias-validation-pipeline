"""Enrich SchoolRecords with refined email domains.

Derives email_domain from url_canonical (post-liveness) or url_original
(fallback for unreachable schools).  No DNS lookups; entirely offline.
"""

from __future__ import annotations
from typing import Optional
from urllib.parse import urlparse
from .models import SchoolRecord
import logging

logger = logging.getLogger(__name__)

# Confidence bands for email domains (no DNS — purely from URL)
# TODO think about other TLDs in use - it really is a free for all in the school sector 
_HIGH_TLDS = frozenset({".sch.uk"})
_MEDIUM_TLDS = frozenset({".ac.uk", ".org.uk", ".co.uk"}) # ac.uk is as reliable as sch.uk it just doesn't apply much for schools

def _apex_from_url(url: Optional[str]) -> Optional[str]:
    """Extract apex domain from a URL string."""
    if not url:
        return None
    if "://" not in url:
        url = "https://" + url
    try:
        parsed = urlparse(url)
        fqdn = parsed.netloc.lower().split(":")[0]
        if not fqdn:
            return None
        return fqdn[4:] if fqdn.startswith("www.") else fqdn
    except Exception:
        return None

def _email_confidence(apex: str) -> str:
    """ high, medium, or low based on the apex TLD."""
    for tld in _HIGH_TLDS:
        if apex.endswith(tld):
            return "high"
    for tld in _MEDIUM_TLDS:
        if apex.endswith(tld):
            return "medium"
    return "low"


def _refine_email_domain(record: SchoolRecord) -> None:
    """Update email_domain and email_domain_confidence on *record* in place.

    Rules (no DNS, no network):
      - url_canonical apex ends in .sch.uk              -> high
      - url_canonical apex ends in .ac.uk/.org.uk/.co.uk -> medium
      - url_canonical apex (any other TLD)              -> low
      - is_reachable=False, url_original has apex       -> low + email_domain_unverified
      - no URL at all                                   -> email_domain = None
    """
    canonical_apex = _apex_from_url(record.url_canonical)

    if canonical_apex:
        record.email_domain = canonical_apex
        record.email_domain_confidence = _email_confidence(canonical_apex)
    elif record.is_reachable is False:
        original_apex = _apex_from_url(record.url_original)
        if original_apex:
            record.email_domain = original_apex
            record.email_domain_confidence = "low"
            if "email_domain_unverified" not in record.flags:
                record.flags.append("email_domain_unverified")
        else:
            record.email_domain = None
            record.email_domain_confidence = "low"
    else:
        record.email_domain = None
        record.email_domain_confidence = "low"

def enrich(records: list[SchoolRecord]) -> list[SchoolRecord]:
    """Refine email domains for all records (no DNS, no network).

    Parameters
    ----------
    records:
        List of SchoolRecord objects (post-liveness).

    Returns
    -------
    list[SchoolRecord]
        The same list with email_domain and email_domain_confidence updated.
    """
    logger.info("Refining email domains")
    for record in records:
        _refine_email_domain(record)
    logger.info("Email domain refinement complete")
    return records
