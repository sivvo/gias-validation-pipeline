"""Filter the raw GIAS DataFrame to in-scope schools.

Scope is controlled by four boolean flags (all read from config.yaml's
``scope`` section; defaults are shown below):

  include_closed      (default: false) — include Closed establishments
  include_independent (default: true)  — include independent schools
  include_nursery     (default: true)  — include nursery / early-years schools
  include_fe          (default: true)  — include further-education providers

  We may want to exclude types of records not relevant to our analysis, such as:
    - Schools that have closed
    - Independent schools
    - Nursery schools
    - Further education providers
Because I wasn't sure at time of writing, i've made these all be config configurable
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Establishment-type groups
# ---------------------------------------------------------------------------

TYPES_MAINSTREAM: frozenset[str] = frozenset(
    {
        # Mainstream
        "Community school",
        "Voluntary aided school",
        "Voluntary controlled school",
        "Foundation school",
        "Academy converter",
        "Academy sponsor led",
        "Free schools",
        "Studio schools",
        "University technical college",
        # Special / AP / PRU
        "Community special school",
        "Foundation special school",
        "Academy special converter",
        "Academy special sponsor led",
        "Free schools special",
        "Pupil referral unit",
        "Academy alternative provision converter",
        "Academy alternative provision sponsor led",
        "Free schools alternative provision",
        # 16-19 (school-based)
        "Free schools 16 to 19",
    }
)

TYPES_INDEPENDENT: frozenset[str] = frozenset(
    {
        "Other independent school",
        "Other independent special school",
        "Non-maintained special school",
    }
)

TYPES_NURSERY: frozenset[str] = frozenset(
    {
        "Local authority nursery school",
    }
)

TYPES_FE: frozenset[str] = frozenset(
    {
        "Further education",
        "Sixth form centres",
        "Higher education institutions",
        "Special post 16 institution",
    }
)

# ---------------------------------------------------------------------------
# Scope defaults
# ---------------------------------------------------------------------------

_SCOPE_DEFAULTS: dict = {
    "include_closed": False,
    "include_independent": True,
    "include_nursery": True,
    "include_fe": True,
}

_COL_TYPE = "TypeOfEstablishment (name)"
_COL_STATUS = "EstablishmentStatus (name)"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_type_set(scope: dict) -> frozenset[str]:
    types: set[str] = set(TYPES_MAINSTREAM)
    if scope.get("include_independent", _SCOPE_DEFAULTS["include_independent"]):
        types |= TYPES_INDEPENDENT
    if scope.get("include_nursery", _SCOPE_DEFAULTS["include_nursery"]):
        types |= TYPES_NURSERY
    if scope.get("include_fe", _SCOPE_DEFAULTS["include_fe"]):
        types |= TYPES_FE
    return frozenset(types)


def _build_status_set(scope: dict) -> frozenset[str]:
    statuses: set[str] = {"Open"}
    if scope.get("include_closed", _SCOPE_DEFAULTS["include_closed"]):
        statuses.add("Closed")
    return frozenset(statuses)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class FilterResult(NamedTuple):
    active: pd.DataFrame
    excluded: pd.DataFrame


def filter_schools(df: pd.DataFrame, scope: dict | None = None) -> FilterResult:
    """Split *df* into in-scope establishments and everything else.

    Parameters
    ----------
    df:
        Raw GIAS DataFrame as returned by :func:`fetch.fetch`.
    scope:
        Dict of scope flags (from ``config.yaml``'s ``scope`` section).
        Missing keys fall back to :data:`_SCOPE_DEFAULTS`.

    Returns
    -------
    FilterResult
        ``active`` contains establishments matching the scope.
        ``excluded`` retains all other rows for reference / audit.
    """
    if _COL_TYPE not in df.columns:
        raise ValueError(
            f"Expected column '{_COL_TYPE}' not found. "
            f"Available columns: {list(df.columns)}"
        )
    if _COL_STATUS not in df.columns:
        raise ValueError(
            f"Expected column '{_COL_STATUS}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    scope = scope or {}
    type_set = _build_type_set(scope)
    status_set = _build_status_set(scope)

    type_series = df[_COL_TYPE].fillna("").str.strip()
    status_series = df[_COL_STATUS].fillna("").str.strip()

    mask_active = status_series.isin(status_set) & type_series.isin(type_set)

    active = df[mask_active].copy().reset_index(drop=True)
    excluded = df[~mask_active].copy().reset_index(drop=True)

    logger.info(
        "Filter result: %d active, %d excluded (total %d) "
        "[closed=%s, independent=%s, nursery=%s, fe=%s]",
        len(active),
        len(excluded),
        len(df),
        scope.get("include_closed", _SCOPE_DEFAULTS["include_closed"]),
        scope.get("include_independent", _SCOPE_DEFAULTS["include_independent"]),
        scope.get("include_nursery", _SCOPE_DEFAULTS["include_nursery"]),
        scope.get("include_fe", _SCOPE_DEFAULTS["include_fe"]),
    )

    return FilterResult(active=active, excluded=excluded)
