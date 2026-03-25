"""Detect changes between the current pipeline run and the previous run.

Compares current records against the most recent previous schools.csv by URN.
Produces three categories:
  - added:   URNs in current but not previous
  - removed: URNs in previous but not current
  - changed: URNs in both but url_canonical changed

If no previous run exists, returns empty delta and logs a warning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .models import SchoolRecord

logger = logging.getLogger(__name__)


@dataclass
class DeltaResult:
    added: pd.DataFrame       # full records for new URNs
    removed: pd.DataFrame     # rows from previous run for removed URNs
    changed: pd.DataFrame     # rows with old + new url_canonical


def _find_previous_csv(output_dir: Path) -> Path | None:
    """Find the most recent schools.csv in output_dir or its subdirectories.

    Checks for a latest/ symlink first, then falls back to the file with the
    most recent modification time.
    """
    latest_link = output_dir / "latest" / "schools.csv"
    if latest_link.exists():
        return latest_link

    candidates = list(output_dir.glob("schools.csv"))
    candidates += list(output_dir.glob("*/schools.csv"))

    if not candidates:
        return None

    return max(candidates, key=lambda p: p.stat().st_mtime)


def _records_to_df(records: list[SchoolRecord]) -> pd.DataFrame:
    rows = [
        {
            "urn": r.urn,
            "name": r.name,
            "url_canonical": r.url_canonical or "",
        }
        for r in records
    ]
    return pd.DataFrame(rows).set_index("urn")


def compute_delta(
    records: list[SchoolRecord],
    output_dir: Path,
) -> DeltaResult:
    """Compare *records* to the previous run's schools.csv.

    Parameters
    ----------
    records:
        Current run's SchoolRecord list.
    output_dir:
        Directory to search for previous schools.csv.

    Returns
    -------
    DeltaResult
        DataFrames for added, removed, and changed records.
    """
    empty_added = pd.DataFrame(
        columns=["urn", "name", "url_canonical"]
    )
    empty_removed = pd.DataFrame(
        columns=["urn", "name", "url_canonical"]
    )
    empty_changed = pd.DataFrame(
        columns=["urn", "name", "old_url_canonical", "new_url_canonical"]
    )

    prev_path = _find_previous_csv(output_dir)
    if prev_path is None:
        logger.warning(
            "No previous schools.csv found in %s — first run, generating empty delta files",
            output_dir,
        )
        return DeltaResult(
            added=empty_added,
            removed=empty_removed,
            changed=empty_changed,
        )

    logger.info("Loading previous run from %s", prev_path)
    try:
        prev_df = pd.read_csv(prev_path, dtype=str, low_memory=False)
    except Exception as exc:
        logger.warning("Could not read previous schools.csv: %s — skipping delta", exc)
        return DeltaResult(
            added=empty_added,
            removed=empty_removed,
            changed=empty_changed,
        )

    if "urn" not in prev_df.columns:
        logger.warning("Previous schools.csv has no 'urn' column — skipping delta")
        return DeltaResult(
            added=empty_added,
            removed=empty_removed,
            changed=empty_changed,
        )

    prev_df = prev_df.fillna("").set_index("urn")

    curr_df = _records_to_df(records)

    prev_urns = set(prev_df.index)
    curr_urns = set(curr_df.index)

    # Added
    added_urns = curr_urns - prev_urns
    added_df = curr_df.loc[list(added_urns)].reset_index() if added_urns else empty_added

    # Removed
    removed_urns = prev_urns - curr_urns
    prev_cols = [c for c in ["name", "url_canonical", "mat_uid"] if c in prev_df.columns]
    if removed_urns:
        removed_df = prev_df.loc[list(removed_urns), prev_cols].reset_index()
    else:
        removed_df = empty_removed

    # Changed — URNs in both runs
    common_urns = curr_urns & prev_urns
    changed_rows = []
    prev_url_col = "url_canonical" if "url_canonical" in prev_df.columns else None

    for urn in common_urns:
        curr_row = curr_df.loc[urn]
        prev_row = prev_df.loc[urn]

        old_url = prev_row[prev_url_col] if prev_url_col else ""
        new_url = curr_row["url_canonical"]

        if old_url != new_url:
            changed_rows.append(
                {
                    "urn": urn,
                    "name": curr_row.get("name", ""),
                    "old_url_canonical": old_url,
                    "new_url_canonical": new_url,
                }
            )

    changed_df = pd.DataFrame(changed_rows) if changed_rows else empty_changed

    logger.info(
        "Delta: added=%d, removed=%d, changed=%d",
        len(added_df),
        len(removed_df),
        len(changed_df),
    )
    return DeltaResult(added=added_df, removed=removed_df, changed=changed_df)
