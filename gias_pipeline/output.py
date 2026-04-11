"""Write all output files for a completed pipeline run.

Files produced (all to config output.path):
  schools.csv           — all records, full schema
  email_domains.txt     — deduplicated, sorted email domains
  urls_canonical.txt    — canonical URLs for reachable schools
  urls_original.txt     — original URLs for all schools with a URL
  unreachable.csv       — records where is_reachable=False
  manual_review.csv     — records with domain_missing or social_media_only flag
  la_hosted.csv         — records with la_hosted flag
  delta_added.csv       — from delta step
  delta_removed.csv     — from delta step
  delta_changed.csv     — from delta step
  run_summary.json      — run statistics
"""

from __future__ import annotations

import json
import logging
import pandas as pd
from collections import Counter
from pathlib import Path
from typing import Optional
from .delta import DeltaResult
from .models import SchoolRecord

logger = logging.getLogger(__name__)

# Column order for schools.csv
_SCHOOLS_COLUMNS = [
    "urn",
    "name",
    "phase",
    "establishment_type",
    "la_code",
    "la_name",
    "url_original",
    "url_canonical",
    "url_confidence",
    "url_source",
    "is_reachable",
    "http_status",
    "email_domain",
    "email_domain_confidence",
    "flags",
    "gias_last_updated",
    "pipeline_run_id",
]

_MANUAL_FLAGS = frozenset({"domain_missing", "social_media_only", "parse_failed"})

def _records_to_df(records: list[SchoolRecord]) -> pd.DataFrame:
    rows = []
    for r in records:
        rows.append(
            {
                "urn": r.urn,
                "name": r.name,
                "phase": r.phase,
                "establishment_type": r.establishment_type,
                "la_code": r.la_code,
                "la_name": r.la_name,
                "url_original": r.url_original or "",
                "url_canonical": r.url_canonical or "",
                "url_confidence": r.url_confidence,
                "url_source": r.url_source,
                "is_reachable": "" if r.is_reachable is None else str(r.is_reachable),
                "http_status": "" if r.http_status is None else str(r.http_status),
                "email_domain": r.email_domain or "",
                "email_domain_confidence": r.email_domain_confidence,
                "flags": "|".join(r.flags),
                "gias_last_updated": r.gias_last_updated,
                "pipeline_run_id": r.pipeline_run_id,
            }
        )
    df = pd.DataFrame(rows)
    # Ensure column order and presence
    for col in _SCHOOLS_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[_SCHOOLS_COLUMNS]


def _has_flag(flags_str: str, flag: str) -> bool:
    return flag in flags_str.split("|") if flags_str else False


def _has_any_flag(flags_str: str, flag_set: frozenset[str]) -> bool:
    parts = set(flags_str.split("|")) if flags_str else set()
    return bool(parts & flag_set)


def write_outputs(
    records: list[SchoolRecord],
    delta: DeltaResult,
    output_dir: Path,
    run_summary: dict,
) -> None:
    """Write all output files to *output_dir*.

    Parameters
    ----------
    records:
        Full list of SchoolRecord objects (post-enrich).
    delta:
        DeltaResult from delta step.
    output_dir:
        Destination directory (must exist).
    run_summary:
        Dictionary matching run_summary.json schema; runtime_seconds and
        delta counts will be filled in here.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    df = _records_to_df(records)

    schools_path = output_dir / "schools.csv"
    df.to_csv(schools_path, index=False)
    logger.info("Wrote %s (%d rows)", schools_path, len(df))

    email_domains = sorted(
        {r.email_domain for r in records if r.email_domain}
    )
    email_path = output_dir / "email_domains.txt"
    email_path.write_text("\n".join(email_domains) + "\n", encoding="utf-8")
    logger.info("Wrote %s (%d domains)", email_path, len(email_domains))

    canonical_urls = sorted(
        {r.url_canonical for r in records if r.is_reachable is True and r.url_canonical}
    )
    canonical_path = output_dir / "urls_canonical.txt"
    canonical_path.write_text("\n".join(canonical_urls) + "\n", encoding="utf-8")
    logger.info("Wrote %s (%d URLs)", canonical_path, len(canonical_urls))

    original_urls = sorted(
        {r.url_original for r in records if r.url_original}
    )
    original_path = output_dir / "urls_original.txt"
    original_path.write_text("\n".join(original_urls) + "\n", encoding="utf-8")
    logger.info("Wrote %s (%d URLs)", original_path, len(original_urls))

    unreachable_df = df[df["is_reachable"] == "False"]
    unreachable_path = output_dir / "unreachable.csv"
    unreachable_df.to_csv(unreachable_path, index=False)
    logger.info("Wrote %s (%d rows)", unreachable_path, len(unreachable_df))

    manual_mask = df["flags"].apply(
        lambda f: _has_any_flag(f, _MANUAL_FLAGS)
    )
    manual_df = df[manual_mask]
    manual_path = output_dir / "manual_review.csv"
    manual_df.to_csv(manual_path, index=False)
    logger.info("Wrote %s (%d rows)", manual_path, len(manual_df))

    la_mask = df["flags"].apply(lambda f: _has_flag(f, "la_hosted"))
    la_df = df[la_mask]
    la_path = output_dir / "la_hosted.csv"
    la_df.to_csv(la_path, index=False)
    logger.info("Wrote %s (%d rows)", la_path, len(la_df))

    delta_added_path = output_dir / "delta_added.csv"
    delta.added.to_csv(delta_added_path, index=False)
    logger.info("Wrote %s (%d rows)", delta_added_path, len(delta.added))

    delta_removed_path = output_dir / "delta_removed.csv"
    delta.removed.to_csv(delta_removed_path, index=False)
    logger.info("Wrote %s (%d rows)", delta_removed_path, len(delta.removed))

    delta_changed_path = output_dir / "delta_changed.csv"
    delta.changed.to_csv(delta_changed_path, index=False)
    logger.info("Wrote %s (%d rows)", delta_changed_path, len(delta.changed))

    flag_counts: Counter[str] = Counter(f for r in records for f in r.flags)
    confidence_counts: Counter[str] = Counter(r.url_confidence for r in records)

    run_summary.update(
        {
            "total_processed": len(records),
            "reachable": sum(1 for r in records if r.is_reachable is True),
            "unreachable": sum(1 for r in records if r.is_reachable is False),
            "no_url": sum(1 for r in records if not r.url_original),
            "url_confidence": {
                "high": confidence_counts.get("high", 0),
                "medium": confidence_counts.get("medium", 0),
                "low": confidence_counts.get("low", 0),
            },
            "flags": dict(flag_counts),
            "delta": {
                "added": len(delta.added),
                "removed": len(delta.removed),
                "changed": len(delta.changed),
            },
        }
    )

    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(
        json.dumps(run_summary, indent=2, default=str), encoding="utf-8"
    )
    logger.info("Wrote %s", summary_path)
