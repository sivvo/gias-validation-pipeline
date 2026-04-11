#!/usr/bin/env python3
"""GIAS Pipeline — full pipeline entry point.
The plan here is to pull data from GIAS, process it, and then feed it into an EASM

Steps:
  1. fetch         — download / load cached GIAS CSV
  2. filter        — filter for open, in-scope schools
  3. extract       — classify URLs, build SchoolRecord list
  4. liveness      — async HTTP reachability check (--skip-liveness to skip), needed because of DQ issues in GIAS
  5. enrich        — email domain refinement
  6. delta         — compare to previous run
  7. output        — write all output files
"""

from __future__ import annotations

import argparse
import datetime
import logging
import sys
import time
import uuid
import yaml
from collections import Counter
from pathlib import Path

from gias_pipeline.delta import compute_delta
from gias_pipeline.enrich import enrich
from gias_pipeline.extract import extract
from gias_pipeline.fetch import fetch
from gias_pipeline.filter import filter_schools
from gias_pipeline.liveness import check_liveness
from gias_pipeline.models import SchoolRecord
from gias_pipeline.output import write_outputs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

def _load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as fh:
        return yaml.safe_load(fh)

def _print_summary(records: list[SchoolRecord], runtime_seconds: float) -> None:
    total = len(records)
    confidence_counts: Counter[str] = Counter(r.url_confidence for r in records)
    flag_counts: Counter[str] = Counter(f for r in records for f in r.flags)

    high = confidence_counts.get("high", 0)
    medium = confidence_counts.get("medium", 0)
    low = confidence_counts.get("low", 0)
    reachable = sum(1 for r in records if r.is_reachable is True)
    unreachable = sum(1 for r in records if r.is_reachable is False)
    no_url = sum(1 for r in records if not r.url_original)

    pct = lambda n: f"{n / total * 100:.1f}%" if total else "0%"

    print()
    print("=" * 60)
    print("GIAS Pipeline — Run Summary")
    print("=" * 60)
    print(f"Total schools processed:          {total}")
    print(f"Reachable:                        {reachable} ({pct(reachable)})")
    print(f"Unreachable:                      {unreachable} ({pct(unreachable)})")
    print(f"No URL:                           {no_url} ({pct(no_url)})")
    print()
    print("URL confidence:")
    print(f"  High:                           {high} ({pct(high)})")
    print(f"  Medium:                         {medium} ({pct(medium)})")
    print(f"  Low:                            {low} ({pct(low)})")
    print()
    print("Flags breakdown:")
    if flag_counts:
        for flag, count in sorted(flag_counts.items(), key=lambda x: -x[1]):
            print(f"  {flag}: {count}")
    else:
        print("  (none)")
    print()
    print(f"Total wall-clock time:            {runtime_seconds:.1f}s")
    print("=" * 60)
    print()

def main(config_path: str = "config.yaml", skip_liveness: bool = False) -> None:
    cfg = _load_config(config_path)
    output_dir = Path(cfg["output"]["path"])
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path("./data")

    run_id = str(uuid.uuid4())
    run_at = datetime.datetime.now(datetime.UTC)
    wall_start = time.monotonic()

    logger.info(f"Pipeline run started (run_id={run_id})")

    logger.info(f"Fetching GIAS data")
    df = fetch(config_path=config_path)

    gias_date = run_at.strftime("%Y%m%d")     # Capture the GIAS file date from the cached filename if possible
    # this works because GIAS uses a deterministic format for its name

    logger.info(f"Applying Filters")
    filter_result = filter_schools(df, scope=cfg.get("scope", {})) # apply filters - configured in config.yaml

    logger.info(f"Extracting URLs")
    records = extract(filter_result.active, data_dir=data_dir, run_id=run_id)

    # Liveness check - validate connectivity
    if skip_liveness:
        logger.info(f"Liveness check SKIPPED (--skip-liveness)")
    else:
        logger.info(f"Running liveness checks")
        records = check_liveness(records, config_path=config_path)

    logger.info("Parse email domains")
    records = enrich(records)

    # Delta - what changed this run to last
    # not currently used.. TODO remove this?
    logger.info(f"Computing delta")
    delta = compute_delta(records, output_dir=output_dir)

    logger.info(f"Writing output files to {output_dir}")
    runtime_seconds = time.monotonic() - wall_start

    run_summary = {
        "run_id": run_id,
        "run_at": run_at.isoformat() + "Z",
        "gias_date": gias_date,
        "runtime_seconds": round(runtime_seconds, 2),
    }

    write_outputs(
        records=records,
        delta=delta,
        output_dir=output_dir,
        run_summary=run_summary,
    )

    runtime_seconds = time.monotonic() - wall_start
    _print_summary(records, runtime_seconds)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GIAS school website pipeline"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    parser.add_argument(
        "--skip-liveness",
        action="store_true",
        help="Skip HTTP liveness checks (faster re-runs during development)",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    main(config_path=args.config, skip_liveness=args.skip_liveness)
