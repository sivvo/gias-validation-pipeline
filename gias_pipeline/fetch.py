"""Download and cache the GIAS bulk data export CSV."""

from __future__ import annotations

import logging
import os
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
import yaml

logger = logging.getLogger(__name__)

_ENCODING = "windows-1252"
_CHUNK_SIZE = 1 << 20  # 1 MiB


def _load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as fh:
        return yaml.safe_load(fh)


def _url_for_date(template: str, d: date) -> str:
    return template.format(date=d.strftime("%Y%m%d"))


def _cached_path(cache_dir: Path, d: date) -> Path:
    return cache_dir / f"edubasealldata_{d.strftime('%Y%m%d')}.csv"


def _download(url: str, dest: Path) -> None:
    """Stream-download *url* to *dest*, writing atomically via a temp file."""
    tmp = dest.with_suffix(".tmp")
    logger.info("Downloading %s → %s", url, dest)
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=_CHUNK_SIZE):
                fh.write(chunk)
    tmp.rename(dest)
    logger.info("Saved %s (%.1f MiB)", dest, dest.stat().st_size / (1 << 20))


def fetch(config_path: str = "config.yaml") -> pd.DataFrame:
    """Return a DataFrame of all GIAS records.

    Downloads today's file if not already cached; falls back to yesterday's
    file if today's has not yet been published by GIAS.

    Parameters
    ----------
    config_path:
        Path to ``config.yaml``.

    Returns
    -------
    pd.DataFrame
        Raw GIAS data with columns stripped of leading/trailing whitespace.
    """
    cfg = _load_config(config_path)
    url_template: str = cfg["gias"]["download_url"]
    cache_locally: bool = cfg["gias"].get("cache_locally", True)
    cache_dir = Path(cfg["gias"].get("cache_path", "./cache"))

    if cache_locally:
        cache_dir.mkdir(parents=True, exist_ok=True)

    today = date.today()
    candidates = [today, today - timedelta(days=1)]

    csv_path: Path | None = None
    for d in candidates:
        dest = _cached_path(cache_dir, d)

        if cache_locally and dest.exists():
            logger.info("Using cached file %s", dest)
            csv_path = dest
            break

        url = _url_for_date(url_template, d)
        try:
            _download(url, dest)
            csv_path = dest
            break
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                logger.warning("File not yet published for %s, trying previous day", d)
                continue
            raise

    if csv_path is None:
        raise RuntimeError(
            "Could not download GIAS data for today or yesterday. "
            "Check network connectivity or the GIAS download URL."
        )

    logger.info("Reading CSV %s (encoding=%s)", csv_path, _ENCODING)
    df = pd.read_csv(csv_path, encoding=_ENCODING, dtype=str, low_memory=False)

    # Normalise column names: strip surrounding whitespace
    df.columns = [c.strip() for c in df.columns]

    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))
    return df
