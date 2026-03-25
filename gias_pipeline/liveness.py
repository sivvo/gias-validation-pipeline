"""Async HTTP liveness checking for school URLs.

For each SchoolRecord with url_original set, attempts an HTTP GET and records:
  - url_canonical  — final URL after redirects
  - is_reachable   — True/False
  - http_status    — response status code
  - redirect_chain — list of intermediate URLs

Schools without url_original (domain_missing, social_media_only) are skipped.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import aiohttp
import yaml

from .models import SchoolRecord

logger = logging.getLogger(__name__)


def _load_liveness_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)
    liveness = cfg.get("liveness", {})
    return {
        "concurrency": liveness.get("concurrency", 50),
        "timeout_seconds": liveness.get("timeout_seconds", 5),
        "max_redirects": liveness.get("max_redirects", 5),
        "user_agent": liveness.get("user_agent", "DfE-CyberPipeline/1.0"),
    }


async def _check_one(
    session: aiohttp.ClientSession,
    record: SchoolRecord,
    semaphore: asyncio.Semaphore,
    max_redirects: int,
) -> None:
    """Perform liveness check for a single record, mutating it in place."""
    if not record.url_original:
        return

    url = record.url_original
    if "://" not in url:
        url = "https://" + url

    async with semaphore:
        try:
            redirect_chain: list[str] = []

            async with session.get(url, allow_redirects=True, max_redirects=max_redirects) as resp:
                # Collect redirect history
                for hist_resp in resp.history:
                    redirect_chain.append(str(hist_resp.url))

                final_url = str(resp.url)
                record.http_status = resp.status
                record.redirect_chain = redirect_chain

                if resp.status < 400:
                    record.is_reachable = True
                    record.url_canonical = final_url
                    if final_url.rstrip("/") != url.rstrip("/"):
                        if "redirect_followed" not in record.flags:
                            record.flags.append("redirect_followed")
                else:
                    record.is_reachable = False
                    if "unreachable" not in record.flags:
                        record.flags.append("unreachable")

        except asyncio.TimeoutError:
            record.is_reachable = False
            if "unreachable" not in record.flags:
                record.flags.append("unreachable")

        except aiohttp.ClientError:
            record.is_reachable = False
            if "unreachable" not in record.flags:
                record.flags.append("unreachable")

        except Exception as exc:
            logger.debug("Unexpected error checking %s: %s", url, exc)
            record.is_reachable = False
            if "unreachable" not in record.flags:
                record.flags.append("unreachable")


async def _run_liveness(
    records: list[SchoolRecord],
    concurrency: int,
    timeout_seconds: float,
    max_redirects: int,
    user_agent: str,
) -> None:
    """Run liveness checks on all records concurrently."""
    semaphore = asyncio.Semaphore(concurrency)
    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    connector = aiohttp.TCPConnector(ssl=False)
    headers = {"User-Agent": user_agent}

    total = len(records)
    checked = 0

    async with aiohttp.ClientSession(
        timeout=timeout,
        connector=connector,
        headers=headers,
    ) as session:
        tasks = []
        for record in records:
            if record.url_original:
                tasks.append(_check_one(session, record, semaphore, max_redirects))

        # Process with progress logging
        batch_size = 1000
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i : i + batch_size]
            await asyncio.gather(*batch, return_exceptions=True)
            checked += len(batch)
            logger.info(
                "Liveness: checked %d / %d schools...", checked, len(tasks)
            )

    skipped = total - len(tasks)
    if skipped:
        logger.info("Liveness: skipped %d schools with no URL", skipped)


def check_liveness(
    records: list[SchoolRecord],
    config_path: str = "config.yaml",
) -> list[SchoolRecord]:
    """Run HTTP liveness checks against all records with a url_original.

    Parameters
    ----------
    records:
        List of SchoolRecord objects from the extract step.
    config_path:
        Path to config.yaml.

    Returns
    -------
    list[SchoolRecord]
        The same list, with liveness fields populated in place.
    """
    cfg = _load_liveness_config(config_path)
    logger.info(
        "Starting liveness checks: concurrency=%d, timeout=%ss, max_redirects=%d",
        cfg["concurrency"],
        cfg["timeout_seconds"],
        cfg["max_redirects"],
    )

    asyncio.run(
        _run_liveness(
            records,
            concurrency=cfg["concurrency"],
            timeout_seconds=cfg["timeout_seconds"],
            max_redirects=cfg["max_redirects"],
            user_agent=cfg["user_agent"],
        )
    )

    reachable = sum(1 for r in records if r.is_reachable is True)
    unreachable = sum(1 for r in records if r.is_reachable is False)
    logger.info(
        "Liveness complete: reachable=%d, unreachable=%d", reachable, unreachable
    )
    return records
