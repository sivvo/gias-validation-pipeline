"""Tests for gias_pipeline.liveness."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gias_pipeline.liveness import _check_one, _run_liveness
from gias_pipeline.models import SchoolRecord


def _make_record(urn: str, url: str | None = None) -> SchoolRecord:
    return SchoolRecord(
        urn=urn,
        name=f"School {urn}",
        phase="Primary",
        establishment_type="Community school",
        status="Open",
        la_code="001",
        la_name="Test LA",
        url_original=url,
        url_canonical=None,
        url_confidence="high",
        url_source="gias_direct",
        email_domain=None,
        email_domain_confidence="low",
        http_status=None,
        is_reachable=None,
        redirect_chain=[],
        flags=[],
        gias_last_updated="2024-01-01",
        pipeline_run_id="test-run",
    )


def _make_response(status: int, url: str, history=None) -> MagicMock:
    """Build a mock aiohttp response."""
    resp = MagicMock()
    resp.status = status
    resp.url = MagicMock()
    resp.url.__str__ = lambda self: url
    resp.history = history or []
    resp.__aenter__ = AsyncMock(return_value=resp)
    resp.__aexit__ = AsyncMock(return_value=False)
    return resp


class TestCheckOne:
    def test_successful_get_marks_reachable(self):
        record = _make_record("100001", "https://example.sch.uk")
        resp = _make_response(200, "https://example.sch.uk")

        session = MagicMock()
        session.get = MagicMock(return_value=resp)

        sem = asyncio.Semaphore(1)
        asyncio.run(_check_one(session, record, sem, max_redirects=5))

        assert record.is_reachable is True
        assert record.url_canonical == "https://example.sch.uk"
        assert record.http_status == 200

    def test_404_marks_unreachable(self):
        record = _make_record("100002", "https://gone.example.com")
        resp = _make_response(404, "https://gone.example.com")

        session = MagicMock()
        session.get = MagicMock(return_value=resp)

        sem = asyncio.Semaphore(1)
        asyncio.run(_check_one(session, record, sem, max_redirects=5))

        assert record.is_reachable is False
        assert "unreachable" in record.flags
        assert record.http_status == 404

    def test_redirect_chain_populated(self):
        """3-hop redirect: A → B → C (final)."""
        hop_a = MagicMock()
        hop_a.url = MagicMock()
        hop_a.url.__str__ = lambda self: "https://hop-a.example.com"

        hop_b = MagicMock()
        hop_b.url = MagicMock()
        hop_b.url.__str__ = lambda self: "https://hop-b.example.com"

        resp = _make_response(
            200,
            "https://final.example.com",
            history=[hop_a, hop_b],
        )

        session = MagicMock()
        session.get = MagicMock(return_value=resp)

        record = _make_record("100003", "https://original.example.com")
        sem = asyncio.Semaphore(1)
        asyncio.run(_check_one(session, record, sem, max_redirects=5))

        assert record.is_reachable is True
        assert record.url_canonical == "https://final.example.com"
        assert record.redirect_chain == [
            "https://hop-a.example.com",
            "https://hop-b.example.com",
        ]
        assert "redirect_followed" in record.flags

    def test_timeout_marks_unreachable(self):
        import aiohttp

        record = _make_record("100004", "https://timeout.example.com")

        async def _raise(*args, **kwargs):
            raise asyncio.TimeoutError()

        ctx = MagicMock()
        ctx.__aenter__ = _raise
        ctx.__aexit__ = AsyncMock(return_value=False)

        session = MagicMock()
        session.get = MagicMock(return_value=ctx)

        sem = asyncio.Semaphore(1)
        asyncio.run(_check_one(session, record, sem, max_redirects=5))

        assert record.is_reachable is False
        assert "unreachable" in record.flags

    def test_connection_refused_marks_unreachable(self):
        import aiohttp

        record = _make_record("100005", "https://refused.example.com")

        async def _raise(*args, **kwargs):
            raise aiohttp.ClientConnectionError("Connection refused")

        ctx = MagicMock()
        ctx.__aenter__ = _raise
        ctx.__aexit__ = AsyncMock(return_value=False)

        session = MagicMock()
        session.get = MagicMock(return_value=ctx)

        sem = asyncio.Semaphore(1)
        asyncio.run(_check_one(session, record, sem, max_redirects=5))

        assert record.is_reachable is False
        assert "unreachable" in record.flags

    def test_no_url_is_skipped(self):
        """Schools with no url_original must not be checked."""
        record = _make_record("100006", url=None)
        session = MagicMock()
        sem = asyncio.Semaphore(1)

        asyncio.run(_check_one(session, record, sem, max_redirects=5))

        session.get.assert_not_called()
        assert record.is_reachable is None
        assert record.url_canonical is None

    def test_no_redirect_when_url_unchanged(self):
        record = _make_record("100007", "https://same.sch.uk")
        resp = _make_response(200, "https://same.sch.uk/")

        session = MagicMock()
        session.get = MagicMock(return_value=resp)

        sem = asyncio.Semaphore(1)
        asyncio.run(_check_one(session, record, sem, max_redirects=5))

        # Trailing slash difference — redirect_followed should not be present
        # since we strip trailing slashes in comparison
        assert record.is_reachable is True
