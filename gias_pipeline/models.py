"""Data models for the GIAS pipeline."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class SchoolRecord:
    urn: str
    name: str
    phase: str
    establishment_type: str
    status: str
    la_code: str
    la_name: str
    url_original: Optional[str]
    url_canonical: Optional[str]        # None until liveness check in session 2
    url_confidence: str                 # high / medium / low
    url_source: str                     # gias_direct / dns_inference / manual
    email_domain: Optional[str]
    email_domain_confidence: str
    http_status: Optional[int]          # None until session 2
    is_reachable: Optional[bool]        # None until session 2
    redirect_chain: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    gias_last_updated: str = ""
    pipeline_run_id: str = ""
