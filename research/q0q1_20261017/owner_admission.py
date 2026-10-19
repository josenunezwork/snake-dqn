"""Bind owner admission to the complete registration and its latest amendment."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Mapping

OWNER_RELEASE = "APPROVED_Q0Q1_20261017"
SCHEMA = "q0q1-owner-admission/v1"
REGISTRATION_PATH = "research/q0q1_20261017/PREREGISTRATION.md"
REGISTRATION = Path(__file__).with_name("PREREGISTRATION.md")
DEFAULT_ADMISSION = Path(__file__).with_name("OWNER_ADMISSION_20261018_A5.json")


def latest_amendment(registration: Path = REGISTRATION) -> str:
    """Read numbered amendments, excluding later erratum headings."""
    amendments = re.findall(r"^## Amendment (A\d+)\b", registration.read_text(), re.MULTILINE)
    if not amendments:
        raise ValueError("registration has no numbered amendment")
    return amendments[-1]


def validate(admission: Mapping[str, Any], registration: Path = REGISTRATION) -> None:
    """Reject stale digests and amendment labels at generation and execution."""
    amendment = latest_amendment(registration)
    required = {
        "schema": SCHEMA,
        "status": "OWNER_ADMITTED",
        "owner_release": OWNER_RELEASE,
        "amendment": amendment,
        "registration_path": REGISTRATION_PATH,
        "registration_sha256": hashlib.sha256(registration.read_bytes()).hexdigest(),
    }
    if any(admission.get(name) != value for name, value in required.items()):
        raise ValueError(f"owner admission does not bind the approved {amendment} registration")
