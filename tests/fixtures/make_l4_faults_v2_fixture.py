"""Regenerate the v2 §2.4 projection of acceptance-1792762502 (both arms).

Keeps only call/transfer identity, timing and status fields plus terminal
fields; exception argv/error text is dropped except the terminal error string.
Run: python tests/fixtures/make_l4_faults_v2_fixture.py <acceptance-root> <out>
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

CALL_KEYS = (
    "attempt",
    "command",
    "elapsed_seconds",
    "status",
    "timeout_seconds",
    "failure_class",
    "request_id",
    "request_sha256",
)
CONTENT_KEYS = ("direction", "status", "remote", "elapsed_seconds")
TRANSFER_KEYS = ("direction", "remote", "status")
TERMINAL_KEYS = ("job_id", "status", "error", "failure_class", "stop_verified", "cleanup_error")


def project(path: Path) -> dict:
    raw = path.read_bytes()
    receipt = json.loads(raw)
    return {
        "source_receipt": str(path),
        "source_receipt_sha256": hashlib.sha256(raw).hexdigest(),
        **{key: receipt.get(key) for key in TERMINAL_KEYS},
        "cli_calls": [{k: c[k] for k in CALL_KEYS if k in c} for c in receipt["cli_calls"]],
        "contents_transfers": [
            {k: row.get(k) for k in CONTENT_KEYS} for row in receipt.get("contents_transfers", [])
        ],
        "transfers": [
            {k: row.get(k) for k in TRANSFER_KEYS} for row in receipt.get("transfers", [])
        ],
    }


def main() -> None:
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    value = {
        "schema": "l4-label-faults-v2-sanity-fixture/v1",
        "acceptance": root.name,
        "arms": {arm: project(root / "outputs" / arm / "receipt.json") for arm in ("l4", "cpu")},
    }
    out.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
