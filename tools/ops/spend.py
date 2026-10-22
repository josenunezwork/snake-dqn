#!/usr/bin/env python3
"""Report verified prior Colab spend from immutable supervisor receipts.

Only the receipt-level elapsed duration and the supervisor's own priced rate are
used.  Provider balance deltas and aggregate accounting fields are deliberately
not substituted for either value.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


class ReceiptRefused(ValueError):
    """Raised when a receipt cannot prove an unambiguous own-spend value."""


@dataclass(frozen=True)
class SpendRow:
    """The independently auditable charge extracted from one receipt."""

    path: Path
    sha256: str
    elapsed_seconds: float
    own_rate_units_per_hour: float
    units: float
    identity: tuple[str, str] | None


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReceiptRefused(f"{label} must be a finite numeric value")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ReceiptRefused(f"{label} must be a finite non-negative numeric value")
    return number


def _load_receipt(path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ReceiptRefused(f"cannot read receipt {path}: {error}") from error
    try:
        receipt = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ReceiptRefused(f"invalid JSON receipt {path}: {error.msg}") from error
    if not isinstance(receipt, dict):
        raise ReceiptRefused(f"receipt {path} must be a JSON object")
    return receipt, hashlib.sha256(raw).hexdigest()


def read_spend(path: Path) -> SpendRow:
    """Read one receipt and calculate its own allocation charge."""
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ReceiptRefused(f"cannot resolve receipt {path}: {error}") from error
    receipt, digest = _load_receipt(resolved)
    accounting = receipt.get("accounting")
    if not isinstance(accounting, dict):
        raise ReceiptRefused(f"receipt {resolved} is missing object accounting")
    elapsed = _finite_number(receipt.get("elapsed_seconds"), f"receipt {resolved} elapsed_seconds")
    rate = _finite_number(
        accounting.get("own_rate_units_per_hour"),
        f"receipt {resolved} accounting.own_rate_units_per_hour",
    )
    generation, session = receipt.get("generation"), receipt.get("session")
    if generation is None and session is None:
        identity = None
    elif isinstance(generation, str) and generation and isinstance(session, str) and session:
        identity = (generation, session)
    else:
        raise ReceiptRefused(f"receipt {resolved} has incomplete generation/session identity")
    units = elapsed * rate / 3600
    if not math.isfinite(units):
        raise ReceiptRefused(f"receipt {resolved} own-spend calculation is not finite")
    return SpendRow(resolved, digest, elapsed, rate, units, identity)


def verify_receipts(paths: Sequence[Path]) -> list[SpendRow]:
    """Load receipts once each, refusing duplicate files or job identities."""
    rows: list[SpendRow] = []
    seen_paths: set[Path] = set()
    seen_identities: set[tuple[str, str]] = set()
    for path in paths:
        row = read_spend(path)
        if row.path in seen_paths:
            raise ReceiptRefused(f"duplicate receipt path: {row.path}")
        if row.identity is not None and row.identity in seen_identities:
            raise ReceiptRefused(
                "duplicate receipt generation/session identity: "
                f"generation={row.identity[0]!r} session={row.identity[1]!r}"
            )
        seen_paths.add(row.path)
        if row.identity is not None:
            seen_identities.add(row.identity)
        rows.append(row)
    return rows


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipts", metavar="RECEIPT", nargs="+", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        rows = verify_receipts(args.receipts)
    except ReceiptRefused as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 2
    total = sum(row.units for row in rows)
    if not math.isfinite(total):
        print("REFUSED: total own-spend calculation is not finite", file=sys.stderr)
        return 2
    for row in rows:
        print(
            f"receipt={row.path} sha256={row.sha256} elapsed_seconds={row.elapsed_seconds:.6f} "
            f"own_rate_units_per_hour={row.own_rate_units_per_hour:.6f} units={row.units:.9f}"
        )
    print(f"TOTAL receipts={len(rows)} units={total:.9f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
