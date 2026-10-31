"""Mac-side binding of Astra's pre-data confirmation ruling for Watch6 V3.

Keeping the ruling receipt outside protocol.identity() lets Argon freeze and
dry-run-verify the exact canary and 24 shard manifests now, while mechanically
blocking live Colab dispatch until Astra issues RULING_V3.md and the operator
binds it via this module.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from research.watch6_search_efficacy_v3 import protocol as p
from tools.ops.package_spec import digest, publish_json

SCHEMA = "watch6-v3-pre-data-ruling/v1"
DEFAULT_RULING_JSON = (
    p.ROOT.parent / "snake-dqn-artifacts/t34-sol/argon-v3-confirm/astra-ruling.json"
)


def resolve_ruling_json(override: Path | None = None) -> Path:
    """Resolve the active Mac-side ruling receipt path."""
    if override is not None:
        return override.resolve()
    env = os.environ.get("WATCH6_V3_RULING_JSON")
    if env:
        return Path(env).resolve()
    return DEFAULT_RULING_JSON.resolve()


def record_ruling(ruling_md: Path, out: Path | None = None, *, judge: str = "Astra") -> dict:
    """Bind Astra's written pre-data ruling markdown to REGISTRATION_V3.md and science."""
    ruling_md = ruling_md.resolve(strict=True)
    if not ruling_md.is_file() or ruling_md.stat().st_size == 0:
        raise ValueError(f"ruling markdown must be a nonempty regular file: {ruling_md}")
    if judge not in ("Astra", "Fable"):
        raise ValueError("independent pre-data judge must be Astra (or Fable stand-in)")
    target = resolve_ruling_json(out)
    payload = {
        "schema": SCHEMA,
        "status": "RATIFIED",
        "judge": judge,
        "implementer": "Argon",
        "independent_implementation_review": True,
        "namespace": p.NAMESPACE,
        "science": p.identity(),
        "registration_path": str(p.registration()),
        "registration_sha256": digest(p.registration()),
        "ruling_path": str(ruling_md),
        "ruling_sha256": digest(ruling_md),
    }
    publish_json(target, payload)
    return payload


def validate_ruling_receipt(path: Path | None = None) -> dict:
    """Fail closed if Astra's pre-data ruling receipt is absent, stale, or mismatched."""
    target = resolve_ruling_json(path)
    if not target.is_file():
        raise ValueError(
            f"REFUSE: missing ratified Astra pre-data ruling receipt at {target} "
            "(run `python -m research.watch6_search_efficacy_v3.ratify --ruling <RULING_V3.md>`)"
        )
    receipt = json.loads(target.read_text())
    ruling_md = Path(receipt.get("ruling_path", ""))
    if (
        receipt.get("schema") != SCHEMA
        or receipt.get("status") != "RATIFIED"
        or receipt.get("judge") not in ("Astra", "Fable")
        or receipt.get("implementer") != "Argon"
        or receipt.get("namespace") != p.NAMESPACE
        or receipt.get("science") != p.identity()
        or receipt.get("registration_sha256") != digest(p.registration())
        or not ruling_md.is_absolute()
        or not ruling_md.is_file()
        or digest(ruling_md) != receipt.get("ruling_sha256")
    ):
        raise ValueError("REFUSE: Astra pre-data ruling receipt is stale, invalid, or mismatched")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ruling", type=Path, required=True, help="Path to Astra's RULING_V3.md")
    parser.add_argument("--out", type=Path, default=None, help="Output path for astra-ruling.json")
    parser.add_argument("--judge", default="Astra", choices=("Astra", "Fable"))
    args = parser.parse_args()
    receipt = record_ruling(args.ruling, args.out, judge=args.judge)
    validate_ruling_receipt(args.out)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
