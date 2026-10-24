"""Publish a funded D4096 union only after the sealed early-collection gates."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from . import d4096_sealed, mac_driver, mac_gates

ROOT = Path(__file__).resolve().parents[2]


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=30, check=False
    )


def require_pinned_source(commit: str) -> str:
    resolved = _git("rev-parse", "--verify", f"{commit}^{{commit}}")
    if resolved.returncode:
        raise ValueError("HOLD: union publisher code commit does not resolve")
    full = resolved.stdout.strip()
    if _git("rev-parse", "HEAD").stdout.strip() != full:
        raise ValueError("HOLD: union publisher must run at the pinned code commit")
    if _git("status", "--porcelain").stdout.strip():
        raise ValueError("HOLD: union publisher requires a clean checkout")
    if _git("merge-base", "--is-ancestor", full, "origin/main").returncode:
        raise ValueError("HOLD: pinned union-publisher code is not published on origin/main")
    return full


def _sealed_paths(pack: Path, control: dict) -> tuple[Path, Path, Path | None]:
    sealed = control.get("mac_gate", {}).get("bindings", {}).get("sealed_collection", {})
    if not isinstance(sealed, dict):
        raise ValueError("HOLD: D4096 sealed-collection bindings are absent")
    readiness, projection = sealed.get("readiness"), sealed.get("projection")
    if not isinstance(readiness, dict) or not isinstance(projection, dict):
        raise ValueError("HOLD: D4096 sealed-collection bindings are malformed")
    if not isinstance(readiness.get("path"), str) or not isinstance(projection.get("path"), str):
        raise ValueError("HOLD: D4096 sealed-collection receipt paths are malformed")
    discard = sealed.get("discard")
    if discard is not None and not isinstance(discard, str):
        raise ValueError("HOLD: D4096 discard binding is malformed")
    return (
        Path(readiness["path"]),
        Path(projection["path"]),
        Path(discard) if discard else pack / "d4096-not-run-by-rule.json",
    )


def publish_final(pack: Path, out: Path, code_commit: str, funding_receipt: Path) -> dict:
    """Create the final ledger only for a complete, funded, undiscarded D4096."""
    code_commit = require_pinned_source(code_commit)
    if out.exists() or Path(str(out) + ".complete").exists():
        raise FileExistsError("final D4096 union is create-only; verify the existing one")
    from . import d4096_resume

    control = d4096_resume.original_control(pack)
    readiness, projection, discard = _sealed_paths(pack, control)
    d4096_sealed.validate_readiness(readiness)
    d4096_sealed.validate_projection(projection, readiness)
    d4096_sealed.require_not_discarded(discard)
    d4096_sealed.require_funding_receipt(funding_receipt)
    attempts, elapsed = d4096_resume.attempt_chain(pack, control, final=True)
    value = mac_gates.publish_ledger(
        pack, attempts, out, control=control, funding_receipt=funding_receipt
    )
    d4096_resume.verify_union(out, control, attempts)
    _, _, pending = mac_driver._prepare_jobs(control, out)
    if pending:
        raise ValueError("final D4096 union leaves allocated parents pending; HOLD")
    return {
        "ledger": str(out),
        "sha256": mac_gates._verify(out),
        "attempts": len(attempts),
        "units": len(value["units"]),
        "cumulative_elapsed_seconds": elapsed,
        "pending_parents": 0,
        "code_commit": code_commit,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--funding-receipt", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    print(json.dumps(publish_final(args.pack, args.out, args.code_commit, args.funding_receipt)))


if __name__ == "__main__":
    main()
