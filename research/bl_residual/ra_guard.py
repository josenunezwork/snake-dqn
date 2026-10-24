"""Shared metadata-only admission guards for the single frozen F4 R-A look."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from research.bl_ppo.exports import digest, verify_export
from research.bl_residual import f2
from research.bl_residual.protocol import F4_REGISTRATION_SHA, allocation_hash, canonical_sha

ROOT = Path(__file__).resolve().parents[2]
# Deliberately independent of command-line inputs, packets, outputs and worktrees.
CLAIM_ROOT = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/f4/r-a-claims")
D4096_CONTROL = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/bl-labels-f0-speed/"
    "d4096-old-equiv-02-pack-01/d4096-control.json"
)
D4096_SCIENCE = "111c47818a097c57451386ef4f8a7009616f8f77"
OLD_COLLECTOR = "61900af28932e47e22e332840fc4406559adb062"
START_DEADLINE = datetime.fromisoformat("2026-10-25T06:00:00-07:00").timestamp()
CUTOFF = datetime.fromisoformat("2026-10-25T18:00:00-07:00").timestamp()
SOURCE_FILES = tuple(
    sorted(
        set(f2.SOURCE_FILES)
        | {str(path.relative_to(ROOT)) for path in (ROOT / "research/bl_residual").glob("*.py")}
    )
)


def now_epoch() -> float:
    return datetime.now(timezone.utc).timestamp()


def reserve_identity(
    d256_control: Path, d1024_control: Path, d256_allocation: Path, d1024_allocation: Path
) -> dict[str, Any]:
    """Verify and carry the existing controls' ordered list-of-life-ID digest."""
    identities = []
    for dose, control_path, allocation_path in (
        (256, d256_control, d256_allocation),
        (1024, d1024_control, d1024_allocation),
    ):
        verify_export(control_path)
        verify_export(allocation_path)
        control = json.loads(control_path.read_text())
        allocation = json.loads(allocation_path.read_text())
        lives = allocation["doses"][str(dose)]["heldout_lives"]
        if (
            not isinstance(lives, list)
            or len(lives) != 64
            or len(set(lives)) != 64
            or any(not isinstance(life, str) or not life for life in lives)
            or control.get("schema") != "bl-residual-control/v1"
            or control.get("synthetic") is not False
            or control.get("doses") != [dose]
            or control.get("series_id") != control.get("namespace")
            or not control.get("series_id")
            or control.get("registration_sha256") != F4_REGISTRATION_SHA
            or allocation_hash(control) != control.get("allocation")
            or control.get("input_hashes", {}).get("life-allocation.json")
            != digest(allocation_path)
            or control.get("reserve_allocation_sha256") != canonical_sha(lives)
        ):
            raise ValueError("frozen control reserve identity differs from its allocation")
        catalog = {row["life_id"]: row for row in allocation["catalog"]}
        ancestry = [
            {
                "life_id": life,
                "parent_world": catalog[life]["parent_world"],
                "allocation_index": catalog[life]["allocation_index"],
            }
            for life in sorted(lives)
        ]
        identities.append(
            {
                "series_id": control["series_id"],
                "reserve_allocation_sha256": control["reserve_allocation_sha256"],
                "reserve_ancestry_sha256": canonical_sha(ancestry),
                "reserve_life_ids": sorted(lives),
            }
        )
    if identities[0] != identities[1]:
        raise ValueError("D256/D1024 frozen reserve identities differ")
    return identities[0]


def claim_key(reserve: dict[str, Any]) -> str:
    return canonical_sha(
        {
            "series_id": reserve["series_id"],
            "registration_sha256": F4_REGISTRATION_SHA,
            "reserve_allocation_sha256": reserve["reserve_allocation_sha256"],
        }
    )


def claim_path(reserve: dict[str, Any]) -> Path:
    return CLAIM_ROOT / f"F4_R_A_{claim_key(reserve)}.lock.json"


def lock_identity(reserve: dict, left: dict, right: dict) -> dict:
    return {
        "schema": "bl-residual-r-a-lock/v2",
        "lock_root": str(CLAIM_ROOT),
        "reserve": reserve,
        "d256": left,
        "d1024": right,
    }


def validate_claim(claim: dict, path: Path, preflight: dict) -> None:
    if (
        claim.get("schema") != "bl-residual-r-a-lock/v2"
        or preflight not in (claim.get("d256"), claim.get("d1024"))
        or claim.get("lock_root") != str(CLAIM_ROOT)
        or path != claim_path(claim.get("reserve", {}))
        or path.parent.resolve() != CLAIM_ROOT.resolve()
    ):
        raise ValueError("R-A claim does not bind this immutable fit at the authoritative location")


def source_identity(pin: str) -> dict[str, str]:
    """Bind the actual executing source to a published full commit."""
    if len(pin) != 40 or any(c not in "0123456789abcdef" for c in pin):
        raise ValueError("R-A source pin must be a full published commit")
    if f2._git(ROOT, "rev-parse", "HEAD").decode().strip() != pin:
        raise ValueError("R-A source pin differs from executing HEAD")
    f2._git(ROOT, "merge-base", "--is-ancestor", pin, "origin/main")
    result = {}
    for name in SOURCE_FILES:
        payload = (ROOT / name).read_bytes()
        if payload != f2._git(ROOT, "show", f"{pin}:{name}"):
            raise ValueError("R-A source differs from its source pin")
        result[name] = digest(ROOT / name)
    return result


def receipt_binding(receipt: Path) -> dict[str, str]:
    """Verify F2 metadata without changing or reading the frozen statistic."""
    from research.bl_residual.github_authority import published_receipt_binding

    return published_receipt_binding(receipt)


def publication_evidence(binding: dict | None) -> dict | None:
    """Require sealed, remotely revalidated server publication evidence."""
    from research.bl_residual.github_authority import publication_evidence as verify

    return None if binding is None else verify(binding)


def trusted_time() -> dict[str, Any]:
    from research.bl_residual.github_authority import trusted_time as observe

    return observe()


def authoritative_starts() -> dict:
    """Read only the admitted D4096 control and all its attempt-start metadata.

    The canonical pack/state root is fixed here, never selected by the caller.
    Absence is bound too: a pack/start appearing after preparation invalidates
    that packet. No runtime results, rates, labels or lease files are opened.
    """
    state = D4096_CONTROL.parent / "d4096-state-mac-01"
    admitted_path = D4096_CONTROL.with_name("d4096-admitted.json")
    paths = sorted(state.glob("attempt-*.start.json"))
    if state.is_symlink():
        raise ValueError("D4096 timing state root is not authoritative")
    if not D4096_CONTROL.exists():
        if paths or admitted_path.exists():
            raise ValueError("D4096 timing authority lacks its original control")
        return {"control": None, "admitted": None, "starts": []}
    sha = verify_export(D4096_CONTROL)
    original = json.loads(D4096_CONTROL.read_text())
    original_gate = original.get("mac_gate", {})
    if (
        original_gate.get("stage") != "d4096"
        or original_gate.get("status") != "HOLD"
        or original_gate.get("source_commit") != D4096_SCIENCE
        or original.get("source_commit") != D4096_SCIENCE
        or not original.get("namespace")
        or not isinstance(original.get("allocation"), list)
    ):
        raise ValueError("R-A timing requires the admitted OLD D4096 control")
    authority = {
        "control": {"path": str(D4096_CONTROL), "sha256": sha},
        "admitted": None,
        "starts": [],
    }
    if not admitted_path.exists():
        if paths:
            raise ValueError("D4096 timing start has no admitted control")
        return authority
    admitted_sha = verify_export(admitted_path)
    control = json.loads(admitted_path.read_text())
    gate = control.get("mac_gate", {})
    if (
        gate.get("stage") != "d4096"
        or gate.get("status") != "APPROVED"
        or gate.get("source_commit") != D4096_SCIENCE
        or control.get("source_commit") != D4096_SCIENCE
        or gate.get("deadline_seconds") != 468000
        or gate.get("package_root") != str(D4096_CONTROL.parent.resolve())
        or gate.get("state_dir") != str(state.resolve())
        or gate.get("original_control_path") != str(D4096_CONTROL.resolve())
        or gate.get("original_control_sha256") != sha
        or gate.get("collector_source_commit") != OLD_COLLECTOR
        or gate.get("bindings", {}).get("speed_equivalence", {}).get("status")
        != "FAIL_OLD_FALLBACK"
        or control.get("namespace") != original["namespace"]
        or control.get("allocation") != original["allocation"]
    ):
        raise ValueError("D4096 admitted timing control differs from its frozen original")
    authority["admitted"] = {"path": str(admitted_path), "sha256": admitted_sha}
    starts = []
    for path in paths:
        receipt_sha = verify_export(path)
        value = json.loads(path.read_text())
        epoch = value.get("started_epoch")
        if (
            value.get("schema") != "bl-labels-f0-mac-attempt-start/v1"
            or value.get("funding_state") != "PENDING_SEALED"
            or value.get("judgment_sha256") != f2.JUDGMENT_SHA
            or type(epoch) not in (int, float)
            or not math.isfinite(epoch)
            or epoch <= 0
            or epoch > now_epoch()
            or value.get("namespace") != control["namespace"]
            or value.get("allocation_sha256") != canonical_sha(control["allocation"])
            or value.get("control_sha256") != canonical_sha(control)
            or value.get("source_commit") != control["source_commit"]
        ):
            raise ValueError("R-A timing requires a sealed D4096 attempt-start receipt")
        starts.append(
            {
                "path": str(path),
                "sha256": receipt_sha,
                "started_epoch": epoch,
                "namespace": value["namespace"],
                "allocation_sha256": value["allocation_sha256"],
            }
        )
    return {**authority, "starts": starts}


def timing_identity(start_receipt: Path | None, publication: dict | None) -> dict:
    from research.bl_residual.github_authority import TIME_POLICY

    authority = authoritative_starts()
    starts = authority["starts"]
    start = min(starts, key=lambda row: (row["started_epoch"], row["path"])) if starts else None
    if start_receipt is not None and (start is None or str(start_receipt) != start["path"]):
        raise ValueError("R-A timing must bind the earliest authoritative D4096 start")
    return {
        "judgment_sha256": f2.JUDGMENT_SHA,
        "time_authority": TIME_POLICY,
        "start_deadline_epoch": START_DEADLINE,
        "cutoff_epoch": CUTOFF,
        "d4096_start": start,
        "start_authority": authority,
        "branch": (
            "WAIT_FOR_F2_OR_CUTOFF"
            if start and start["started_epoch"] <= START_DEADLINE
            else "READY_WITHOUT_WAIT"
        ),
        "f2_publication": publication,
    }


def timing_eligible(timing: dict, *, at: float) -> bool:
    """Enforce the wait and strict publication boundaries before claiming R-A."""
    if not math.isfinite(at) or (
        timing["d4096_start"] and timing["d4096_start"]["started_epoch"] > at
    ):
        raise ValueError("HOLD: unverifiable time or future D4096 start")
    pub = timing["f2_publication"]
    published = pub is not None and pub["published_epoch"] < min(CUTOFF, at)
    if timing["branch"] == "WAIT_FOR_F2_OR_CUTOFF" and at < CUTOFF and not published:
        raise ValueError("HOLD: R-A waits for verified F2 publication or the fixed cutoff")
    return published
