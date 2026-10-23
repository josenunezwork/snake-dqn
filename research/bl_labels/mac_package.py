"""Prepare the ruled Mac F0 equivalence and held-D256 operational packages.

This module only writes immutable, outcome-blind controls and manifests.  It
does not run the factory, read label rows, or turn an F4 HOLD into admission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any

from .exports import _write_once, empty_ledger, publish_json

F1_BANK_SHA256 = "a8c1dc4bbe1fd579d59b69d6ca6ca9dbd2814b088f74277f008f504f6d1bbb0d"
F1_BANK_PATH = (
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/bl-labels-f0/f1-pack-10a/f1-root-bank.json.xz"
)
CANARY_A_CONTROL_SHA256 = "9ee31223d989b8fc023efa54ac01837c6332b28e91ba64196e526fd253d879bf"
F4_REGISTRATION_SHA256 = "66388c795b3bec5bd014fa03d700994527b67763479143f7c3e5f35d1bb2fd65"
F4_APPROVAL_SCHEMA = "bl-residual-f4-judgment/v1"
F4_JUDGMENT_SHA256 = "0fe1a103384035839cb143a6df9193abf8269197e6d02f44bf7dceb000ab8362"
F4_RECIPE_SHA256 = "49d12f486bffa455adcedbc318e88f1524f5c502894d0c1ace47daa84b9d3a5a"
F4_CODEX_V_REVIEW_SHA256 = "9f7f26efa36e10361a4b66bad00e3df2b1798f9db35d79bf1e1a6f6d8001e890"
CANARY_COUNT = 36
CANARY_WORKERS = 6
CANARY_WALL_SECONDS = 3 * 60 * 60
D256_WALL_SECONDS = 16 * 60 * 60
# The 16-hour W6 cap at the ruling's 8 rescue-bearing-lives/worker-hour floor
# can require 768 admitted parents.  The expected collection is about 320, but
# this finite pre-data window prevents an after-outcome refill when it is slow.
D256_COLLECTION_LIVES = 864


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _control(value: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    if value.get("schema") != "bl-labels-f0-control/v1":
        raise ValueError("Canary-B requires an F0 control")
    namespace, allocation = value.get("namespace"), value.get("allocation")
    if not isinstance(namespace, str) or not namespace or not isinstance(allocation, list):
        raise ValueError("Canary-B control lacks namespace/allocation")
    if len(allocation) != CANARY_COUNT:
        raise ValueError("Canary-B requires the exact frozen 36-allocation list")
    ids = [row.get("allocation_id") for row in allocation if isinstance(row, dict)]
    if (
        len(ids) != CANARY_COUNT
        or len(set(ids)) != CANARY_COUNT
        or any(not isinstance(x, str) for x in ids)
    ):
        raise ValueError("Canary-B allocation IDs are malformed")
    for row in allocation:
        if type(row.get("seed")) is not int or row.get("stream") not in {
            "gate-frozen",
            "gate-mixed",
            "gate-scripted",
            "watch6",
        }:
            raise ValueError("Canary-B allocation differs from the frozen F0 list")
    return namespace, allocation


def make_canary_control(
    canary_a_control: Path, *, expected_sha256: str = CANARY_A_CONTROL_SHA256
) -> dict[str, Any]:
    """Retain Canary-A's namespace, allocation IDs and seeds exactly for Canary-B."""
    if _sha(canary_a_control) != expected_sha256:
        raise ValueError("Canary-B control must byte-bind the frozen Canary-A control")
    source = json.loads(canary_a_control.read_text())
    namespace, allocation = _control(source)
    return {
        "schema": "bl-labels-f0-control/v1",
        "namespace": namespace,
        "allocation": allocation,
        "workers": CANARY_WORKERS,
        "platform": "darwin-arm64",
        "source_canary_a_control_sha256": expected_sha256,
        "purpose": "Canary-B registered Mac F0 admission look; no pooled Canary-A rows",
    }


def canary_manifest(control: dict[str, Any], root: Path) -> dict[str, Any]:
    """Describe the single F0 look without permitting a hidden retry or re-look."""
    namespace, allocation = _control(control)
    output = root / "canary-output-mac-01"
    state = root / "canary-state-mac-01"
    ledger = root / "verified-ledger.json"
    return {
        "schema": "bl-labels-f0-mac-canary/v1",
        "f1_bank_path": F1_BANK_PATH,
        "status": "READY_AFTER_F1_WORKER_INVARIANCE",
        "platform": "darwin-arm64",
        "namespace": namespace,
        "allocation": allocation,
        "allocation_sha256": hashlib.sha256(
            json.dumps(allocation, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "workers": CANARY_WORKERS,
        "nice": 10,
        "wall_cap_seconds": CANARY_WALL_SECONDS,
        "output": str(output),
        "state_dir": str(state),
        "ledger": str(ledger),
        "registered_look": {
            "count": 1,
            "read_after": "36/36 verified sessions and published verified ledger",
            "incomplete": (
                "one mechanical completion only inside the original 3h cap; otherwise HOLD"
            ),
            "integrity_failure": "INVALID; one mechanical fix then one re-canary",
            "rescue_floor_per_mac_worker_hour": 8,
            "denominator": (
                "6 P-core workers times cumulative full-attempt wall seconds, "
                "including recovery/resume overhead"
            ),
        },
        "runtime": {
            "module": "research.bl_labels.mac_driver",
            "argv": [
                "--control",
                str(root / "control.json"),
                "--ledger",
                str(ledger),
                "--output",
                str(output),
                "--state-dir",
                str(state),
                "--workers",
                "6",
                "--deadline-seconds",
                str(CANARY_WALL_SECONDS),
            ],
        },
        "preconditions": {
            "f1_bank_sha256": F1_BANK_SHA256,
            "f1_worker_schedules": ["one", "half", "all"],
            "f1_required_workers": [1, 3, 6],
            "f1_compare": "digest and label_sha256 identical per root across W1/W3/W6",
        },
        "provenance_note": (
            "full_digest includes host/session provenance (global RNG and runtime carry), so it is "
            "platform/session-local; the provenance-free state digest is the "
            "cross-runtime diagnostic."
        ),
    }


def d256_allocation(namespace: str) -> list[dict[str, Any]]:
    """Freeze the bounded pre-selection population before any rescue is known.

    The 96:768 Watch6:gate parent ratio preserves the D256 4:32 ratio, while
    the gate 3:3:2 split preserves frozen:mixed:scripted (12:12:8).  Selection
    remains an internal, registered life-selection rule; collection never
    grows after observing support.
    """
    rows: list[dict[str, Any]] = []
    for stream, count, horizon in (
        ("watch6", 96, 14400),
        ("gate-frozen", 288, 5000),
        ("gate-mixed", 288, 5000),
        ("gate-scripted", 192, 5000),
    ):
        for index in range(count):
            token = f"{namespace}/{stream}/{index}".encode()
            rows.append(
                {
                    "allocation_id": f"d256-{namespace.split('-')[-2][:8]}-{stream}-{index:03d}",
                    "seed": int.from_bytes(hashlib.sha256(token).digest()[:4], "big"),
                    "stream": stream,
                    "horizon": horizon,
                    "allocation_index": len(rows),
                }
            )
    assert len(rows) == D256_COLLECTION_LIVES
    assert len({row["seed"] for row in rows}) == len(rows)
    return rows


def d256_control(source_commit: str, *, materialize: bool = False) -> dict[str, Any]:
    """Return the exact finite production population without reading an outcome."""
    namespace = f"f0-hindsight-labels-{source_commit}-mac-d256"
    allocation = d256_allocation(namespace)
    if materialize:
        from .manifests import materialize_gate_plans

        allocation = materialize_gate_plans(allocation)
    return {
        "schema": "bl-labels-f0-control/v1",
        "namespace": namespace,
        "allocation": allocation,
        "workers": CANARY_WORKERS,
        "source_commit": source_commit,
        "platform": "darwin-arm64",
        "purpose": "F4 D256 finite pre-data collection population; held pending M-E/F0/F4 gates",
    }


def d256_seed_audit(allocation: list[dict[str, Any]], *, repo: Path) -> dict[str, Any]:
    """Outcome-blind seed scan that intentionally leaves closed Round 0 unread."""
    seeds = [str(row["seed"]) for row in allocation]
    if len(seeds) != len(set(seeds)):
        raise ValueError("D256 frozen allocation has an internal seed collision")
    hits: dict[str, list[str]] = {}
    for root in (repo, repo.parent / "snake-dqn-artifacts"):
        command = [
            "rg",
            "-I",
            "-l",
            "-F",
            "-w",
            "--no-config",
            "--glob",
            "!**/t34-round0-study-01/**",
        ]
        command.extend(part for seed in seeds for part in ("-e", seed))
        command.extend(("--", str(root)))
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=600)
        if result.returncode not in (0, 1):
            raise RuntimeError(f"D256 seed audit failed under {root}")
        hits[str(root)] = sorted(line for line in result.stdout.splitlines() if line)
    if any(hits.values()):
        raise ValueError("D256 fresh seed collision")
    return {
        "schema": "bl-labels-f0-d256-seed-audit/v1",
        "seeds": [int(x) for x in seeds],
        "hits_by_root": hits,
        "excluded_glob": "**/t34-round0-study-01/**",
        "outcome_blind": True,
    }


def d256_manifest(
    root: Path, *, source_commit: str, control: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Stage a Mac-only D256 HOLD that cannot become executable before both gates."""
    if len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit):
        raise ValueError("D256 manifest requires a full lowercase source commit")
    control = control or d256_control(source_commit)
    namespace, allocation = control["namespace"], control["allocation"]
    return {
        "schema": "bl-labels-f0-mac-d256-hold/v1",
        "status": "HOLD",
        "platform": "darwin-arm64",
        "namespace": namespace,
        "workers": CANARY_WORKERS,
        "nice": 10,
        "wall_cap_seconds": D256_WALL_SECONDS,
        "selection": {
            "collection_parents": D256_COLLECTION_LIVES,
            "expected_collection_parents": 320,
            "allocation_sha256": hashlib.sha256(
                json.dumps(allocation, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "dose": "D256",
            "fit_lives": 192,
            "reserve_lives": 64,
            "reserve_watch6": 16,
            "reserve_other": 48,
            "parent_reserve_cap": 4,
            "folds": [1, 2, 3, 4],
            "composition": (
                "freeze D256 S-Watch6:S-gate parent and gate-roster ratios; "
                "report realised shares"
            ),
            "support_shortfall": "INSUFFICIENT; no refill, resize, or cap raise",
        },
        "operational_cap": {"base_wall_hours": 8, "effective_wall_hours": 16, "multiplier": 2},
        "gates": {
            "mac_equivalence_receipt": (
                "required: F1 W1/W3/W6 identities and Canary-B COMPLETE/PASS"
            ),
            "f4_approval": {
                "schema": F4_APPROVAL_SCHEMA,
                "decision": "APPROVE",
                "judgment_sha256": F4_JUDGMENT_SHA256,
                "registration_sha256": F4_REGISTRATION_SHA256,
                "recipe_sha256": F4_RECIPE_SHA256,
            },
            "codex_v_review_sha256": F4_CODEX_V_REVIEW_SHA256,
            "forbidden_until_gates_clear": [
                "factory collection",
                "label read",
                "fit",
                "reserve look",
            ],
        },
        "driver": {
            "module": "research.bl_labels.mac_driver",
            "required_arguments": ["--control", "--ledger", "--b8", "--frp3", "--output"],
            "default_workers": 6,
        },
    }


def prepare(
    canary_a_control: Path,
    out: Path,
    *,
    source_commit: str,
    expected_canary_a_sha256: str = CANARY_A_CONTROL_SHA256,
    audit_seeds: bool = False,
) -> dict[str, Path]:
    """Create the fresh Mac package and its empty, verified resume ledger."""
    if out.exists():
        raise FileExistsError("Mac package path must be fresh")
    out.mkdir(parents=True)
    control = make_canary_control(canary_a_control, expected_sha256=expected_canary_a_sha256)
    control["mac_gate"] = {
        "stage": "canary-b",
        "status": "HOLD",
        "package_root": str(out.resolve()),
    }
    control["source_commit"] = source_commit
    control_path = out / "control.json"
    publish_json(control_path, control)
    ledger_path = out / "verified-ledger.json"
    publish_json(ledger_path, empty_ledger(control["namespace"], control["allocation"]))
    # Root bytes are fixed by the judgment.  The F1 ruler verifies the actual
    # compressed bank at launch before this otherwise-empty ledger can be used.
    from .f1_units import empty_ledger as empty_f1_ledger

    f1_ledger_path = out / "f1-verified-ledger.json"
    publish_json(
        f1_ledger_path, empty_f1_ledger(F1_BANK_SHA256, "mac", source_commit=source_commit)
    )
    manifest_path = out / "canary-b-manifest.json"
    publish_json(manifest_path, canary_manifest(control, out))
    d256 = d256_control(source_commit, materialize=True)
    if audit_seeds:
        publish_json(
            out / "d256-seed-audit.json",
            d256_seed_audit(d256["allocation"], repo=Path(__file__).resolve().parents[2]),
        )
    d256["mac_gate"] = {
        "stage": "d256",
        "status": "HOLD",
        "package_root": str(out.resolve()),
    }
    d256_path = out / "d256-mac-hold.json"
    publish_json(d256_path, d256_manifest(out, source_commit=source_commit, control=d256))
    d256_control_path = out / "d256-control.json"
    publish_json(d256_control_path, d256)
    d256_ledger_path = out / "d256-verified-ledger.json"
    publish_json(d256_ledger_path, empty_ledger(d256["namespace"], d256["allocation"]))
    write_commands(out, source_commit)
    return {
        "control": control_path,
        "ledger": ledger_path,
        "f1_ledger": f1_ledger_path,
        "canary": manifest_path,
        "d256": d256_path,
        "d256_control": d256_control_path,
        "d256_ledger": d256_ledger_path,
    }


def write_commands(out: Path, source_commit: str) -> None:
    """Publish exact synchronous operator commands for a tools/ops pinned worktree."""
    header = (
        "\n".join(
            [
                "#!/bin/bash",
                "set -euo pipefail",
                "PACK=" + shlex.quote(str(out.resolve())),
                "PIN=" + shlex.quote(source_commit),
                "PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python",
                "CKPT=/Users/josenunez/Projects/ml/snake-dqn/saved_snakes",
                "ART=/Users/josenunez/Projects/ml/snake-dqn-artifacts/bl-labels-f0",
                "B8=$CKPT/b8_s3_t7_20261016.pth",
                "FRP3=$CKPT/frp3_m3_s12_u60000_20261005.pth",
                "export F0_COMMIT=$PIN F1_COMMIT=$PIN SNAKE_DQN_DEVICE=cpu",
                "export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1",
                "export NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1",
            ]
        )
        + "\n"
    )
    me = header + r"""
"$PY" -m research.bl_labels.mac_driver guard-command --state-dir "$PACK/f1-state" --seconds \
3600 -- "$PY" -m research.bl_labels.f1_ruler run --bank \
"$ART/f1-pack-10a/f1-root-bank.json.xz" --ledger "$PACK/f1-verified-ledger.json" --output \
"$PACK/f1-output" --platform mac --deadline-seconds 3600 --b8 "$B8" --frp3 "$FRP3"
"$PY" -m research.bl_labels.mac_gates admit-canary --pack "$PACK" --f1-output \
"$PACK/f1-output" --out "$PACK/canary-admitted.json"
"$PY" -m research.bl_labels.mac_driver --check --control "$PACK/canary-admitted.json" \
--ledger "$PACK/verified-ledger.json" --b8 "$B8" --frp3 "$FRP3" --output \
"$PACK/canary-output-mac-01" --state-dir "$PACK/canary-state-mac-01" --workers 6 \
--deadline-seconds 10800
"$PY" -m research.bl_labels.mac_driver --control "$PACK/canary-admitted.json" --ledger \
"$PACK/verified-ledger.json" --b8 "$B8" --frp3 "$FRP3" --output "$PACK/canary-output-mac-01" \
--state-dir "$PACK/canary-state-mac-01" --workers 6 --deadline-seconds 10800
"$PY" -m research.bl_labels.mac_gates ledger --pack "$PACK" --attempt \
"$PACK/canary-output-mac-01" --out "$PACK/canary-verified/verified-ledger.json"
"$PY" -m research.bl_labels.mac_driver guard-command --state-dir "$PACK/replay-state" \
--seconds 3600 -- "$PY" -m research.bl_labels.mac_gates replay-canary-a --pack "$PACK" --out \
"$PACK/canary-a-replay-audit.json"
"$PY" -m research.bl_labels.mac_gates look-canary --pack "$PACK" --ledger \
"$PACK/canary-verified/verified-ledger.json" --attempt "$PACK/canary-output-mac-01" --out \
"$PACK/f0-look/decision.json"
"""
    d256 = header + r"""
APPROVAL=${1:?Pass the actual Claude-created F4 judgment.json path}
"$PY" -m research.bl_labels.mac_gates admit-d256 --pack "$PACK" --f1-output \
"$PACK/f1-output" --f0-look "$PACK/f0-look/decision.json" --f4-approval "$APPROVAL" --out \
"$PACK/d256-admitted.json"
"$PY" -m research.bl_labels.mac_driver --check --control "$PACK/d256-admitted.json" --ledger \
"$PACK/d256-verified-ledger.json" --b8 "$B8" --frp3 "$FRP3" --output \
"$PACK/d256-output-mac-01" --state-dir "$PACK/d256-state-mac-01" --workers 6 \
--deadline-seconds 57600
"$PY" -m research.bl_labels.mac_driver --control "$PACK/d256-admitted.json" --ledger \
"$PACK/d256-verified-ledger.json" --b8 "$B8" --frp3 "$FRP3" --output \
"$PACK/d256-output-mac-01" --state-dir "$PACK/d256-state-mac-01" --workers 6 \
--deadline-seconds 57600
"""
    for name, contents in (("RUN_ME.sh", me), ("RUN_D256.sh", d256)):
        path = out / name
        _write_once(path, contents.encode())
        digest = _sha(path)
        _write_once(Path(str(path) + ".sha256"), f"{digest}  {path.name}\n".encode())
        _write_once(Path(str(path) + ".complete"), b"complete\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canary-a-control", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--expected-canary-a-sha256", default=CANARY_A_CONTROL_SHA256)
    parser.add_argument("--audit-seeds", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            {
                key: str(value)
                for key, value in prepare(
                    args.canary_a_control,
                    args.out,
                    source_commit=args.commit,
                    expected_canary_a_sha256=args.expected_canary_a_sha256,
                    audit_seeds=args.audit_seeds,
                ).items()
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
