#!/usr/bin/env python3
"""Tier-1 screen (non-authoritative): Apex + v3 tail-aware veto vs Apex + v2 veto.

Pre-registration: ``protocol.md`` beside this file. The harness is
``research/apex_safety_20260926/dev_screen.py`` driven by :data:`SPEC`:

  A  champion + ``free-space-veto/v2-speed-preserving`` (rollout's built-in install;
     the configuration that got STRICT_PASS in apex-veto-strict-20260927 run-v3).
  B  champion + ``free-space-veto/v3-tail-aware`` (src/evaluation/safety_veto_v3.py,
     default options; v2-comparison diagnostics on, which never change actions).
  C  arm A repeated on the first 8 worlds per mix (determinism control).

The screen always holds a shared CPU slot lock and refuses to start on battery.
After ``dev_screen.main`` finishes, this wrapper self-checks every record and writes
``receipt.json`` (governance Tier-1 artifact). Nothing here changes a default.

Usage (about 1.5-2 h at the pre-registered size):
  SNAKE_DQN_DEVICE=cpu ./venv/bin/python research/apex_veto_v3_screen_20261001/screen.py \\
    --out /path/to/new/dir --deadline-utc 2026-10-02T06:00:00+00:00
Smoke (plumbing only, separate smoke namespace, <= 2 episodes x 500 frames):
  ... screen.py --out /tmp/x --deadline-utc ... --smoke-frames 500 --worlds-per-mix 1 \\
    --determinism-worlds 0 --smoke-mixes scripted
"""

from __future__ import annotations

import os

os.environ.setdefault("SNAKE_DQN_DEVICE", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import json  # noqa: E402
import sys  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List, Mapping, Sequence  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.apex_safety_20260926 import dev_screen  # noqa: E402

HERE = Path(__file__).resolve().parent
SCREEN_ID = "apex-veto-v3-screen-v1"
SCHEMA = "apex-veto-v3-screen/v1"
AUTHORITY = "screen (non-authoritative)"
DOMAIN = SCREEN_ID
SMOKE_DOMAIN = "apex-veto-v3-screen-smoke-v1"
NAMESPACE = "worlds"
RECEIPT_SCHEMA = "apex-veto-v3-screen-receipt/v1"
V2_METHOD = "free-space-veto/v2-speed-preserving"
V3_METHOD = "free-space-veto/v3-tail-aware"
ARM_METHODS = {"A": V2_METHOD, "B": V3_METHOD, "C": V2_METHOD}
# Pre-stated wall-time cap (protocol.md "Compute cap and operations").
MAX_WALL_SECONDS = 3 * 3600
EXCLUSION_PREFIX = 1000  # seeds per earlier domain/purpose checked (>= any bank's size)
# Every earlier SHA-prefix domain of this recipe we know of (name -> purposes).
EARLIER_DOMAINS: Dict[str, Sequence[str]] = {
    dev_screen.SCREEN_DOMAIN: (dev_screen.SCREEN_NAMESPACE,),
    **{
        f"apex-veto-strict-{bank}-v{version}": ("worlds",)
        for bank in ("dev", "final", "serving")
        for version in (1, 2, 3)
    },
    "apex-veto-strict-smoke-v1": ("worlds",),
    # Concurrent web-serving lane (research/apex_veto_serving_20261001, 2026-10-01 recipe).
    "apex-veto-web-serving-v1": ("watch", "play", "parity"),
    "apex-veto-web-serving-smoke-v1": ("watch", "play", "parity"),
    SMOKE_DOMAIN: (NAMESPACE,),
}


def earlier_namespaces(prefix: int = EXCLUSION_PREFIX) -> Dict[str, List[int]]:
    """``domain/purpose[0:prefix]`` -> seeds, for the disjointness preflight."""
    return {
        f"{domain}/{purpose}[0:{prefix}]": [
            dev_screen.uint32_seed(domain, purpose, index) for index in range(prefix)
        ]
        for domain, purposes in EARLIER_DOMAINS.items()
        for purpose in purposes
    }


def install_v3(hero: Any) -> Any:
    """Arm B's installer: v3 with default options plus v2-comparison diagnostics."""
    from src.evaluation.safety_veto_v3 import install_tail_aware_veto

    return install_tail_aware_veto(hero, diagnostics=True)


def _spec(domain: str) -> dev_screen.ScreenSpec:
    builtin = dev_screen.BUILTIN_VETO
    return dev_screen.ScreenSpec(
        name=SCREEN_ID,
        schema=SCHEMA,
        authority=AUTHORITY,
        domain=domain,
        namespace=NAMESPACE,
        protocol=HERE / "protocol.md",
        arm_vetoes={"A": builtin, "B": install_v3, "C": builtin},
        arm_descriptions={
            "A": f"champion + {V2_METHOD} (src/evaluation/safety_veto.py, built-in install)",
            "B": f"champion + {V3_METHOD} (src/evaluation/safety_veto_v3.py, defaults)",
            "C": "arm A repeated on the first determinism worlds per mix",
        },
        extra_namespaces={
            name: seeds
            for name, seeds in earlier_namespaces().items()
            if not name.startswith(f"{domain}/")
        },
        require_slot_locks=True,
        require_ac_power=True,
        owner="Apex safety lane",
        hypothesis=(
            "Replacing the v2 free-space veto with the v3 tail-aware veto raises the Apex "
            "champion's H5000 mass_integral (B - A > 0)."
        ),
        decision_informs=(
            "RECOMMEND_STRICT_GATE: design a Tier-2 strict gate for v3 against v2; "
            "NOT_ADVANCED: stop"
        ),
        primary_metric="paired mass_integral B-A per world per mix",
        estimator="one-sided paired t, Holm over 3 mixes, alpha 0.05, >=2 of 3",
        max_wall_seconds=MAX_WALL_SECONDS,
    )


SPEC = _spec(DOMAIN)
SMOKE_SPEC = _spec(SMOKE_DOMAIN)


def expected_descriptor(arm: str) -> Dict[str, Any]:
    """The probe identity each arm's records must carry (writer: ``veto.record()``)."""
    from src.evaluation.safety_veto import FreeSpaceVeto
    from src.evaluation.safety_veto_v3 import TailAwareFreeSpaceVeto

    return (TailAwareFreeSpaceVeto() if arm == "B" else FreeSpaceVeto()).descriptor()


# ---------------------------------------------------------------- self-check and receipt
# Each rule names its writer. Gating rules cover the decision path (arm identity,
# pairing/world identity, profile digest, denominators); diagnostics are reported only.


def check_record(
    record: Mapping[str, Any],
    arm: str,
    *,
    world_seed: int,
    mix: str,
    roster_hashes: Sequence[str],
    smoke: bool,
) -> List[str]:
    """Gating checks on one rollout record (the shape the strict producer writes)."""
    from src.evaluation.strict_promotion import (
        StrictPromotionArtifactError,
        _expected_world_identity,
        _validate_candidate_wrapper_probe,
    )

    failures: List[str] = []
    probes = record.get("probes") if isinstance(record, Mapping) else None
    probe = probes.get("safety_veto") if isinstance(probes, Mapping) else None
    # Writer: tournament_eval.rollout attaches ``safety_veto.record()``.
    try:
        _validate_candidate_wrapper_probe(probe, expected_descriptor(arm))
    except (StrictPromotionArtifactError, TypeError, KeyError) as exc:
        failures.append(f"safety_veto probe is not arm {arm}'s veto: {exc}")
    if record.get("seed") != world_seed:
        failures.append("record seed differs from the planned world seed")
    mass = record.get("mass_integral")
    if not isinstance(mass, (int, float)) or isinstance(mass, bool) or mass != mass:
        failures.append("mass_integral is not a finite number")
    if smoke:  # legacy (profile-free) smoke path: no profile, identity or denominators
        return failures
    # Writer: rollout(profile=...) via EvaluationMetricsAccumulator and the roster.
    if record.get("evaluation_profile_digest") != dev_screen.PROFILE_DIGEST:
        failures.append("evaluation_profile_digest is not promotion-v2-watch-rect")
    row = {
        "world_seed": world_seed,
        "mix": mix,
        "slots": [{"member_sha256": sha} for sha in roster_hashes],
    }
    if record.get("world_identity") != _expected_world_identity(row):
        failures.append("world_identity does not match the materialized roster")
    denominators = record.get("denominators") or {}
    if denominators.get("scored_frames") != dev_screen.HORIZON:
        failures.append("denominators.scored_frames is not the H5000 horizon")
    counters = (probe or {}).get("counters") or {}
    # One veto decision per frame the hero acted (AISnake.update -> veto.apply).
    if counters.get("decisions") != denominators.get("decision_frames"):
        failures.append("veto decisions differ from denominators.decision_frames")
    return failures


def check_entry(entry: Mapping[str, Any], smoke: bool) -> Dict[str, List[str]]:
    """Gating failures and reported warnings for one screen entry (dev_screen writer)."""
    failures: List[str] = []
    warnings: List[str] = []
    arm = entry.get("arm")
    if arm not in ARM_METHODS:
        return {"failures": [f"unknown arm {arm!r}"], "warnings": []}
    if entry.get("schema_version") != SCHEMA or entry.get("authority") != AUTHORITY:
        failures.append("entry schema/authority is not this screen's")
    if entry.get("screen") != SCREEN_ID or entry.get("hero_sha256") != dev_screen.CHAMPION[1]:
        failures.append("entry screen id or hero sha256 differs")
    if entry.get("safety_veto") is not True or entry.get("safety_veto_method") != ARM_METHODS[arm]:
        failures.append(f"entry veto flag/method is not arm {arm}'s")
    failures.extend(
        check_record(
            entry.get("record") or {},
            arm,
            world_seed=entry.get("world_seed"),
            mix=entry.get("mix"),
            roster_hashes=entry.get("roster_member_sha256s") or [],
            smoke=smoke,
        )
    )
    diagnostics = entry.get("veto_diagnostics")
    if arm == "B":
        counters = ((entry.get("record") or {}).get("probes") or {}).get("safety_veto") or {}
        decisions = (counters.get("counters") or {}).get("decisions")
        if not isinstance(diagnostics, Mapping) or diagnostics.get("decisions") != decisions:
            warnings.append("arm B diagnostics missing or not one per veto decision")
    elif diagnostics is not None:
        warnings.append(f"arm {arm} carries v3 diagnostics")
    return {"failures": failures, "warnings": warnings}


def self_check(out: Path, smoke: bool) -> Dict[str, Any]:
    """Check every saved record of a finished screen directory."""
    intent = json.loads((out / "intent.json").read_text())
    planned = {int(seed) for seed in intent["worlds"]["seeds"]}
    failures: List[str] = []
    warnings: List[str] = []
    paths = sorted((out / "records").glob("*.json"))
    for path in paths:
        entry = json.loads(path.read_text())
        name = f"{entry.get('arm')}-{entry.get('mix')}-{entry.get('world_seed')}.json"
        result = check_entry(entry, smoke)
        if path.name != name:
            result["failures"].append("file name differs from arm-mix-seed")
        if entry.get("world_seed") not in planned:
            result["failures"].append("world seed is not in the intent's planned seeds")
        failures.extend(f"{path.name}: {message}" for message in result["failures"])
        warnings.extend(f"{path.name}: {message}" for message in result["warnings"])
    return {
        "records": len(paths),
        "failures": failures,
        "warnings": warnings,
        "passes": not failures and bool(paths),
    }


SELF_DEATH_CAUSE = "self"  # probes.death_cause written by rollout for self-collision


def closing_pocket_report(entries: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Reported only: did v3 rescue the hero, or admit a pocket that then closed?

    Per arm-B entry (writers: ``TailAwareDiagnostics`` via ``veto_diagnostics``;
    ``probes.death_cause`` via rollout). ``admitted_closing_pocket``: the hero died
    by self-collision with a tail-admitted decision in its last
    ``CLOSING_POCKET_WINDOW`` decisions (the known growth gap, protocol.md).
    ``rescued``: tail-admitted decisions and no such death. Never gates anything.
    """
    from src.evaluation.safety_veto_v3 import CLOSING_POCKET_WINDOW

    rows: Dict[str, Dict[str, int]] = {}
    for entry in entries:
        if entry.get("arm") != "B":
            continue
        diag = entry.get("veto_diagnostics") or {}
        probes = (entry.get("record") or {}).get("probes") or {}
        row = rows.setdefault(
            str(entry.get("mix")),
            {
                "episodes": 0,
                "with_tail_admission": 0,
                "admitted_closing_pocket": 0,
                "rescued": 0,
                "self_collision_deaths": 0,
                "missing_diagnostics": 0,
            },
        )
        row["episodes"] += 1
        if "tail_admitted" not in diag:
            row["missing_diagnostics"] += 1
            continue
        self_death = probes.get("death_cause") == SELF_DEATH_CAUSE
        row["self_collision_deaths"] += int(self_death)
        if int(diag["tail_admitted"]) <= 0:
            continue
        row["with_tail_admission"] += 1
        if self_death and int(diag.get("tail_admitted_in_final_window", 0)) > 0:
            row["admitted_closing_pocket"] += 1
        else:
            row["rescued"] += 1
    total = {
        key: sum(row[key] for row in rows.values()) for key in next(iter(rows.values()), {}).keys()
    }
    return {
        "window_decisions": CLOSING_POCKET_WINDOW,
        "per_mix": dict(sorted(rows.items())),
        "total": total,
        "gating": False,
    }


def build_receipt(out: Path, check: Mapping[str, Any]) -> Dict[str, Any]:
    """Tier-1 ``receipt.json`` content from a finished screen directory."""
    intent_path, summary_path = out / "intent.json", out / "summary.json"
    intent = json.loads(intent_path.read_text())
    summary = json.loads(summary_path.read_text())
    records = []
    frames = 0
    seconds = 0.0
    by_arm: Dict[str, int] = {}
    entries: List[Mapping[str, Any]] = []
    for path in sorted((out / "records").glob("*.json")):
        entry = json.loads(path.read_text())
        entries.append(entry)
        records.append({"path": f"records/{path.name}", "sha256": dev_screen.sha256_file(path)})
        by_arm[entry["arm"]] = by_arm.get(entry["arm"], 0) + 1
        denominators = entry["record"].get("denominators") or {}
        frames += int(denominators.get("scored_frames", intent.get("smoke_frames") or 0))
        seconds += float(entry["wall_seconds"])
    pooled = summary.get("pooled_informational", {})
    decision = summary["decision"] if check["passes"] else "INVALID_SELF_CHECK_FAILED"
    return {
        "schema_version": RECEIPT_SCHEMA,
        "screen_id": SCREEN_ID,
        "authority": AUTHORITY,
        "label": "screen (non-authoritative)",
        "promotion_authorized": False,
        "intent_sha256": dev_screen.sha256_file(intent_path),
        "summary_sha256": dev_screen.sha256_file(summary_path),
        "records": records,
        "counts": {
            "episodes": len(records),
            "planned_episodes": summary.get("planned_episodes"),
            "by_arm": dict(sorted(by_arm.items())),
            "scored_frames": frames,
            "self_check_failures": len(check["failures"]),
        },
        "wall": {
            "started_utc": intent.get("started_utc"),
            "finished_utc": summary.get("finished_utc"),
            "episode_seconds_total": seconds,
            "stopped_reason": summary.get("stopped_reason"),
        },
        "primary": {
            "metric": "paired mass_integral B - A per world (B = v3, A = v2)",
            "holm_one_sided": summary.get("holm_primary"),
            "per_mix_one_sided": {
                mix: row["primary_mass_integral"]["one_sided_test"]
                for mix, row in summary.get("per_mix", {}).items()
            },
            "stratified_mean_of_means_90": pooled.get("stratified_mean_of_means"),
        },
        "decision": decision,
        "screen_decision_before_self_check": summary["decision"],
        "self_check": dict(check),
        "reported": {"v3_closing_pocket": closing_pocket_report(entries)},
    }


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = dev_screen.parse_args(argv)
    smoke = args.smoke_frames is not None
    design = (args.worlds_per_mix, args.determinism_worlds)
    if not smoke and design != dev_screen.PREREGISTERED_DESIGN:
        # Refuse before any world is played: a smaller look at the real worlds
        # followed by the pre-registered run would reach a decision after a peek.
        print(
            f"refusing a non-smoke run of size {design}; the pre-registered design is "
            f"{dev_screen.PREREGISTERED_DESIGN} (worlds per mix, determinism worlds)",
            file=sys.stderr,
        )
        return 2
    code = dev_screen.main(argv, spec=SMOKE_SPEC if smoke else SPEC)
    if code != 0:
        return code
    out = Path(args.out).resolve()
    check = self_check(out, smoke)
    receipt = build_receipt(out, check)
    receipt["written_utc"] = datetime.now(timezone.utc).isoformat()
    dev_screen.write_new_json(out / "receipt.json", dev_screen.json_safe(receipt))
    print(json.dumps({"decision": receipt["decision"], "receipt": str(out / "receipt.json")}))
    return 0 if check["passes"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
