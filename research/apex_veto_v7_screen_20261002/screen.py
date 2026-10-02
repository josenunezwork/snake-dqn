#!/usr/bin/env python3
"""Tier-1 screen (non-authoritative): Apex + v7 space preference vs Apex + v5 veto.

Pre-registration: ``protocol.md`` beside this file. The harness is
``research/apex_safety_20260926/dev_screen.py`` driven by a spec built from the DEV lambda
sweep's ``summary.json`` (``research/apex_veto_v7_lambda_sweep_20261002``):

  A  champion + ``free-space-veto/v5-boost-aware`` (released Watch-hero configuration).
  B  champion + ``free-space-veto/v7-space-preference(lambda=<selected>)``.
  C  arm A repeated on the first 8 worlds per mix (determinism control).
  D  arm B repeated on the first 4 worlds per mix (B replay control).

Outside a smoke the screen refuses to start unless ``--sweep-summary`` names a real (not
smoke) sweep summary whose selection is ``SELECTED`` and whose source commit equals this
run's commit, both trees free of tracked modifications. It always holds a CPU slot lock
and refuses to start on battery; afterwards it self-checks every record and writes
``receipt.json``. Nothing here changes a default.

Usage (about 2.5-3 h at the pre-registered size; hard cap 4 h):
  SNAKE_DQN_DEVICE=cpu ./venv/bin/python research/apex_veto_v7_screen_20261002/screen.py \\
    --sweep-summary <sweep run>/summary.json --out /path/to/new/dir \\
    --deadline-utc <now + at most 4 h, with UTC offset>
Smoke (plumbing only, smoke namespace, <= 2 episodes x 500 frames):
  ... screen.py --smoke-lambda 0.5 --out /tmp/x --deadline-utc ... --smoke-frames 500 \\
    --worlds-per-mix 1 --determinism-worlds 0 --smoke-mixes scripted
"""

from __future__ import annotations

import os

os.environ.setdefault("SNAKE_DQN_DEVICE", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from functools import partial  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List, Mapping, Sequence, Tuple  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.apex_safety_20260926 import dev_screen  # noqa: E402
from research.apex_veto_v5_screen_20261001 import screen as v5screen  # noqa: E402
from research.apex_veto_v6_screen_20261002 import screen as v6screen  # noqa: E402
from research.apex_veto_v7_lambda_sweep_20261002 import sweep  # noqa: E402

HERE = Path(__file__).resolve().parent
SCREEN_ID = "apex-veto-v7-screen-v1"
SCHEMA = "apex-veto-v7-screen/v1"
AUTHORITY = "screen (non-authoritative)"
DOMAIN = SCREEN_ID
SMOKE_DOMAIN = "apex-veto-v7-screen-smoke-v1"
NAMESPACE = "worlds"
RECEIPT_SCHEMA = "apex-veto-v7-screen-receipt/v1"
V5_METHOD = sweep.V5_METHOD
V5_ARMS = ("A", "C")
V7_ARMS = ("B", "D")
REPLAY_WORLDS = 4
MAX_WALL_SECONDS = 4 * 3600
EXCLUSION_PREFIX = 1000
# Every domain the v6 screen excluded, the v6 screen and its smoke, the v7 DEV sweep and
# its smoke (whose worlds chose lambda), and this screen's other domain.
EARLIER_DOMAINS: Dict[str, Sequence[str]] = {
    **{name: tuple(purposes) for name, purposes in v6screen.EARLIER_DOMAINS.items()},
    sweep.DOMAIN: (sweep.NAMESPACE,),
    sweep.SMOKE_DOMAIN: (sweep.NAMESPACE,),
    DOMAIN: (NAMESPACE,),
    SMOKE_DOMAIN: (NAMESPACE,),
}


class SweepBindingError(ValueError):
    """The sweep summary does not authorize a screen run."""


def earlier_namespaces(domain: str, prefix: int = EXCLUSION_PREFIX) -> Dict[str, List[int]]:
    """``domain/purpose[0:prefix]`` -> seeds for every earlier domain except ``domain``."""
    return {
        f"{name}/{purpose}[0:{prefix}]": [
            dev_screen.uint32_seed(name, purpose, index) for index in range(prefix)
        ]
        for name, purposes in EARLIER_DOMAINS.items()
        if name != domain
        for purpose in purposes
    }


def tracked_modifications(dirty_paths: Any) -> List[str]:
    """``git status --porcelain`` lines other than untracked (``??``) files."""
    lines = [line for line in str(dirty_paths or "").splitlines() if line.strip()]
    return [line for line in lines if not line.startswith("??")]


def load_sweep_selection(path: Path, git: Mapping[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """``(lambda, binding)`` from a sweep ``summary.json``; raises :class:`SweepBindingError`.

    Requirements: the file and its ``intent.json`` exist and match (``intent_sha256``); the
    sweep schema and id; not a smoke; selection ``SELECTED`` with ``passes`` and a lambda of
    the sweep grid; the sweep's source commit (summary and intent) equals ``git['commit']``;
    neither the sweep's tree nor this run's has tracked modifications.
    """
    path = Path(path)
    if not path.is_file():
        raise SweepBindingError(f"sweep summary {path} does not exist")
    intent_path = path.parent / "intent.json"
    if not intent_path.is_file():
        raise SweepBindingError("sweep intent.json is missing beside the summary")
    summary = json.loads(path.read_text())
    intent = json.loads(intent_path.read_text())
    if summary.get("schema_version") != sweep.SCHEMA or summary.get("sweep_id") != sweep.SWEEP_ID:
        raise SweepBindingError("not a v7 lambda sweep summary")
    if summary.get("smoke") is not False:
        raise SweepBindingError("a smoke sweep cannot select lambda")
    if summary.get("intent_sha256") != dev_screen.sha256_file(intent_path):
        raise SweepBindingError("sweep intent.json does not match the summary's intent_sha256")
    selection = summary.get("selection") or {}
    lam = selection.get("selected_lambda")
    if selection.get("status") != sweep.STATUS_SELECTED or selection.get("passes") is not True:
        raise SweepBindingError(f"sweep selection did not pass: {selection.get('status')!r}")
    if isinstance(lam, bool) or lam not in sweep.LAMBDAS:
        raise SweepBindingError(f"selected lambda {lam!r} is not on the sweep grid")
    source = summary.get("source") or {}
    commit = git.get("commit")
    if (
        not commit
        or source.get("commit") != commit
        or intent.get("git", {}).get("commit") != commit
    ):
        raise SweepBindingError(
            f"sweep source commit {source.get('commit')!r} differs from this run's {commit!r}"
        )
    if tracked_modifications(source.get("dirty_paths")):
        raise SweepBindingError("the sweep ran with tracked modifications")
    if tracked_modifications(git.get("dirty_paths")):
        raise SweepBindingError("this run has tracked modifications")
    binding = {
        "summary_path": str(path.resolve()),
        "summary_sha256": dev_screen.sha256_file(path),
        "intent_sha256": summary["intent_sha256"],
        "source_commit": commit,
        "selected_lambda": float(lam),
        "selection": selection,
    }
    return float(lam), binding


def install_v5(hero: Any) -> Any:
    """Arms A and C: the released v5 boost-aware veto."""
    return sweep.install_v5(hero)


def build_spec(domain: str, lam: float, binding: Mapping[str, Any]) -> dev_screen.ScreenSpec:
    """The screen spec for ``lambda = lam`` (B and D share one installer object)."""
    from src.evaluation.safety_veto_v7 import method_for

    method = method_for(lam)
    install_b = partial(sweep.install_v7, lam=float(lam))
    origin = (
        f"lambda {float(lam)!r} from sweep summary sha256 {binding.get('summary_sha256')} "
        f"(commit {binding.get('source_commit')})"
        if binding.get("summary_sha256")
        else f"lambda {float(lam)!r} given for a smoke (--smoke-lambda)"
    )
    return dev_screen.ScreenSpec(
        name=SCREEN_ID,
        schema=SCHEMA,
        authority=AUTHORITY,
        domain=domain,
        namespace=NAMESPACE,
        protocol=HERE / "protocol.md",
        arm_vetoes={"A": install_v5, "B": install_b, "C": install_v5, "D": install_b},
        arm_descriptions={
            "A": f"champion + {V5_METHOD} (src/evaluation/safety_veto_v5.py, released)",
            "B": f"champion + {method} (src/evaluation/safety_veto_v7.py; {origin})",
            "C": "arm A repeated on the first determinism worlds per mix",
            "D": f"arm B repeated on the first {REPLAY_WORLDS} worlds per mix",
        },
        extra_namespaces=earlier_namespaces(domain),
        require_slot_locks=True,
        require_ac_power=True,
        owner="Apex safety lane",
        hypothesis=(
            f"Re-ranking v5's eligible moves toward the larger reachable region "
            f"(v7, {origin}) raises the Apex champion's H5000 mass_integral over the "
            "released v5 veto (B - A > 0)."
        ),
        decision_informs=(
            "RECOMMEND_STRICT_GATE: design a Tier-2 strict gate for v7 against v5; "
            "NOT_ADVANCED: stop (no change to the served v5 configuration)"
        ),
        primary_metric="paired mass_integral B-A per world per mix",
        estimator="one-sided paired t, Holm over 3 mixes, alpha 0.05, >=2 of 3",
        max_wall_seconds=MAX_WALL_SECONDS,
        replay_worlds=REPLAY_WORLDS,
    )


def expected_descriptor(arm: str, lam: float) -> Dict[str, Any]:
    """The probe identity each arm's records must carry (writer: ``veto.record()``)."""
    from src.evaluation.safety_veto_v5 import BoostAwareFreeSpaceVeto
    from src.evaluation.safety_veto_v7 import SpacePreferenceVeto

    if arm in V7_ARMS:
        return SpacePreferenceVeto(lam).descriptor()
    return BoostAwareFreeSpaceVeto().descriptor()


# ---------------------------------------------------------------- self-check and receipt


def check_record(
    record: Mapping[str, Any],
    arm: str,
    lam: float,
    *,
    world_seed: int,
    mix: str,
    roster_hashes: Sequence[str],
    smoke: bool,
) -> List[str]:
    """Gating checks on one rollout record (the v6 screen's rules, v7 descriptor for B/D)."""
    from src.evaluation.strict_promotion import (
        StrictPromotionArtifactError,
        _expected_world_identity,
        _validate_candidate_wrapper_probe,
    )

    failures: List[str] = []
    probes = record.get("probes") if isinstance(record, Mapping) else None
    probe = probes.get("safety_veto") if isinstance(probes, Mapping) else None
    try:
        _validate_candidate_wrapper_probe(probe, expected_descriptor(arm, lam))
    except (StrictPromotionArtifactError, TypeError, KeyError) as exc:
        failures.append(f"safety_veto probe is not arm {arm}'s veto: {exc}")
    if record.get("seed") != world_seed:
        failures.append("record seed differs from the planned world seed")
    mass = record.get("mass_integral")
    if not isinstance(mass, (int, float)) or isinstance(mass, bool) or mass != mass:
        failures.append("mass_integral is not a finite number")
    if smoke:
        return failures
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
    if counters.get("decisions") != denominators.get("decision_frames"):
        failures.append("veto decisions differ from denominators.decision_frames")
    return failures


def check_entry(entry: Mapping[str, Any], lam: float, smoke: bool) -> Dict[str, List[str]]:
    """Gating failures and reported warnings for one screen entry (dev_screen writer)."""
    from src.evaluation.safety_veto_v7 import method_for

    arm = entry.get("arm")
    if arm not in V5_ARMS + V7_ARMS:
        return {"failures": [f"unknown arm {arm!r}"], "warnings": []}
    failures: List[str] = []
    warnings: List[str] = []
    method = method_for(lam) if arm in V7_ARMS else V5_METHOD
    if entry.get("schema_version") != SCHEMA or entry.get("authority") != AUTHORITY:
        failures.append("entry schema/authority is not this screen's")
    if entry.get("screen") != SCREEN_ID or entry.get("hero_sha256") != dev_screen.CHAMPION[1]:
        failures.append("entry screen id or hero sha256 differs")
    if entry.get("safety_veto") is not True or entry.get("safety_veto_method") != method:
        failures.append(f"entry veto flag/method is not arm {arm}'s")
    record = entry.get("record") or {}
    failures.extend(
        check_record(
            record,
            arm,
            lam,
            world_seed=entry.get("world_seed"),
            mix=entry.get("mix"),
            roster_hashes=entry.get("roster_member_sha256s") or [],
            smoke=smoke,
        )
    )
    diagnostics = entry.get("veto_diagnostics")
    counters = ((record.get("probes") or {}).get("safety_veto") or {}).get("counters") or {}
    if arm in V7_ARMS:
        if not isinstance(diagnostics, Mapping) or "rerank_changes" not in diagnostics:
            warnings.append(f"arm {arm} v7 counters missing")
        elif not sweep.v7_identities_hold(diagnostics, counters):
            warnings.append(f"arm {arm} v7 counters disagree with the probe's v2 counters")
    elif not isinstance(diagnostics, Mapping) or "boost_landing_vetoes" not in diagnostics:
        warnings.append(f"arm {arm} v5 counters missing")
    elif not v5screen.probe_identities_hold(diagnostics, counters):
        warnings.append(f"arm {arm} v5 counters disagree with the probe's v2 counters")
    return {"failures": failures, "warnings": warnings}


def self_check(out: Path, lam: float, smoke: bool) -> Dict[str, Any]:
    """Check every saved record of a finished screen directory."""
    intent = json.loads((out / "intent.json").read_text())
    planned = {int(seed) for seed in intent["worlds"]["seeds"]}
    failures: List[str] = []
    warnings: List[str] = []
    paths = sorted((out / "records").glob("*.json"))
    for path in paths:
        entry = json.loads(path.read_text())
        name = f"{entry.get('arm')}-{entry.get('mix')}-{entry.get('world_seed')}.json"
        result = check_entry(entry, lam, smoke)
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


def v7_report(entries: Sequence[Mapping[str, Any]], arm: str = "B") -> Dict[str, Any]:
    """Reported only: summed v7 counters and per-decision cost for ``arm`` per mix and total.

    Writers: ``SpacePreferenceCounters.to_dict`` via ``veto_diagnostics`` and dev_screen's
    ``wall_seconds``; the ``*_seconds_*`` fields are wall-clock on the screen host.
    """

    def fresh() -> Dict[str, Any]:
        row: Dict[str, Any] = {key: 0.0 for key in sweep.V7_SUMMED}
        row.update({"episodes": 0, "missing_diagnostics": 0, "area_cap_max": 0, "wall": []})
        return row

    def add(row: Dict[str, Any], entry: Mapping[str, Any]) -> None:
        row["episodes"] += 1
        row["wall"].append(float(entry.get("wall_seconds", 0.0)))
        diag = entry.get("veto_diagnostics")
        if not isinstance(diag, Mapping) or "rerank_changes" not in diag:
            row["missing_diagnostics"] += 1
            return
        for key in sweep.V7_SUMMED:
            row[key] += diag.get(key) or 0
        row["area_cap_max"] = max(row["area_cap_max"], int(diag.get("area_cap_max") or 0))

    def finish(row: Dict[str, Any]) -> Dict[str, Any]:
        wall = row.pop("wall")
        decisions = row["decisions"]
        row["mean_apply_seconds_per_decision"] = (
            row["apply_seconds_total"] / decisions if decisions else None
        )
        row["mean_area_eval_seconds_per_decision"] = (
            row["area_eval_seconds_total"] / decisions if decisions else None
        )
        row["rerank_change_rate_per_decision"] = (
            row["rerank_changes"] / decisions if decisions else None
        )
        row["mean_episode_wall_seconds"] = sum(wall) / len(wall) if wall else None
        return row

    per_mix: Dict[str, Dict[str, Any]] = {}
    total = fresh()
    for entry in entries:
        if entry.get("arm") != arm:
            continue
        add(per_mix.setdefault(str(entry.get("mix")), fresh()), entry)
        add(total, entry)
    return {
        "arm": arm,
        "per_mix": {mix: finish(row) for mix, row in sorted(per_mix.items())},
        "total": finish(total) if per_mix else {},
        "gating": False,
    }


def build_receipt(
    out: Path, check: Mapping[str, Any], lam: float, binding: Mapping[str, Any]
) -> Dict[str, Any]:
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
        "lambda": float(lam),
        "sweep_binding": dict(binding),
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
            "max_wall_seconds": MAX_WALL_SECONDS,
        },
        "primary": {
            "metric": "paired mass_integral B - A per world (B = v7, A = v5)",
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
        "reported": {
            "determinism_control": summary.get("determinism_control"),
            "replay_control": summary.get("replay_control"),
            "v7_rerank_arm_B": v7_report(entries, "B"),
            "v5_landing_arm_A": v5screen.v5_landing_report(entries, "A"),
            "death_causes": v6screen.death_cause_report(entries),
        },
    }


def parse_binding_args(argv: Sequence[str]) -> Tuple[argparse.Namespace, List[str]]:
    """This wrapper's own flags; everything else goes to ``dev_screen.parse_args``."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--sweep-summary", type=Path, default=None)
    parser.add_argument("--smoke-lambda", type=float, default=None)
    return parser.parse_known_args(list(argv))


def main(argv: Sequence[str] | None = None) -> int:
    own, rest = parse_binding_args(sys.argv[1:] if argv is None else argv)
    args = dev_screen.parse_args(rest)
    smoke = args.smoke_frames is not None
    design = (args.worlds_per_mix, args.determinism_worlds)
    if not smoke and design != dev_screen.PREREGISTERED_DESIGN:
        print(
            f"refusing a non-smoke run of size {design}; the pre-registered design is "
            f"{dev_screen.PREREGISTERED_DESIGN} (worlds per mix, determinism worlds)",
            file=sys.stderr,
        )
        return 2
    binding: Dict[str, Any] = {}
    if smoke and own.smoke_lambda is not None:
        if own.smoke_lambda not in sweep.LAMBDAS or own.sweep_summary is not None:
            print("--smoke-lambda must be a sweep-grid lambda, alone", file=sys.stderr)
            return 2
        lam = float(own.smoke_lambda)
    elif own.smoke_lambda is not None:
        print("--smoke-lambda is allowed only with --smoke-frames", file=sys.stderr)
        return 2
    elif own.sweep_summary is None:
        print("refusing: --sweep-summary (a SELECTED v7 lambda sweep) is required", file=sys.stderr)
        return 2
    else:
        try:
            lam, binding = load_sweep_selection(own.sweep_summary, dev_screen._git_state())
        except (SweepBindingError, OSError, json.JSONDecodeError) as exc:
            print(f"refusing: {exc}", file=sys.stderr)
            return 2
    spec = build_spec(SMOKE_DOMAIN if smoke else DOMAIN, lam, binding)
    code = dev_screen.main(rest, spec=spec)
    if code != 0:
        return code
    out = Path(args.out).resolve()
    check = self_check(out, lam, smoke)
    receipt = build_receipt(out, check, lam, binding)
    receipt["written_utc"] = datetime.now(timezone.utc).isoformat()
    dev_screen.write_new_json(out / "receipt.json", dev_screen.json_safe(receipt))
    print(json.dumps({"decision": receipt["decision"], "receipt": str(out / "receipt.json")}))
    return 0 if check["passes"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
