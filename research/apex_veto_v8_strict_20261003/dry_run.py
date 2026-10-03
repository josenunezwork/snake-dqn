#!/usr/bin/env python3
"""Plumbing dry run of the v8 strict spec (never a Tier-2 run; plays no game).

``build_intent(..., dry_run=True)`` with this study's real ``SPEC`` (arm identities, excluded
seeds, banks, rosters, bands, pre-registration documents, frozen plan N_max 249 / MDE 30),
except that ``episode_runner`` and ``worker_setup`` are replaced by a tiny deterministic fake
(records shaped like the real ones, so the real ``validate_record`` runs). The in-process
executor is required by the template; this script also runs the real frozen skew probe and the
real independent audit as supervised children. Its own slot lock root and ledger live under
``--out`` (never the global ones), so it cannot burn the final namespace or take CPU slots.
A dry run can only end ``DRY_RUN_*`` or a failure and never writes ``receipt.json``.

``--worker-check`` additionally drives ONE calibration segment through the production
``SubprocessExecutor`` (real ``python -I`` worker children, inherited slot fds under a scratch
lock root, heartbeats, the worker's own ``resolve_spec`` of :data:`FAKE_SPEC_REF`), outside
``run()`` (which allows only the in-process executor for a dry run). The fake episodes make it
a plumbing check of the worker path only; it leaves a scratch root without a closeout.

Usage: ``python research/apex_veto_v8_strict_20261003/dry_run.py --out <scratch dir>
[--scenario screen|large|null] [--worker-check]``.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Mapping

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.apex_veto_v8_strict_20261003 import spec as S  # noqa: E402
from research.sequential_strict_template import sequential_runner as R  # noqa: E402

HERE = Path(__file__).resolve().parent


def _unit(*parts: Any) -> float:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


class FakeEpisodes:
    """Deterministic fake records: incumbent mass/survival per world, candidate = incumbent
    + a screen delta (``screen``), + 150 (``large``) or + 0 (``null``)."""

    def __init__(self, scenario: str) -> None:
        self.scenario = scenario
        self.deltas = json.loads((HERE / "screen_deltas.json").read_text())

    def __call__(
        self, episode: Mapping[str, Any], row: Mapping[str, Any], context: Any
    ) -> Dict[str, Any]:
        seed, mix, arm = int(episode["world_seed"]), episode["mix"], episode["arm"]
        base = 200.0 + 250.0 * _unit(seed, mix, "mass")
        survival = 0.5 + 0.5 * _unit(seed, mix, "survival")
        if arm == "candidate":
            if self.scenario == "screen":
                pool = self.deltas[mix]
                base += pool[int(_unit(seed, mix, "pick") * len(pool))]
            elif self.scenario == "large":
                base += 150.0
                survival = min(1.0, survival + 0.2)
        return {
            "seed": seed,
            "mass_integral": base,
            "survival_fraction": survival,
            "world_identity": {"seed": seed, "mix_id": mix},
            "probes": {"safety_veto": {"method": S.ARM_METHODS[arm]}},
            "veto_diagnostics": {S.DIAGNOSTICS_MARKER[arm]: 0, "fake": True},
        }


def _no_setup(intent: Mapping[str, Any]) -> None:
    return None


FAKE_SPEC_REF = "research.apex_veto_v8_strict_20261003.dry_run:FAKE_SPEC"
FAKE_SPEC = dataclasses.replace(
    S.SPEC, episode_runner=FakeEpisodes("screen"), worker_setup=_no_setup
)


def intent_for(spec, out_root: Path, scratch: Path, skew_reps: int, tag: str) -> Path:
    (scratch / f"locks-{tag}").mkdir()
    intent = R.build_intent(
        spec,
        spec_ref=FAKE_SPEC_REF,
        out_root=out_root,
        n_max=249,
        mde=30.0,
        n_calibration=S.N_CALIBRATION,
        skew_input=HERE / "screen_deltas.json",
        deadline=datetime.now(timezone.utc) + timedelta(hours=16),
        authorization_quote="dry run (plumbing only, fake episodes)",
        skew_reps=skew_reps,
        slot_lock_root=scratch / f"locks-{tag}",
        ledger_path=scratch / f"ledger-{tag}.jsonl",
        dry_run=True,
    )
    return R.prepare(intent)


def worker_check(scratch: Path, skew_reps: int) -> Dict[str, Any]:
    """One calibration segment through the real subprocess workers (fake episodes)."""
    path = intent_for(FAKE_SPEC, scratch / "worker-check", scratch, skew_reps, "wc")
    intent = R.read_json(path)
    slots = R.acquire_run_slots(Path(intent["slot_lock_root"]), intent["caps"]["workers"])
    try:
        ctx = R.RunContext(
            intent=intent,
            intent_path=path,
            intent_sha=R.sha256_file(path),
            spec=FAKE_SPEC,
            output=path.parent / "output",
            executor=R.SubprocessExecutor(slots),
            skew_runner=R.subprocess_skew_runner,
            audit_runner=R.subprocess_audit_runner,
            provenance={"executor": "subprocess (worker plumbing check)"},
        )
        ctx.output.mkdir()
        R.write_once(
            ctx.output / "started.json",
            {"pid": os.getpid(), "intent_sha256": ctx.intent_sha, "worker_check": True},
        )
        ctx.state["stage_started_mono"] = {}
        R.admit_rosters(ctx)
        R.run_segment(ctx, "calibration", 0)
        entries = R.collect_prefix(ctx, "calibration", 0)
        seg = R.segment_dir(ctx.output, "calibration", 0)
        supervision = R.read_json(seg / "supervision.json")
        started = [R.read_json(seg / f"shard-{k}" / "started.json") for k in range(2)]
        return {
            "records": len(entries),
            "expected": 3 * S.N_CALIBRATION,
            "cause": supervision["cause"],
            "children": [
                {k: c.get(k) for k in ("returncode", "termination", "confirmed_exit")}
                for c in supervision["children"]
            ],
            "shards_started": [s["shard"] for s in started],
            "passes": len(entries) == 3 * S.N_CALIBRATION and supervision["cause"] is None,
        }
    finally:
        R.release_run_slots(slots)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scenario", choices=("screen", "large", "null"), default="screen")
    parser.add_argument("--skew-reps", type=int, default=R.DEFAULT_SKEW_REPS)
    parser.add_argument("--worker-check", action="store_true")
    args = parser.parse_args(argv)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    spec = dataclasses.replace(FAKE_SPEC, episode_runner=FakeEpisodes(args.scenario))
    path = intent_for(spec, out / "run", out, args.skew_reps, "run")
    closeout = R.run(path, spec=spec, executor=R.InProcessExecutor())
    keys = (
        "outcome",
        "stop_look",
        "stop_decision",
        "worlds_per_mix_used",
        "audit_passed",
        "failure",
        "deadline_stop",
        "audit_failures",
    )
    report: Dict[str, Any] = {"dry_run": {k: closeout.get(k) for k in keys}}
    ok = closeout["outcome"].startswith("DRY_RUN")
    if args.worker_check:
        report["worker_check"] = worker_check(out, args.skew_reps)
        ok = ok and report["worker_check"]["passes"]
    print(json.dumps(report, indent=1, default=str))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
