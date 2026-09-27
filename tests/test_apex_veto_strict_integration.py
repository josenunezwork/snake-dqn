"""Producer -> independent audit integration for the Apex veto strict challenge.

``strict_run.run`` executes its real non-smoke pipeline (real intent from ``build_intent``,
real calibration -> final -> serving stage layout, supervision/heartbeat/started files, the
real ``strict_audit.py`` child under ``python -I`` and the real self-check child). Only the
rollout is replaced: stage workers run in-process and each episode returns a real Tier-1
screen record moved onto its world (``tests/fixtures/apex_veto_strict``). Nothing is written
outside ``tmp_path``.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from research.apex_safety_20260926 import dev_screen
from research.apex_veto_strict_20260927 import strict_run as sr
from src.core import game_config
from src.evaluation.strict_promotion import _expected_world_identity
from tests import test_apex_veto_strict_audit as helpers

HAVE_PILOT = dev_screen.DEFAULT_PILOT_OUTPUT.is_dir()
HAVE_CHECKPOINTS = helpers.HAVE_CHECKPOINTS and all(
    (dev_screen.DEFAULT_CHECKPOINT_DIR / name).is_file() for name, _ in dev_screen.POOL
)
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not (HAVE_PILOT and HAVE_CHECKPOINTS), reason="strict pilot or pool checkpoints absent"
    ),
]


def small_screen_pilot(root: Path) -> Path:
    """Screen-shaped pilot plus the summary.json cross-check ``prepare`` requires (N = 40)."""
    helpers.build_pilot(root, delta_sd=20.0)
    deltas = helpers.pilot_deltas(root)
    summary = {
        "per_mix": {
            mix: {"primary_mass_integral": {"deltas_B_minus_A": deltas[mix]}}
            for mix in helpers.MIXES
        }
    }
    helpers.write_json(root / "summary.json", summary)
    return root


def fake_rollout(intent: Dict[str, Any], shift: float):
    """``run_unit_episode`` stand-in: a real screen record on this unit's world."""
    index = sr.world_index_maps(intent)

    def episode(unit, row, lookup, profile, smoke_frames):
        assert smoke_frames is None and profile.digest == intent["profile"]["digest"]
        seed = int(unit["world_seed"])
        position = index[unit["stage"]][seed]
        if unit["stage"] == "serving":
            return helpers.serving_entry_template(position, seed)["record"]
        arm = "A" if unit["arm"] == "incumbent" else "B"
        mass = 120.0 + seed % 61
        if arm == "B":
            mass += shift + (seed // 7) % 21 - 10
        entry = helpers.relabel(arm, unit["mix"], position, seed, None, mass)
        record = entry["record"]
        record["world_identity"] = _expected_world_identity(row)
        # Keep the validator's mass identities: mean = integral x H / alive <= max = peak.
        alive = record["denominators"]["alive_frames"]
        record["mean_mass_alive"] = mass * helpers.SA.HORIZON / alive
        peak = max(float(record["max_mass"]), record["mean_mass_alive"])
        record["max_mass"] = record["probes"]["peak_length"] = peak
        return record

    return episode


def in_process_supervisor(output: Path, lock_root: Path):
    """Workers run in-process (fake rollout); audit children stay real subprocesses."""
    real = sr.supervise_children

    def supervise(commands, **kwargs):
        if "worker" not in commands[0]:
            return real(commands, **kwargs)
        children: List[Dict[str, Any]] = []
        for command, log in zip(commands, kwargs["logs"]):
            Path(log).write_text("in-process test worker\n", encoding="utf-8")
            stage = command[command.index("--stage") + 1]
            shard = int(command[command.index("--shard") + 1])
            deadline = sr.parse_utc(command[command.index("--stage-deadline") + 1])
            intent = sr.read_json(Path(command[command.index("--intent") + 1]))
            admitted = sr.read_json(output / "admitted.json")
            shard_dir = output / stage / f"shard-{shard}"
            shard_dir.mkdir()
            handle = sr.acquire_slot(lock_root, shard + 1, timeout=5)
            try:
                code = sr._worker_body(
                    intent, admitted, stage, shard, deadline, shard_dir, output / stage / "records"
                )
            finally:
                handle.close()
            children.append(
                {
                    "command": list(command),
                    "returncode": code,
                    "termination": "natural_exit",
                    "confirmed_exit": True,
                    "peak_group_rss_bytes": 0,
                }
            )
        return {
            "cause": None,
            "elapsed_seconds": 0.0,
            "wall_seconds": kwargs["wall_seconds"],
            "children": children,
            "min_available_bytes": None,
            "rss_limit_bytes": sr.RSS_BYTES,
            "available_floor_bytes": sr.AVAILABLE_BYTES,
        }

    return supervise


def run_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shift: float):
    pilot = small_screen_pilot(tmp_path / "screen")
    locks = tmp_path / "locks"
    locks.mkdir()
    intent = sr.build_intent(
        out_root=tmp_path / "run-v1",
        deadline=datetime.now(timezone.utc) + timedelta(hours=11),
        authorization_quote="integration test",
        screen_run=pilot,
        slot_lock_root=locks,
        python=Path(sys.executable),
        allow_dirty_source=True,
    )
    assert intent["sizing"]["final_worlds_per_mix"] == 40 and intent["sizing"]["feasible"]
    path = sr.prepare(intent)
    output = Path(intent["output_root"]) / "output"
    environment = {"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
    for key, value in environment.items():
        monkeypatch.setenv(key, value)  # restored at teardown (run_stages updates os.environ)
    stub = SimpleNamespace(
        GLOBAL_LOCK_ROOT=locks,
        capacity_environment=lambda threads: dict(environment),
        capacity_preflight=lambda gib: {"stub": "integration test", "reserve_gib": gib},
        close_slots=lambda handles: [h.close() for h in handles],
    )
    monkeypatch.setattr(sr, "load_supervisor", lambda path, sha: stub)
    monkeypatch.setattr(sr, "supervise_children", in_process_supervisor(output, locks))
    monkeypatch.setattr(sr, "run_unit_episode", fake_rollout(intent, shift))
    monkeypatch.setattr(sr.dev_screen, "agent_lookup", lambda snapshots: {})
    # In-process workers load the eval config and set torch threads: keep both test-local.
    monkeypatch.setattr(sr.dev_screen, "_configure_torch", lambda: None)
    monkeypatch.setattr(game_config, "_current_config", game_config._current_config)
    return sr.run(path), output


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("shift, outcome", [(60.0, "STRICT_PASS"), (-5.0, "STRICT_FAIL")])
def test_run_is_accepted_by_the_independent_audit(tmp_path, monkeypatch, shift, outcome):
    closeout, output = run_pipeline(tmp_path, monkeypatch, shift)
    audit = read(output / "audit" / "audit.json")
    failed = [row["rule"] for row in audit.get("failures", [])]
    assert closeout["outcome"] == outcome, (closeout, failed)
    assert audit["status"] == "PASS" and not failed
    assert closeout["audit_passed"] is True and closeout["decisions_agree"] is True
    assert closeout["independent_audit_failures"] in (None, [])
    assert read(output / "self-check" / "report.json")["passed"] is True
    assert read(output / "producer-outcome.json")["outcome"] == outcome
    counts = {s: len(list((output / s / "records").glob("*.json"))) for s in sr.STAGES}
    assert counts == {"calibration": 48, "final": 240, "serving": 50}
    assert (output / "receipt.json").is_file() == (outcome == "STRICT_PASS")
    assert closeout["promotion_performed"] is False
    assert closeout["serving_path_qualified"] is False


# ---------------------------------------------------------------- smoke dry-run audit


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A real 2-episode x 40-frame smoke (real rollout), run once for the module."""
    base = tmp_path_factory.mktemp("smoke")
    (base / "locks").mkdir()
    intent = sr.build_intent(
        out_root=base / "smoke",
        deadline=datetime.now(timezone.utc) + timedelta(hours=1),
        authorization_quote="integration smoke",
        slot_lock_root=base / "locks",
        python=Path(sys.executable),
        smoke_frames=40,
    )
    assert intent["audit"]["independent_command"][-1] == "--smoke"
    environment = dict(os.environ)
    try:
        closeout = sr.run(sr.prepare(intent))
    finally:  # run_stages exports the supervisor's thread limits into os.environ
        os.environ.clear()
        os.environ.update(environment)
    assert closeout["outcome"] == "SMOKE_NO_DECISION", closeout
    return Path(intent["output_root"])


def smoke_audit(root: Path, tmp_path: Path) -> List[str]:
    report = helpers.SA.run_smoke_audit(root.resolve(), (tmp_path / "audit-out").resolve())
    assert report["mode"] == "smoke"
    return [row["rule"] for row in report["failures"]]


def test_smoke_run_passes_the_gating_independent_audit(smoke_run, tmp_path):
    output = smoke_run / "output"
    audit = read(output / "audit" / "audit.json")
    assert audit["status"] == "PASS" and audit["mode"] == "smoke", audit.get("failures")
    closeout = read(output / "closeout.json")
    assert closeout["audit_passed"] is True and closeout["independent_audit_failures"] in (
        None,
        [],
    )
    assert not (output / "producer-outcome.json").exists()
    assert smoke_audit(smoke_run, tmp_path) == []
    full = helpers.SA.run_audit(
        smoke_run.resolve(), (tmp_path / "full").resolve(), helpers.REAL_SCREEN
    )
    assert "mode.not_smoke" in [row["rule"] for row in full["failures"]]


def clone_smoke(smoke_run: Path, tmp_path: Path) -> Path:
    import shutil

    target = tmp_path / "smoke"
    shutil.copytree(smoke_run, target, ignore=shutil.ignore_patterns("audit", "closeout.json"))
    return target


def mutate_record(root: Path, arm: str, fn) -> None:
    """Edit one real smoke record and re-list its hash, so only record rules can fail."""
    output = root / "output" / "final"
    path = next((output / "records").glob(f"final-{arm}-*.json"))
    entry = read(path)
    fn(entry)
    path.write_text(json.dumps(entry, sort_keys=True, indent=2), encoding="utf-8")
    for report_path in output.glob("shard-*/report.json"):
        report = read(report_path)
        if path.stem in report["records_sha256"]:
            report["records_sha256"][path.stem] = dev_screen.sha256_file(path)
            report_path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")


SMOKE_RECORD_MUTATIONS = {
    "candidate_probe_removed": ("candidate", lambda e: e["record"]["probes"].pop("safety_veto")),
    "candidate_flag_off": ("candidate", lambda e: e.update(safety_veto=False)),
    "incumbent_wrapper": ("incumbent", lambda e: e.update(wrapper=helpers.SA.VETO_METHOD)),
    "incumbent_probe": (
        "incumbent",
        lambda e: e["record"]["probes"].update(safety_veto={"method": "x", "counters": {}}),
    ),
    "wrong_seed": ("incumbent", lambda e: e["record"].update(seed=e["record"]["seed"] + 1)),
    "wrong_roster": ("incumbent", lambda e: e.update(roster_member_sha256s=["0" * 64] * 5)),
    "wrong_hero": ("candidate", lambda e: e.update(hero_sha256="0" * 64)),
    "frames_over_cap": ("incumbent", lambda e: e["record"].update(frames_completed=41)),
    "veto_counter_over_cap": (
        "candidate",
        lambda e: e["record"]["probes"]["safety_veto"]["counters"].update(decisions=41),
    ),
    "negative_mass": ("candidate", lambda e: e["record"].update(mass_integral=-1.0)),
}


@pytest.mark.parametrize("mutation", sorted(SMOKE_RECORD_MUTATIONS))
def test_smoke_audit_rejects_mutated_real_smoke_records(smoke_run, tmp_path, mutation):
    root = clone_smoke(smoke_run, tmp_path)
    arm, fn = SMOKE_RECORD_MUTATIONS[mutation]
    mutate_record(root, arm, fn)
    assert "smoke.records_valid" in smoke_audit(root, tmp_path)


def test_smoke_audit_rejects_run_level_tampering(smoke_run, tmp_path):
    root = clone_smoke(smoke_run, tmp_path)
    record = next((root / "output" / "final" / "records").glob("*.json"))
    record.write_text(record.read_text() + " ", encoding="utf-8")  # bytes changed
    assert "shards.plan_assignment_and_hashes" in smoke_audit(root, tmp_path / "a")
    root = clone_smoke(smoke_run, tmp_path / "b")
    helpers.write_json(root / "output" / "producer-outcome.json", {"outcome": "STRICT_PASS"})
    assert "smoke.no_decision_claims" in smoke_audit(root, tmp_path / "b")
    root = clone_smoke(smoke_run, tmp_path / "c")
    intent = read(root / "intent.json")
    intent["smoke_frames"] = None
    helpers.write_json(root / "intent.json", intent)
    assert "smoke.intent" in smoke_audit(root, tmp_path / "c")
    root = clone_smoke(smoke_run, tmp_path / "d")
    extra = root / "output" / "calibration" / "records" / "x.json"
    helpers.write_json(extra, {"arm": "incumbent"})
    assert "smoke.no_tier2_stages" in smoke_audit(root, tmp_path / "d")
