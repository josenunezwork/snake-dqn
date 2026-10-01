"""Producer -> independent audit integration for the v5 strict challenge.

``strict_run.run`` executes its real pipeline (real intent from ``build_intent``, real
calibration -> final -> serving stage layout, supervision/heartbeat/started files, the real
``strict_audit.py`` child under ``python -I`` and the real self-check child). No episode is
played: ``tournament_eval.rollout`` and ``dev_screen.run_episode`` raise, stage workers run
in-process, and ``run_unit_episode`` is replaced by a stand-in returning real record shapes
(``tests/fixtures/apex_veto_v5_strict``) moved onto each unit's world. Nothing is written
outside ``tmp_path``.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from research.apex_safety_20260926 import dev_screen
from research.apex_veto_v5_strict_20261001 import strict_run as sr
from src.core import game_config
from src.evaluation.strict_promotion import _expected_world_identity
from tests import test_apex_veto_v5_strict_audit as helpers

block_episode_runners = helpers.block_episode_runners  # autouse fixture (re-exported)
HAVE_PILOT = dev_screen.DEFAULT_PILOT_OUTPUT.is_dir()
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not (HAVE_PILOT and helpers.HAVE_CHECKPOINTS and sr.SUPERVISOR_SOURCE.is_file()),
        reason="strict pilot, pool checkpoints or supervisor helpers absent",
    ),
]


def fake_episode(intent: Dict[str, Any], shift: float):
    """``run_unit_episode`` stand-in: a real record shape on this unit's world."""
    index = sr.world_index_maps(intent)
    smoke = intent["smoke_frames"] is not None

    def episode(unit, row, lookup, profile, smoke_frames):
        assert smoke_frames == intent["smoke_frames"]
        letter = helpers.LETTER[unit["arm"]]
        diag = dict(helpers.DIAGNOSTICS) if unit["arm"] == "candidate" else None
        seed = int(unit["world_seed"])
        if smoke:
            record = json.loads(json.dumps(helpers.SMOKE[letter]["record"]))
            record["seed"] = seed
            return record, diag
        assert profile.digest == intent["profile"]["digest"]
        mass = 120.0 + seed % 61
        if unit["arm"] == "candidate" and unit["stage"] == "final":
            mass += shift + (seed // 7) % 21 - 10
        record = helpers.move_record(letter, unit["mix"], index[unit["stage"]][seed], seed, mass)
        record["world_identity"] = _expected_world_identity(row)
        return record, diag

    return episode


def in_process_supervisor(output: Path, lock_root: Path):
    """Workers run in-process (fake episodes); audit children stay real subprocesses."""
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
            slot_fd = int(command[command.index("--slot-fd") + 1])
            intent = sr.read_json(Path(command[command.index("--intent") + 1]))
            admitted = sr.read_json(output / "admitted.json")
            shard_dir = output / stage / f"shard-{shard}"
            shard_dir.mkdir()
            assert kwargs["pass_fds"][shard] == (slot_fd,)
            sr.assert_inherited_slot(slot_fd, lock_root, shard + 1)
            code = sr._worker_body(
                intent, admitted, stage, shard, deadline, shard_dir, output / stage / "records"
            )
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


def run_pipeline(
    base: Path, monkeypatch: pytest.MonkeyPatch, shift: float, smoke_frames: int | None = None
):
    pilot = helpers.build_pilot(base / "screen", delta_sd=10.0)
    locks = base / "locks"
    locks.mkdir()
    out_root = base / ("smoke" if smoke_frames else "run-v1")
    monkeypatch.setattr(sr, "TIER2_OUT_ROOT", base / "run-v1")
    monkeypatch.setattr(sr, "on_ac_power", lambda: True)
    intent = sr.build_intent(
        out_root=out_root,
        deadline=datetime.now(timezone.utc) + timedelta(hours=15),
        authorization_quote="integration test",
        screen_run=pilot,
        slot_lock_root=locks,
        python=Path(sys.executable),
        allow_dirty_source=True,
        smoke_frames=smoke_frames,
    )
    path = sr.prepare(intent)
    output = Path(intent["output_root"]) / "output"
    environment = {"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
    for key, value in environment.items():
        monkeypatch.setenv(key, value)  # restored at teardown (run_stages updates os.environ)
    stub = SimpleNamespace(
        GLOBAL_LOCK_ROOT=locks,
        capacity_environment=lambda threads: dict(environment),
        capacity_preflight=lambda gib: {"stub": "integration test", "reserve_gib": gib},
    )
    monkeypatch.setattr(sr, "load_supervisor", lambda path, sha: stub)
    monkeypatch.setattr(sr, "supervise_children", in_process_supervisor(output, locks))
    monkeypatch.setattr(sr, "run_unit_episode", fake_episode(intent, shift))
    monkeypatch.setattr(sr.dev_screen, "agent_lookup", lambda snapshots: {})
    monkeypatch.setattr(sr.dev_screen, "_configure_torch", lambda: None)
    monkeypatch.setattr(game_config, "_current_config", game_config._current_config)
    saved = dict(os.environ)
    try:
        closeout = sr.run(path)
    finally:
        os.environ.clear()
        os.environ.update(saved)
    return closeout, output, intent, pilot


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("shift, outcome", [(60.0, "STRICT_PASS"), (-5.0, "STRICT_FAIL")])
def test_run_is_accepted_by_the_independent_audit(tmp_path, monkeypatch, shift, outcome):
    closeout, output, intent, _ = run_pipeline(tmp_path, monkeypatch, shift)
    audit = read(output / "audit" / "audit.json")
    failed = [row["rule"] for row in audit.get("failures", [])]
    assert closeout["outcome"] == outcome, (closeout, failed)
    assert audit["status"] == "PASS" and not failed
    assert closeout["audit_passed"] is True and closeout["decisions_agree"] is True
    assert closeout["independent_audit_failures"] in (None, [])
    report = read(output / "self-check" / "report.json")
    assert report["passed"] is True, report["failures"]
    landing = report["checks"]["final"]["reported_not_gated"]["v5_landing"]
    assert landing["candidate_with_diagnostics"] == 120
    assert read(output / "producer-outcome.json")["outcome"] == outcome
    counts = {s: len(list((output / s / "records").glob("*.json"))) for s in sr.STAGES}
    assert counts == {"calibration": 48, "final": 240, "serving": 50}
    record = read(next((output / "final" / "records").glob("final-incumbent-*.json")))
    assert record["wrapper"] == sr.INCUMBENT_METHOD and record["safety_veto"] is True
    assert (output / "receipt.json").is_file() == (outcome == "STRICT_PASS")
    if outcome == "STRICT_PASS":
        receipt = read(output / "receipt.json")
        assert receipt["incumbent"]["wrapper"] == sr.INCUMBENT_METHOD
        assert receipt["candidate"]["wrapper"] == sr.CANDIDATE_METHOD
    assert closeout["promotion_performed"] is False
    assert closeout["serving_path_qualified"] is False
    assert len(read(output / "started.json")["cpu_slot_locks_held"]) == sr.WORKERS


# ---------------------------------------------------------------- smoke dry-run audit


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A smoke through the real pipeline (fake 500-frame legacy episodes), once per module."""
    monkeypatch = pytest.MonkeyPatch()
    from src.scripts import tournament_eval

    monkeypatch.setattr(tournament_eval, "rollout", helpers._raise)
    monkeypatch.setattr(dev_screen, "run_episode", helpers._raise)
    try:
        base = tmp_path_factory.mktemp("smoke")
        closeout, output, intent, _ = run_pipeline(base, monkeypatch, 0.0, smoke_frames=500)
    finally:
        monkeypatch.undo()
    assert closeout["outcome"] == "SMOKE_NO_DECISION", closeout
    assert intent["audit"]["independent_command"][-1] == "--smoke"
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
    assert closeout["audit_passed"] is True and closeout["failure"] is None
    assert not (output / "producer-outcome.json").exists()
    records = sorted(p.name for p in (output / "final" / "records").glob("*.json"))
    assert len(records) == 2
    assert read(output / "self-check" / "report.json")["passed"] is True
    assert smoke_audit(smoke_run, tmp_path) == []
    full = helpers.SA.run_audit(smoke_run.resolve(), (tmp_path / "full").resolve(), tmp_path)
    assert "mode.not_smoke" in [row["rule"] for row in full["failures"]]


def clone_smoke(smoke_run: Path, tmp_path: Path) -> Path:
    target = tmp_path / "smoke"
    shutil.copytree(smoke_run, target, ignore=shutil.ignore_patterns("audit", "closeout.json"))
    return target


def mutate_record(root: Path, arm: str, fn) -> None:
    """Edit one smoke record and re-list its hash, so only record rules can fail."""
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
    "candidate_wrapper_v2": ("candidate", lambda e: e.update(wrapper=sr.INCUMBENT_METHOD)),
    "incumbent_probe_removed": ("incumbent", lambda e: e["record"]["probes"].pop("safety_veto")),
    "incumbent_probe_v5": (
        "incumbent",
        lambda e: e["record"]["probes"]["safety_veto"].update(method=sr.CANDIDATE_METHOD),
    ),
    "incumbent_diagnostics": ("incumbent", lambda e: e.update(veto_diagnostics={})),
    "wrong_seed": ("incumbent", lambda e: e["record"].update(seed=e["record"]["seed"] + 1)),
    "wrong_roster": ("incumbent", lambda e: e.update(roster_member_sha256s=["0" * 64] * 5)),
    "wrong_hero": ("candidate", lambda e: e.update(hero_sha256="0" * 64)),
    "frames_over_cap": ("incumbent", lambda e: e["record"].update(frames_completed=501)),
    "veto_counter_over_cap": (
        "candidate",
        lambda e: e["record"]["probes"]["safety_veto"]["counters"].update(vetoes_to_boost=501),
    ),
    "negative_mass": ("candidate", lambda e: e["record"].update(mass_integral=-1.0)),
}


@pytest.mark.parametrize("mutation", sorted(SMOKE_RECORD_MUTATIONS))
def test_smoke_audit_rejects_mutated_smoke_records(smoke_run, tmp_path, mutation):
    root = clone_smoke(smoke_run, tmp_path)
    arm, fn = SMOKE_RECORD_MUTATIONS[mutation]
    mutate_record(root, arm, fn)
    assert "smoke.records_valid" in smoke_audit(root, tmp_path)


SMOKE_BINDING_MUTATIONS = {
    "candidate_sha": ("candidate", lambda e: e.update(wrapper_source_sha256="e" * 64)),
    "candidate_v3_sha": (
        "candidate",
        lambda e: e["wrapper_source_sha256s"].update({sr.V3_SOURCE: "e" * 64}),
    ),
    "candidate_landing_rule": (
        "candidate",
        lambda e: e["record"]["probes"]["safety_veto"].update(landing_rule="x"),
    ),
    "incumbent_rule": (
        "incumbent",
        lambda e: e["record"]["probes"]["safety_veto"].update(replacement_rule="x"),
    ),
}


@pytest.mark.parametrize("mutation", sorted(SMOKE_BINDING_MUTATIONS))
def test_smoke_audit_binds_records_to_the_intent(smoke_run, tmp_path, mutation):
    root = clone_smoke(smoke_run, tmp_path)
    arm, fn = SMOKE_BINDING_MUTATIONS[mutation]
    mutate_record(root, arm, fn)
    assert "identity.arm_records_bound" in smoke_audit(root, tmp_path)


def test_smoke_audit_binds_sources_and_rejects_tampering(smoke_run, tmp_path):
    root = clone_smoke(smoke_run, tmp_path / "a")
    intent = read(root / "intent.json")
    key = next(k for k in intent["source_closure"]["files"] if k.endswith("safety_veto_v5.py"))
    intent["source_closure"]["files"][key] = "d" * 64
    helpers.write_json(root / "intent.json", intent)
    assert "identity.wrapper_source_bound_to_closure" in smoke_audit(root, tmp_path / "a")
    root = clone_smoke(smoke_run, tmp_path / "b")
    record = next((root / "output" / "final" / "records").glob("*.json"))
    record.write_text(record.read_text() + " ", encoding="utf-8")
    assert "shards.plan_assignment_and_hashes" in smoke_audit(root, tmp_path / "b")
    root = clone_smoke(smoke_run, tmp_path / "c")
    helpers.write_json(root / "output" / "producer-outcome.json", {"outcome": "STRICT_PASS"})
    assert "smoke.no_decision_claims" in smoke_audit(root, tmp_path / "c")
    root = clone_smoke(smoke_run, tmp_path / "d")
    intent = read(root / "intent.json")
    intent["smoke_frames"] = None
    helpers.write_json(root / "intent.json", intent)
    assert "smoke.intent" in smoke_audit(root, tmp_path / "d")
