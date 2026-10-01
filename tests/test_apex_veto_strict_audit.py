"""Tests for research/apex_veto_strict_20260927/strict_audit.py.

Positive per-record rules run on unmodified real Tier-1 screen records copied to
tests/fixtures/apex_veto_strict/records (provenance.json lists their sha256).
End-to-end runs are built from those real records, relabelled onto the strict
namespaces (world index residue mod 4 preserved, so the balanced roster matches),
with the producer's statistics computed by the frozen eval_stats reducer.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

import pytest

from research.apex_safety_20260926 import dev_screen
from research.apex_veto_strict_20260927 import strict_run as R
from src.evaluation.strict_promotion import _expected_world_identity
from src.scripts import eval_stats

REPO = Path(__file__).resolve().parents[1]
AUDIT_PATH = REPO / "research" / "apex_veto_strict_20260927" / "strict_audit.py"
FIXTURES = REPO / "tests" / "fixtures" / "apex_veto_strict"
REAL_SCREEN = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/apex-safety-screen-20260926/run-v1"
)
PYTHON = sys.executable


def _load_audit_module():
    spec = importlib.util.spec_from_file_location("strict_audit_under_test", AUDIT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


SA = _load_audit_module()
MIXES = SA.MIXES
ARM_NAME = {"A": "incumbent", "B": "candidate"}


def real_records() -> Dict[str, Dict[str, Any]]:
    return {
        path.name: json.loads(path.read_text())
        for path in sorted((FIXTURES / "records").glob("*.json"))
    }


REAL = real_records()


def template(arm: str, mix: str, index: int) -> Dict[str, Any]:
    """Real record for (arm A/B, mix) whose world_index has the same residue mod 4."""
    for entry in REAL.values():
        if entry["arm"] == arm and entry["mix"] == mix and entry["world_index"] % 4 == index % 4:
            return copy.deepcopy(entry)
    raise KeyError((arm, mix, index))


def relabel(
    arm: str,
    mix: str,
    index: int,
    seed: int,
    stage: Optional[str],
    mass: Optional[float] = None,
    alive: Optional[int] = None,
    canonical_arm: bool = True,
) -> Dict[str, Any]:
    """Move a real record onto another world seed / bank position (shape unchanged)."""
    entry = template(arm, mix, index)
    entry["world_seed"] = seed
    entry["world_index"] = index
    entry["record"]["seed"] = seed
    entry["record"]["world_identity"]["seed"] = seed
    if canonical_arm:
        entry["arm"] = ARM_NAME[arm]
    if stage is not None:
        entry["stage"] = stage
        entry["schema_version"] = "apex-veto-strict/v1"
    if mass is not None:
        entry["record"]["mass_integral"] = mass
    if alive is not None:
        den = entry["record"]["denominators"]
        den["alive_frames"] = alive
        den["decision_frames"] = min(alive + 1, SA.HORIZON)
        entry["record"]["survival_fraction"] = alive / SA.HORIZON
        veto = entry["record"]["probes"].get("safety_veto")
        if veto is not None:
            veto["counters"]["decisions"] = den["decision_frames"]
    return entry


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=1) + "\n")


def build_pilot(root: Path, delta_sd: float, shift: float = 40.0, rng_seed: int = 7) -> Path:
    """Screen-shaped pilot (arms A/B, 40 worlds/mix on the screen recipe seeds)."""
    rng = random.Random(rng_seed)
    bank = [SA.uint32_seed(SA.SCREEN_DOMAIN, "worlds", i) for i in range(SA.SCREEN_COUNT)]
    for mix in MIXES:
        for index, seed in enumerate(bank):
            base = 150.0 + rng.uniform(-30, 30)
            delta = shift + rng.gauss(0.0, delta_sd)
            for arm, mass in (("A", base), ("B", max(0.0, base + delta))):
                entry = relabel(arm, mix, index, seed, None, mass=mass, canonical_arm=False)
                write_json(root / "records" / f"{arm}-{mix}-{seed}.json", entry)
    return root


def pilot_deltas(root: Path) -> Dict[str, List[float]]:
    rows: Dict[tuple, Dict[str, Any]] = {}
    for path in (root / "records").glob("*.json"):
        entry = json.loads(path.read_text())
        if entry["arm"] in ("A", "B"):
            rows[(entry["arm"], entry["mix"], entry["world_index"])] = entry["record"]
    return {
        mix: [
            rows[("B", mix, i)]["mass_integral"] - rows[("A", mix, i)]["mass_integral"]
            for i in range(SA.SCREEN_COUNT)
        ]
        for mix in MIXES
    }


CHAMPION = SA.CHAMPION_SHA256
SAVED = Path("/Users/josenunez/Projects/ml/snake-dqn/saved_snakes")
POOL_FILES = {sha: SAVED / name for name, sha in dev_screen.POOL}
HAVE_CHECKPOINTS = all(path.is_file() for path in POOL_FILES.values())
needs_checkpoints = pytest.mark.skipif(not HAVE_CHECKPOINTS, reason="pool checkpoints absent")


def serving_entry_template(index: int, seed: int) -> Dict[str, Any]:
    """Real candidate record on the serving-selfplay roster (producer identity path)."""
    row = R.serving_rows([seed])[0]
    entry = relabel("B", "scripted", index, seed, "serving")
    entry.update({"mix": R.SERVING_MIX, "roster_member_sha256s": [CHAMPION] * 5})
    entry["record"]["world_identity"] = _expected_world_identity(row)
    return entry


def _episode_record(
    episode: Mapping[str, Any],
    row: Mapping[str, Any],
    index: int,
    masses: Mapping[tuple, float],
    alive_offset: int,
) -> Dict[str, Any]:
    """A real rollout record moved onto this episode's world (world identity via producer)."""
    arm, mix, seed = episode["arm"], episode["mix"], episode["world_seed"]
    if episode["stage"] == "serving":
        return serving_entry_template(index, seed)["record"]
    template_arm = "A" if arm == "incumbent" else "B"
    alive = None
    if arm == "candidate":
        base_alive = template("A", mix, index)["record"]["denominators"]["alive_frames"]
        alive = max(1, min(SA.HORIZON, base_alive + alive_offset))
    entry = relabel(template_arm, mix, index, seed, None, masses[(arm, mix, seed)], alive)
    entry["record"]["world_identity"] = _expected_world_identity(row)
    return entry["record"]


def build_run(
    tmp: Path,
    pilot: Path,
    *,
    final_shift: float = 60.0,
    final_sd: float = 30.0,
    candidate_alive_offset: int = 150,
    final_complete: bool = True,
    serving_complete: bool = True,
    claim: Optional[str] = None,
    rng_seed: int = 11,
) -> Path:
    """A run tree written through strict_run's own producer functions.

    Mirrors strict_run.run_stages: intent, snapshots + admitted.json, rosters.json, per
    stage the j % 2 shard plan, episode_entry envelopes and shard reports, then
    calibration.json, decision.json and the pre-audit producer-outcome.json claim.
    """
    rng = random.Random(rng_seed)
    root = tmp / "run-v1"
    output = root / "output"
    seeds = R.namespace_seeds()
    deltas = pilot_deltas(pilot)
    sizing = R.frozen_final_n(deltas)
    n_final = sizing["final_worlds_per_mix"]
    final_seeds = seeds["final"][: min(n_final, R.N_MAX)]
    wrapper = R.wrapper_identity()
    intent = {
        "schema_version": R.SCHEMA,
        "output_root": str(root),
        "profile": {"name": R.PROFILE_NAME, "digest": SA.PROFILE_DIGEST, "horizon": R.HORIZON},
        "checkpoint_pool": [{"name": n, "sha256": s} for n, s in dev_screen.POOL],
        "incumbent": {"checkpoint_sha256": CHAMPION, "name": R.CHAMPION[0], "wrapper": None},
        "candidate": {
            "checkpoint_sha256": CHAMPION,
            "name": R.CHAMPION[0],
            "wrapper": wrapper["method"],
            "wrapper_identity": wrapper,
            "wrapper_source_sha256": wrapper["source_sha256"],
        },
        "namespaces": {
            name: {"domain": R.NAMESPACES[name][0], "namespace": "worlds", "seeds": values}
            for name, values in seeds.items()
        },
        "mixes": list(MIXES),
        "pilot": {"screen_run": str(pilot), "deltas_by_mix": deltas, "summary_deltas_match": True},
        "sizing": sizing,
        "final_seeds": final_seeds,
        "smoke_frames": None,
        "serving_path_qualified": False,
        "source_closure": {"files": {str(R.WRAPPER_SOURCE.resolve()): wrapper["source_sha256"]}},
    }
    write_json(root / "intent.json", intent)
    if not sizing["feasible"]:
        write_json(output / "closeout.json", {"outcome": claim or "STOP_INFEASIBLE"})
        return root
    snapshots = {}
    for sha, source in POOL_FILES.items():
        target = output / "checkpoints" / f"{sha}.pth"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source)
        snapshots[sha] = str(target)
    rosters = R.build_rosters(seeds["dev"], final_seeds, seeds["serving"])
    write_json(output / "rosters.json", rosters)
    write_json(output / "admitted.json", {"checkpoint_snapshots": snapshots})
    index_maps = R.world_index_maps(intent)
    masses: Dict[tuple, float] = {}
    for mix in MIXES:
        for seed in seeds["dev"]:
            masses[("incumbent", mix, seed)] = 120 + rng.uniform(0, 60)
        for seed in final_seeds:
            base = 120 + rng.uniform(0, 60)
            masses[("incumbent", mix, seed)] = base
            masses[("candidate", mix, seed)] = max(0.0, base + final_shift + rng.gauss(0, final_sd))
    calibration = None
    decision = None
    for stage in R.STAGES:
        rows = rosters[stage]
        by_key = {(row["mix"], int(row["world_seed"])): row for row in rows}
        units = R.stage_plan(stage, rows)
        entries: Dict[str, Dict[str, Any]] = {}
        stop = (stage == "final" and not final_complete) or (
            stage == "serving" and not serving_complete
        )
        for shard in range(R.WORKERS):
            planned = [e for unit in R.shard_units(units, shard) for e in unit]
            done: Dict[str, str] = {}
            for position, episode in enumerate(planned):
                if stop and shard == 1 and position >= len(planned) - 2:
                    break
                seed = episode["world_seed"]
                row = by_key[(episode["mix"], seed)]
                index = index_maps[stage][seed]
                record = _episode_record(episode, row, index, masses, candidate_alive_offset)
                entry = R.episode_entry(episode, row, record, wrapper, shard, 1.0, index)
                path = output / stage / "records" / f"{episode['episode_id']}.json"
                write_json(path, entry)
                done[episode["episode_id"]] = hashlib.sha256(path.read_bytes()).hexdigest()
                entries[episode["episode_id"]] = entry
            complete = len(done) == len(planned)
            report = {
                "schema_version": R.SCHEMA,
                "stage": stage,
                "shard": shard,
                "slot": shard + 1,
                "planned_episode_ids": [e["episode_id"] for e in planned],
                "records_sha256": done,
                "complete": complete,
                "stopped_reason": None if complete else "deadline: 12s left < 45s budget",
            }
            write_json(output / stage / f"shard-{shard}" / "report.json", report)
        if stop:
            closeout = {"outcome": claim or "INCOMPLETE", "deadline_stop": f"{stage}: deadline"}
            write_json(output / "closeout.json", closeout)
            return root
        if stage == "calibration":
            calibration = R.calibration_reference(entries)
            write_json(output / "calibration.json", calibration)
        if stage == "final":
            paired = R.paired_final(entries, rows)
            decision = dev_screen.json_safe(R.final_decision(paired, calibration))
            write_json(output / "decision.json", decision)
    producer = R.producer_claim({"decision": decision})
    if claim is not None:
        producer["outcome"] = claim
    write_json(output / "producer-outcome.json", {**producer, "role": "pre-audit claim"})
    (output / "audit").mkdir()
    (output / "audit" / "child.log").write_text("")
    return root


def run_cli(root: Path, pilot: Path, out: Optional[Path] = None) -> tuple:
    out = out or root / "output" / "audit"
    command = [PYTHON, "-I", str(AUDIT_PATH), "--root", str(root), "--out", str(out)]
    command += ["--pilot-root", str(pilot)]
    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    audit = json.loads((out / "audit.json").read_text()) if (out / "audit.json").is_file() else {}
    return proc.returncode, audit, proc


def run_inline(root: Path, pilot: Path) -> Dict[str, Any]:
    return SA.run_audit(root.resolve(), (root / "output" / "audit").resolve(), pilot.resolve())


def failed_rules(audit: Dict[str, Any]) -> List[str]:
    return [row["rule"] for row in audit.get("failures", [])]


# ----------------------------------------------------------------------------- statistics


@pytest.mark.parametrize("df", [1, 2, 3, 7, 15, 39, 47, 120, 234, 299])
def test_student_t_matches_eval_stats(df: int) -> None:
    for t in (-6.0, -1.5, -0.01, 0.001, 0.3, 1.0, 1.7, 2.5, 4.0, 7.7, 12.0, 30.0):
        ours, ref = SA.t_sf(t, df), eval_stats.student_t_sf(t, df)
        assert abs(ours - ref) <= 1e-11 * ref
    for p in (0.2, 0.05, 0.05 / 3, 0.025):
        assert SA.close(SA.t_isf(p, df), eval_stats.student_t_isf(p, df))


def test_tiny_t_exception_is_narrow() -> None:
    ours, ref = SA.t_sf(1e-6, 299), eval_stats.student_t_sf(1e-6, 299)
    assert not SA.close(ours, ref)  # eval_stats' cancellation at |t| ~ 1e-6
    assert SA.p_values_agree(ours, ref, 1e-6)
    assert not SA.p_values_agree(0.4, 0.5, 1e-6)
    assert not SA.p_values_agree(0.049, 0.051, 1.7)


def _random_deltas(rng: random.Random, n: int, shift: float) -> Dict[str, List[float]]:
    return {m: [shift + rng.gauss(0, 50) for _ in range(n)] for m in MIXES}


@pytest.mark.parametrize("case", range(40))
def test_strict_decision_matches_eval_stats(case: int) -> None:
    rng = random.Random(case)
    n = rng.choice([2, 3, 5, 40, 41, 235])
    deltas = _random_deltas(rng, n, rng.choice([-20.0, 0.0, 5.0, 15.0, 40.0]))
    if case % 10 == 0:
        deltas["frozen"] = [3.0] * n  # zero variance, positive mean
    if case % 10 == 1:
        deltas["scripted"] = [0.0] * n  # zero variance, zero mean
    margin = rng.uniform(1.0, 6.0)
    ref = eval_stats.strict_promotion_decision(deltas, "scripted", margin)
    ours = SA.strict_decision(deltas, margin)
    assert SA._compare_statistics("case", ref, ours) == []
    assert ours["passes"] is ref["passes"]


def test_compare_statistics_detects_tampering() -> None:
    deltas = _random_deltas(random.Random(3), 40, 30.0)
    ref = eval_stats.strict_promotion_decision(deltas, "scripted", 4.0)
    ours = SA.strict_decision(deltas, 4.0)
    for mutate in (
        lambda s: s["superiority"]["per_mix"]["mixed"].__setitem__("p_value", 0.01),
        lambda s: s["superiority"]["adjusted_p_values"].__setitem__("frozen", 0.5),
        lambda s: s["scripted_noninferiority"].__setitem__("lower_bound", 1.0),
        lambda s: s.__setitem__("passes", not s["passes"]),
        lambda s: s["superiority"].__setitem__("successful_mixes", []),
    ):
        bad = copy.deepcopy(ref)
        mutate(bad)
        assert SA._compare_statistics("x", bad, ours)


@pytest.mark.parametrize("case", range(12))
def test_pilot_sizing_matches_eval_stats(case: int) -> None:
    rng = random.Random(100 + case)
    deltas = {m: [rng.gauss(10, rng.choice([5, 40, 90, 150])) for _ in range(40)] for m in MIXES}
    ref = eval_stats.paired_delta_pilot_size(deltas, {m: 20.0 for m in MIXES})
    ours = SA.pilot_sizing(deltas)
    assert ours["required_final_worlds"] == ref["required_final_worlds"]
    for m in MIXES:
        assert ours["per_mix"][m]["recommended_n"] == ref["per_mix"][m]["recommended_n"]


@pytest.mark.skipif(not (REAL_SCREEN / "records").is_dir(), reason="Tier-1 screen not present")
def test_real_screen_pilot_sizes_235() -> None:
    audit = SA.Audit()
    deltas = SA.load_pilot(audit, REAL_SCREEN)
    assert audit.failures() == [] and deltas is not None
    summary = json.loads((REAL_SCREEN / "summary.json").read_text())
    for m in MIXES:
        assert deltas[m] == summary["per_mix"][m]["primary_mass_integral"]["deltas_B_minus_A"]
    sizing = SA.pilot_sizing(deltas)
    ref = eval_stats.paired_delta_pilot_size(deltas, {m: 20.0 for m in MIXES})
    assert sizing["required_final_worlds"] == ref["required_final_worlds"] == 235


# ----------------------------------------------------------------------------- worlds


def test_anchor_hashes_and_rosters_match_strict_promotion() -> None:
    from src.evaluation.strict_promotion import (
        _expected_world_identity,
        materialize_rosters,
        scripted_agent,
    )

    for name in ("greedy_food", "random_safe"):
        assert SA.scripted_anchor_sha(name) == scripted_agent(name)["sha256"]
    anchors = [scripted_agent("greedy_food"), scripted_agent("random_safe")]
    pool = [{"kind": "checkpoint", "sha256": sha} for sha in SA.POOL_SHA256S]
    rules = [
        {
            "slot": slot,
            "eligible_member_sha256s": (
                list(SA.POOL_SHA256S) if slot % 2 else [anchors[1]["sha256"]]
            ),
        }
        for slot in range(1, 6)
    ]
    for stage in ("calibration", "final"):
        bank = SA.stage_bank(stage)
        rows = materialize_rosters(
            final_world_seeds=bank,
            roster_width=5,
            checkpoint_pool=pool,
            scripted_anchors=anchors,
            mixed_slot_rules=rules,
        )
        index = {seed: i for i, seed in enumerate(bank)}
        for row in rows:
            expected = SA.expected_world_identity(
                row["mix"], row["world_seed"], index[row["world_seed"]]
            )
            assert expected == _expected_world_identity(row)


def test_namespaces_fresh_and_match_dev_screen_recipe() -> None:
    earlier = SA.earlier_namespaces()
    for name, seeds in dev_screen.challenger_namespaces().items():
        assert earlier[f"challenger/{name}"] == seeds
    assert earlier[SA.SCREEN_DOMAIN] == dev_screen.screen_seeds(R.SCREEN_PREFIX_CHECKED)
    smoke = [dev_screen.uint32_seed(R.SMOKE_DOMAIN, "worlds", i) for i in range(16)]
    assert earlier[SA.SMOKE_DOMAIN] == smoke
    producer = R.namespace_seeds()
    assert {"calibration": producer["dev"], "final": producer["final"]} == {
        stage: SA.stage_bank(stage) for stage in ("calibration", "final")
    }
    assert producer["serving"] == SA.stage_bank("serving")
    audit = SA.Audit()
    SA.audit_namespaces(audit)
    assert audit.failures() == []
    for stage in SA.STAGES:
        report = dev_screen.disjointness_report(SA.stage_bank(stage), None)
        assert report["disjoint"]
        assert not set(SA.stage_bank(stage)) & set(dev_screen.screen_seeds(40))


# ----------------------------------------------------------------------------- per-record rules

SCREEN_BANK = [SA.uint32_seed(SA.SCREEN_DOMAIN, "worlds", i) for i in range(SA.SCREEN_COUNT)]


def check(entry: Dict[str, Any], stage: str = "final") -> List[str]:
    return SA.validate_entry(stage, "x", entry, bank=SCREEN_BANK)[1]


@pytest.mark.parametrize("name", sorted(REAL))
def test_real_records_pass_every_record_rule(name: str) -> None:
    entry = REAL[name]
    assert check(entry) == []
    if entry["arm"] == "A":
        assert check(entry, "calibration") == []
    episode, _ = SA.validate_entry("final", name, entry, bank=SCREEN_BANK)
    assert episode["arm"] == ARM_NAME[entry["arm"]]


def test_fixture_provenance_is_intact() -> None:
    provenance = json.loads((FIXTURES / "provenance.json").read_text())
    for name, digest in provenance["sha256"].items():
        assert hashlib.sha256((FIXTURES / "records" / name).read_bytes()).hexdigest() == digest


def test_uninstalled_counters_are_not_required_but_bounded() -> None:
    entry = copy.deepcopy(REAL["A-frozen-545722568.json"])
    assert "optimizer_updates" not in json.dumps(entry)  # real rollout records lack it
    entry["counters"] = {"lifecycle_frames": 0, "frames_completed": 5000, "optimizer_updates": 0}
    entry["counters"].update({"frame_attempts": 5000, "parity_frames": 0})
    assert check(entry) == []


def _mutations() -> Dict[str, Callable[[Dict[str, Any]], None]]:
    def rec(e: Dict[str, Any]) -> Dict[str, Any]:
        return e["record"]

    return {
        "optimizer_updates": lambda e: e.__setitem__("optimizer_updates", 1),
        "nested_optimizer": lambda e: e.__setitem__("counters", {"optimizer_updates": 2}),
        "frames_completed_short": lambda e: e.__setitem__("frames_completed", 4999),
        "lifecycle_over_cap": lambda e: e.__setitem__("counters", {"lifecycle_frames": 5001}),
        "scored_frames": lambda e: rec(e)["denominators"].__setitem__("scored_frames", 4000),
        "decision_over_cap": lambda e: rec(e)["denominators"].__setitem__("decision_frames", 5001),
        "survival_mismatch": lambda e: rec(e).__setitem__("survival_fraction", 0.99),
        "negative_mass": lambda e: rec(e).__setitem__("mass_integral", -1.0),
        "hero": lambda e: e.__setitem__("hero_sha256", SA.POOL_SHA256S[1]),
        "profile_digest": lambda e: rec(e).__setitem__("evaluation_profile_digest", "0" * 64),
        "learn": lambda e: rec(e)["evaluation_profile"].__setitem__("learn", True),
        "training": lambda e: rec(e)["evaluation_profile"]["runtime"].__setitem__("training", True),
        "roster": lambda e: rec(e)["world_identity"]["ordered_slot_content_hashes"].reverse(),
        "roster_id": lambda e: rec(e)["world_identity"].__setitem__("roster_id", "1" * 64),
        "seed": lambda e: rec(e).__setitem__("seed", 12345),
        "index": lambda e: e.__setitem__("world_index", e["world_index"] + 1),
        "arm": lambda e: e.__setitem__("arm", "C"),
        "stage_field": lambda e: e.__setitem__("stage", "serving"),
    }


@pytest.mark.parametrize("mutation", sorted(_mutations()))
@pytest.mark.parametrize("name", ["A-frozen-545722568.json", "B-mixed-2400042722.json"])
def test_mutated_real_records_fail(mutation: str, name: str) -> None:
    entry = copy.deepcopy(REAL[name])
    _mutations()[mutation](entry)
    assert check(entry)


def test_wrapper_identity_rules() -> None:
    inc = copy.deepcopy(REAL["A-scripted-545722568.json"])
    cand = copy.deepcopy(REAL["B-scripted-545722568.json"])
    assert check(inc) == [] and check(cand) == []
    probe = copy.deepcopy(cand["record"]["probes"]["safety_veto"])
    inc["record"]["probes"]["safety_veto"] = probe
    assert check(inc)  # incumbent must not carry the wrapper
    old = copy.deepcopy(cand)
    old["record"]["probes"]["safety_veto"]["method"] = "free-space-veto/v1"
    assert check(old)
    missing = copy.deepcopy(cand)
    del missing["record"]["probes"]["safety_veto"]
    assert check(missing)
    flag = copy.deepcopy(cand)
    flag["safety_veto"] = False
    assert check(flag)
    assert check(copy.deepcopy(cand), "calibration")  # calibration is incumbent-only


def test_serving_rules_selfplay_roster_on_serving_bank() -> None:
    bank = SA.stage_bank("serving")
    entry = serving_entry_template(3, bank[3])
    assert SA.validate_entry("serving", "s", entry)[1] == []
    stray = serving_entry_template(0, SA.stage_bank("final")[0])
    assert SA.validate_entry("serving", "s", stray)[1]
    scripted = relabel("B", "scripted", 3, bank[3], "serving")
    assert SA.validate_entry("serving", "s", scripted)[1]  # serving roster is self-play
    inc = serving_entry_template(3, bank[3])
    inc.update({"arm": "incumbent", "safety_veto": False})
    del inc["record"]["probes"]["safety_veto"]
    assert SA.validate_entry("serving", "s", inc)[1]


# ----------------------------------------------------------------------------- end to end


@pytest.fixture(scope="module")
def small_pilot(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Pilot whose paired SD sizes the final at the 40-world floor."""
    pilot = build_pilot(tmp_path_factory.mktemp("pilot") / "screen", delta_sd=10.0)
    assert R.frozen_final_n(pilot_deltas(pilot))["final_worlds_per_mix"] == 40
    return pilot


@pytest.fixture(scope="module")
def pass_run(tmp_path_factory: pytest.TempPathFactory, small_pilot: Path) -> Path:
    if not HAVE_CHECKPOINTS:
        pytest.skip("pool checkpoints absent")
    return build_run(tmp_path_factory.mktemp("pass"), small_pilot)


def clone(run: Path, tmp_path: Path) -> Path:
    target = tmp_path / "run-v1"
    shutil.copytree(run, target, symlinks=True)
    return target


def edit_json(path: Path, fn: Callable[[Any], None]) -> None:
    value = json.loads(path.read_text())
    fn(value)
    path.write_text(json.dumps(value, sort_keys=True, indent=1) + "\n")


def final_record(run: Path, arm: str, mix: str, index: int) -> Path:
    seed = SA.stage_bank("final")[index]
    return run / "output" / "final" / "records" / f"final-{arm}-{mix}-{seed}.json"


def test_cli_pass_is_isolated_and_create_only(
    pass_run: Path, small_pilot: Path, tmp_path: Path
) -> None:
    run = clone(pass_run, tmp_path)
    code, audit, proc = run_cli(run, small_pilot)
    assert code == 0, (proc.stdout, failed_rules(audit))
    assert audit["status"] == "PASS"
    assert audit["producer_outcome"] == "STRICT_PASS"
    assert audit["audit_expected_outcomes"] == ["STRICT_PASS"]
    assert audit["final"]["n_final"] == 40 and audit["serving"]["complete"]
    assert json.loads(proc.stdout)["status"] == "PASS"
    assert "output/audit/child.log" not in audit["evidence_sha256"]
    code, _, proc = run_cli(run, small_pilot)  # audit.json already exists
    assert code == 1 and "cannot write audit" in proc.stdout


@needs_checkpoints
def test_strict_fail_is_audited_as_consistent(small_pilot: Path, tmp_path: Path) -> None:
    run = build_run(tmp_path, small_pilot, final_shift=-5.0)
    audit = run_inline(run, small_pilot)
    assert audit["status"] == "PASS", failed_rules(audit)
    assert audit["producer_outcome"] == "STRICT_FAIL"


@needs_checkpoints
def test_band_failure_forces_strict_fail(small_pilot: Path, tmp_path: Path) -> None:
    run = build_run(tmp_path, small_pilot, candidate_alive_offset=-4000)
    audit = run_inline(run, small_pilot)
    assert audit["status"] == "PASS", failed_rules(audit)
    assert audit["producer_outcome"] == "STRICT_FAIL"
    assert not audit["final"]["bands_pass"] and audit["final"]["decision"]["passes"]


@needs_checkpoints
def test_overclaimed_pass_fails(small_pilot: Path, tmp_path: Path) -> None:
    run = build_run(tmp_path, small_pilot, final_shift=-5.0, claim="STRICT_PASS")
    assert "claims.outcome" in failed_rules(run_inline(run, small_pilot))


@needs_checkpoints
@pytest.mark.parametrize("stage", ["final", "serving"])
def test_deadline_stop_is_incomplete(stage: str, small_pilot: Path, tmp_path: Path) -> None:
    flags = {"final_complete": stage != "final", "serving_complete": stage != "serving"}
    run = build_run(tmp_path / "a", small_pilot, **flags)
    audit = run_inline(run, small_pilot)
    assert audit["status"] == "PASS", failed_rules(audit)
    assert audit["producer_outcome"] == "INCOMPLETE"
    over = build_run(tmp_path / "b", small_pilot, claim="STRICT_PASS", **flags)
    assert "claims.outcome" in failed_rules(run_inline(over, small_pilot))


def test_stop_infeasible(tmp_path: Path) -> None:
    pilot = build_pilot(tmp_path / "wide", delta_sd=150.0)
    run = build_run(tmp_path / "a", pilot)
    audit = run_inline(run, pilot)
    assert audit["status"] == "PASS", failed_rules(audit)
    assert audit["producer_outcome"] == "STOP_INFEASIBLE"
    assert audit["sizing"]["required_final_worlds"] > 300
    stray = run / "output" / "final" / "records" / "stray.json"
    write_json(stray, relabel("A", "frozen", 0, SA.stage_bank("final")[0], "final"))
    rules = failed_rules(run_inline(run, pilot))
    assert "final.absent_when_infeasible" in rules


def _set(path: List[Any], value: Any) -> Callable[[Any], None]:
    def mutate(doc: Any) -> None:
        for key in path[:-1]:
            doc = doc[key]
        doc[path[-1]] = value(doc[path[-1]]) if callable(value) else value

    return mutate


SPD = "strict_promotion_decision"
TAMPERS: Dict[str, tuple] = {
    "p_value": (
        "output/decision.json",
        _set([SPD, "superiority", "per_mix", "frozen", "p_value"], 0.2),
        "claims.statistics",
    ),
    "ni_lower_bound": (
        "output/decision.json",
        _set([SPD, "scripted_noninferiority", "lower_bound"], lambda v: v + 1e-6),
        "claims.statistics",
    ),
    "paired_deltas": (
        "output/decision.json",
        _set(["paired_deltas_by_mix", "mixed", 0], 999.0),
        "claims.statistics",
    ),
    "bands_pass": (
        "output/decision.json",
        _set(["bands_pass"], False),
        "claims.design_and_calibration",
    ),
    "band_value": (
        "output/decision.json",
        _set(["behavioral_bands", "frozen", "value"], 0.0),
        "claims.behavioral_bands",
    ),
    "delta_ni": (
        "output/calibration.json",
        _set(["absolute_delta_ni"], lambda v: v * 1.001),
        "claims.absolute_delta_ni",
    ),
    "band_bound": (
        "output/calibration.json",
        _set(["survival_bands", "scripted", "lower"], 0.0),
        "claims.behavioral_bands",
    ),
    "calibration_mean": (
        "output/calibration.json",
        _set(["per_mix", "frozen", "mean_mass_integral"], lambda v: v + 0.5),
        "claims.design_and_calibration",
    ),
    "required_n": (
        "intent.json",
        _set(["sizing", "pilot_sizing", "required_final_worlds"], 41),
        "claims.required_final_worlds",
    ),
    "frozen_n": (
        "intent.json",
        _set(["sizing", "final_worlds_per_mix"], 41),
        "claims.design_and_calibration",
    ),
    "final_seeds": (
        "intent.json",
        _set(["final_seeds"], lambda v: v[1:] + v[:1]),
        "claims.design_and_calibration",
    ),
    "namespace_seeds": (
        "intent.json",
        _set(["namespaces", "serving", "seeds"], lambda v: list(reversed(v))),
        "claims.design_and_calibration",
    ),
    "pilot_deltas": (
        "intent.json",
        _set(["pilot", "deltas_by_mix", "scripted", 3], 0.0),
        "claims.design_and_calibration",
    ),
    "sizing_sd": (
        "intent.json",
        _set(["sizing", "pilot_sizing", "per_mix", "mixed", "paired_delta_std"], 1.0),
        "claims.sizing",
    ),
    "wrapper_sha": (
        "intent.json",
        _set(["candidate", "wrapper_source_sha256"], "e" * 64),
        "identity.wrapper_source_sha256",
    ),
    "wrapper_identity": (
        "intent.json",
        _set(["candidate", "wrapper_identity", "method"], "free-space-veto/v1"),
        "identity.wrapper_source_sha256",
    ),
    "closure_wrapper_sha": (
        "intent.json",
        _set(["source_closure", "files", str(R.WRAPPER_SOURCE.resolve())], "d" * 64),
        "identity.wrapper_source_bound_to_closure",
    ),
    "descriptor": (
        "intent.json",
        _set(["candidate", "wrapper_identity", "descriptor", "free_space_min_cap"], 31),
        "identity.candidate_records_bound",
    ),
    "incumbent_ckpt": (
        "intent.json",
        _set(["incumbent", "checkpoint_sha256"], SA.POOL_SHA256S[2]),
        "identity.checkpoint_and_wrapper",
    ),
}


@pytest.mark.parametrize("tamper", sorted(TAMPERS))
def test_tampered_producer_claims_fail(
    tamper: str, pass_run: Path, small_pilot: Path, tmp_path: Path
) -> None:
    run = clone(pass_run, tmp_path)
    relative, mutate, rule = TAMPERS[tamper]
    edit_json(run / relative, mutate)
    assert rule in failed_rules(run_inline(run, small_pilot))


def test_missing_final_record_with_pass_claim_fails(
    pass_run: Path, small_pilot: Path, tmp_path: Path
) -> None:
    run = clone(pass_run, tmp_path)
    final_record(run, "candidate", "scripted", 7).unlink()
    rules = failed_rules(run_inline(run, small_pilot))
    assert {"claims.outcome", "claims.statistics", "shards.plan_assignment_and_hashes"} <= set(
        rules
    )


RECORD_BINDING_TAMPERS = {
    "wrapper_source_missing": lambda e: e.pop("wrapper_source_sha256"),
    "wrapper_source_other": lambda e: e.update(wrapper_source_sha256="e" * 64),
    "descriptor_bfs_cap": lambda e: e["record"]["probes"]["safety_veto"].update(
        free_space_bfs_cap=159
    ),
    "descriptor_extra_key": lambda e: e["record"]["probes"]["safety_veto"].update(extra=1),
}


@pytest.mark.parametrize("tamper", sorted(RECORD_BINDING_TAMPERS))
def test_candidate_records_are_bound_to_the_intent(
    tamper: str, pass_run: Path, small_pilot: Path, tmp_path: Path
) -> None:
    """Real candidate records carry the intent's wrapper sha and descriptor (else FAIL)."""
    audit = run_inline(clone(pass_run, tmp_path / "ok"), small_pilot)
    assert "identity.candidate_records_bound" not in failed_rules(audit)
    assert "identity.wrapper_source_bound_to_closure" not in failed_rules(audit)
    run = clone(pass_run, tmp_path / "bad")
    edit_json(final_record(run, "candidate", "mixed", 5), RECORD_BINDING_TAMPERS[tamper])
    assert "identity.candidate_records_bound" in failed_rules(run_inline(run, small_pilot))


def test_changed_record_bytes_fail(pass_run: Path, small_pilot: Path, tmp_path: Path) -> None:
    run = clone(pass_run, tmp_path)
    edit_json(
        final_record(run, "candidate", "frozen", 2),
        _set(["record", "mass_integral"], lambda v: v + 50.0),
    )
    rules = failed_rules(run_inline(run, small_pilot))
    assert "shards.plan_assignment_and_hashes" in rules and "claims.statistics" in rules


def test_extra_record_fails(pass_run: Path, small_pilot: Path, tmp_path: Path) -> None:
    run = clone(pass_run, tmp_path)
    source = final_record(run, "incumbent", "frozen", 3)
    shutil.copyfile(source, source.with_name("extra.json"))
    rules = failed_rules(run_inline(run, small_pilot))
    assert "final.records_valid" in rules and "shards.plan_assignment_and_hashes" in rules


def test_mispaired_world_identity_fails(pass_run: Path, small_pilot: Path, tmp_path: Path) -> None:
    run = clone(pass_run, tmp_path)
    edit_json(
        final_record(run, "candidate", "mixed", 5),
        lambda e: e["record"]["world_identity"]["ordered_slot_content_hashes"].reverse(),
    )
    assert "final.records_valid" in failed_rules(run_inline(run, small_pilot))


def test_nondeterministic_shard_assignment_fails(
    pass_run: Path, small_pilot: Path, tmp_path: Path
) -> None:
    run = clone(pass_run, tmp_path)
    reports = [run / "output" / "final" / f"shard-{k}" / "report.json" for k in (0, 1)]
    docs = [json.loads(path.read_text()) for path in reports]
    moved = docs[0]["planned_episode_ids"].pop()
    docs[1]["planned_episode_ids"].append(moved)
    docs[1]["records_sha256"][moved] = docs[0]["records_sha256"].pop(moved)
    for path, doc in zip(reports, docs):
        write_json(path, doc)
    assert "shards.plan_assignment_and_hashes" in failed_rules(run_inline(run, small_pilot))


def test_physical_counters_real_shape(pass_run: Path, small_pilot: Path, tmp_path: Path) -> None:
    """Strict-pilot physical-report shape: lifecycle_frames 0 is legal, over-cap is not."""
    run = clone(pass_run, tmp_path)
    counters = {"frames_completed": 240000, "lifecycle_frames": 0, "optimizer_updates": 0}
    caps = {"frames_completed": 240000, "lifecycle_frames": 240000, "optimizer_updates": 0}
    report = run / "output" / "final" / "physical-report.json"
    write_json(report, {"caps": caps, "counters": counters, "episodes": []})
    assert run_inline(run, small_pilot)["status"] == "PASS"
    edit_json(report, lambda d: d["counters"].__setitem__("optimizer_updates", 1))
    assert "counters.optimizer_updates_zero_and_at_most_cap" in failed_rules(
        run_inline(run, small_pilot)
    )


def test_artifact_listing_and_snapshot_hashes(
    pass_run: Path, small_pilot: Path, tmp_path: Path
) -> None:
    run = clone(pass_run, tmp_path)
    record = final_record(run, "incumbent", "frozen", 0)
    listing = {"path": "final/records/" + record.name, "sha256": "0" * 64}
    write_json(run / "output" / "listing.json", {"artifacts": [listing]})
    assert "artifacts.hashes" in failed_rules(run_inline(run, small_pilot))
    listing["sha256"] = hashlib.sha256(record.read_bytes()).hexdigest()
    write_json(run / "output" / "listing.json", {"artifacts": [listing]})
    assert run_inline(run, small_pilot)["status"] == "PASS"
    snapshot = run / "output" / "checkpoints" / f"{SA.POOL_SHA256S[1]}.pth"
    snapshot.unlink()
    snapshot.symlink_to(POOL_FILES[SA.POOL_SHA256S[2]])
    assert "artifacts.hashes" in failed_rules(run_inline(run, small_pilot))
    snapshot.unlink()
    assert "artifacts.hashes" in failed_rules(run_inline(run, small_pilot))


def test_incomplete_calibration_cannot_back_claims(
    pass_run: Path, small_pilot: Path, tmp_path: Path
) -> None:
    run = clone(pass_run, tmp_path)
    seed = SA.stage_bank("calibration")[4]
    (
        run / "output" / "calibration" / "records" / f"calibration-incumbent-scripted-{seed}.json"
    ).unlink()
    rules = failed_rules(run_inline(run, small_pilot))
    assert "claims.absolute_delta_ni" in rules and "final.requires_calibration" in rules


def test_record_outside_stage_dir_fails(pass_run: Path, small_pilot: Path, tmp_path: Path) -> None:
    run = clone(pass_run, tmp_path)
    source = final_record(run, "incumbent", "frozen", 0)
    write_json(run / "output" / "records" / "stray.json", json.loads(source.read_text()))
    assert "records.inside_stage_dirs" in failed_rules(run_inline(run, small_pilot))


def test_missing_pilot_fails_closed(pass_run: Path, tmp_path: Path) -> None:
    run = clone(pass_run, tmp_path)
    code, audit, _ = run_cli(run, tmp_path / "no-pilot")
    assert code == 1 and "pilot.available" in failed_rules(audit)


def test_stage_aliases_and_repo_relative_listings(
    pass_run: Path, small_pilot: Path, tmp_path: Path
) -> None:
    run = clone(pass_run, tmp_path)
    (run / "output" / "calibration").rename(run / "output" / "development")
    source = {"path": "research/apex_veto_strict_20260927/protocol.md", "sha256": "a" * 64}
    write_json(run / "output" / "source.json", {"closure": [source]})
    audit = run_inline(run, small_pilot)
    assert audit["status"] == "PASS", failed_rules(audit)
    assert audit["calibration"]["complete"]
    detail = next(r for r in audit["rules"] if r["rule"] == "artifacts.hashes")["detail"]
    assert source["path"] in detail["outside_root_not_checked"]
    missing = {"path": "final/records/gone.json", "sha256": "b" * 64}
    write_json(run / "output" / "listing.json", {"artifacts": [missing]})
    assert "artifacts.hashes" in failed_rules(run_inline(run, small_pilot))


def test_invalid_stop_needs_a_recorded_reason(
    pass_run: Path, small_pilot: Path, tmp_path: Path
) -> None:
    run = clone(pass_run, tmp_path)
    claim = run / "output" / "producer-outcome.json"
    edit_json(claim, lambda d: d.__setitem__("outcome", "INVALID_STOP"))
    assert "claims.outcome" in failed_rules(run_inline(run, small_pilot))
    edit_json(claim, lambda d: d.__setitem__("stop_reason", "RSS cap exceeded on worker 2"))
    assert run_inline(run, small_pilot)["status"] == "PASS"
    write_json(run / "output" / "closeout.json", {"outcome": "STRICT_PASS", "failure": None})
    assert "claims.outcome" in failed_rules(run_inline(run, small_pilot))  # conflicting claims
