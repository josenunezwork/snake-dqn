"""Tests for research/apex_veto_v7_strict_20261002/strict_audit.py (independent, stdlib).

Positive per-record rules run on real record shapes copied to
``tests/fixtures/apex_veto_v7_strict`` (one complete frozen A/B pair of the closed v7 Tier-1
screen: A = champion + v5, B = champion + v7 lambda=4.0; the v7 screen GO smoke pair; this
package's own plumbing-smoke pair; provenance.json lists their sha256). Runs are
built from those records moved onto the strict namespaces through strict_run's own producer
functions, with the producer's statistics from the frozen eval_stats reducer. No test here
plays an episode: ``tournament_eval.rollout``, ``dev_screen.run_episode`` and
``strict_run.run_unit_episode`` raise in every test.
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
from typing import Any, Callable, Dict, List, Optional

import pytest

from research.apex_safety_20260926 import dev_screen
from research.apex_veto_v7_strict_20261002 import strict_run as R
from src.scripts import eval_stats

REPO = Path(__file__).resolve().parents[1]
AUDIT_PATH = REPO / "research" / "apex_veto_v7_strict_20261002" / "strict_audit.py"
FIXTURES = REPO / "tests" / "fixtures" / "apex_veto_v7_strict"
PYTHON = sys.executable


def _load_audit_module():
    spec = importlib.util.spec_from_file_location("v7_strict_audit_under_test", AUDIT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


SA = _load_audit_module()
MIXES = SA.MIXES
ARM_NAME = {"A": "incumbent", "B": "candidate"}
LETTER = {"incumbent": "A", "candidate": "B"}


def _raise(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("episode runners are blocked in tests")


@pytest.fixture(autouse=True)
def block_episode_runners(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test may play an episode (the v3 revision-4 runaway lesson)."""
    from src.scripts import tournament_eval

    monkeypatch.setattr(tournament_eval, "rollout", _raise)
    monkeypatch.setattr(dev_screen, "run_episode", _raise)
    monkeypatch.setattr(R, "run_unit_episode", _raise)


def load(relative: str) -> Dict[str, Any]:
    return json.loads((FIXTURES / relative).read_text())


SCREEN = {letter: load(f"screen/{letter}-frozen-2982303806.json") for letter in ("A", "B")}
SMOKE = {letter: load(f"smoke/{letter}-scripted-1823945485.json") for letter in ("A", "B")}
STRICT_SMOKE = {
    arm: load(f"strict_smoke/final-{arm}-scripted-807620295.json")
    for arm in ("incumbent", "candidate")
}


def fix_mass_identities(record: Dict[str, Any]) -> None:
    """Keep the strict validator's identities after a mass change (mean <= max = peak)."""
    alive = record["denominators"]["alive_frames"]
    record["mean_mass_alive"] = record["mass_integral"] * SA.HORIZON / alive
    peak = max(float(record["max_mass"]), record["mean_mass_alive"])
    record["max_mass"] = record["probes"]["peak_length"] = peak


def move_record(
    letter: str,
    mix: str,
    index: int,
    seed: int,
    mass: Optional[float] = None,
    alive: Optional[int] = None,
) -> Dict[str, Any]:
    """A real screen record moved onto another world (identity rebuilt, shape unchanged)."""
    record = copy.deepcopy(SCREEN[letter]["record"])
    record["seed"] = seed
    record["world_identity"] = SA.expected_world_identity(mix, seed, index)
    if mass is not None:
        record["mass_integral"] = mass
        fix_mass_identities(record)
    if alive is not None:
        den = record["denominators"]
        den["alive_frames"] = alive
        den["decision_frames"] = min(alive + record["deaths"], SA.HORIZON)
        record["survival_fraction"] = alive / SA.HORIZON
        counters = record["probes"]["safety_veto"]["counters"]
        counters["decisions"] = den["decision_frames"]
        counters["kept_base"] = (
            counters["decisions"] - counters["vetoes_applied"] - counters["fallback_no_spacious"]
        )
        fix_mass_identities(record)
    return record


def pilot_entry(letter: str, mix: str, index: int, seed: int, mass: float) -> Dict[str, Any]:
    """A v7-screen-shaped entry (arm letters, screen schema) on the screen recipe."""
    entry = copy.deepcopy(SCREEN[letter])
    entry.update({"mix": mix, "world_index": index, "world_seed": seed})
    entry["record"] = move_record(letter, mix, index, seed, mass=mass)
    entry["roster_member_sha256s"] = entry["record"]["world_identity"][
        "ordered_slot_content_hashes"
    ]
    return entry


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=1) + "\n")


def build_pilot(
    root: Path,
    delta_sd: float,
    shift: float = 40.0,
    rng_seed: int = 7,
    decision: str = "RECOMMEND_STRICT_GATE",
    self_check_passes: bool = True,
) -> Path:
    """v7-screen-shaped pilot: A/B records, summary.json, intent.json and receipt.json.

    Also writes a DEV-sweep-shaped ``summary.json`` + ``intent.json`` beside ``root`` (see
    :func:`sweep_of`) that selected lambda 4.0 at the pilot commit, and binds it the way
    ``screen.py`` does (receipt ``lambda`` and ``sweep_binding``; intent arm B, hypothesis).
    """
    rng = random.Random(rng_seed)
    bank = SA.pilot_bank()
    deltas: Dict[str, List[float]] = {mix: [] for mix in MIXES}
    listed = []
    for mix in MIXES:
        for index, seed in enumerate(bank):
            base = 150.0 + rng.uniform(-30, 30)
            delta = shift + rng.gauss(0.0, delta_sd)
            masses = {"A": base, "B": max(0.0, base + delta)}
            deltas[mix].append(masses["B"] - masses["A"])
            for letter, mass in masses.items():
                path = root / "records" / f"{letter}-{mix}-{seed}.json"
                write_json(path, pilot_entry(letter, mix, index, seed, mass))
                listed.append({"path": f"records/{path.name}", "sha256": sha(path)})
    control = root / "records" / f"C-frozen-{bank[0]}.json"  # controls are skipped
    write_json(control, dict(pilot_entry("A", "frozen", 0, bank[0], 1.0), arm="C"))
    listed.append({"path": f"records/{control.name}", "sha256": sha(control)})
    summary = {
        "decision": decision,
        "per_mix": {m: {"primary_mass_integral": {"deltas_B_minus_A": deltas[m]}} for m in MIXES},
    }
    write_json(root / "summary.json", summary)
    sweep = build_sweep(root.parent / f"{root.name}-sweep")
    sweep_sha = sha(sweep)
    origin = f"lambda 4.0 from sweep summary sha256 {sweep_sha} (commit {R.PILOT_COMMIT})"
    screen_intent = {
        "git": {"commit": R.PILOT_COMMIT, "dirty_paths": ""},
        "arms": {"A": "champion + v5", "B": f"champion + {R.CANDIDATE_METHOD} ({origin})"},
        "hypothesis": f"Re-ranking (v7, {origin}) raises the mass integral (B - A > 0).",
    }
    write_json(root / "intent.json", screen_intent)
    receipt = {
        "lambda": 4.0,
        "sweep_binding": {
            "summary_path": str(sweep.resolve()),
            "summary_sha256": sweep_sha,
            "intent_sha256": json.loads(sweep.read_text())["intent_sha256"],
            "source_commit": R.PILOT_COMMIT,
            "selected_lambda": 4.0,
        },
        "intent_sha256": sha(root / "intent.json"),
        "decision": decision if self_check_passes else "INVALID_SELF_CHECK_FAILED",
        "screen_decision_before_self_check": decision,
        "self_check": {
            "passes": self_check_passes,
            "failures": [] if self_check_passes else ["x.json: probe"],
        },
        "summary_sha256": sha(root / "summary.json"),
        "records": listed,
    }
    write_json(root / "receipt.json", receipt)
    return root


def build_sweep(directory: Path) -> Path:
    """A DEV-sweep-shaped summary (plus its intent) that selected lambda 4.0 at the commit."""
    write_json(directory / "intent.json", {"git": {"commit": R.PILOT_COMMIT, "dirty_paths": ""}})
    summary = {
        "schema_version": R.SWEEP_SCHEMA,
        "sweep_id": R.SWEEP_ID,
        "smoke": False,
        "intent_sha256": sha(directory / "intent.json"),
        "source": {"commit": R.PILOT_COMMIT, "dirty_paths": ""},
        "selection": {"status": "SELECTED", "passes": True, "selected_lambda": 4.0},
    }
    write_json(directory / "summary.json", summary)
    return directory / "summary.json"


def sweep_of(pilot: Path) -> Path:
    """The sweep summary a (possibly copied) pilot's receipt binds (or where it would be)."""
    try:
        receipt = json.loads((Path(pilot) / "receipt.json").read_text())
        return Path(receipt["sweep_binding"]["summary_path"])
    except (OSError, KeyError, ValueError):
        return Path(pilot).parent / f"{Path(pilot).name}-sweep" / "summary.json"


def rebind_sweep(pilot: Path, mutate: Callable[[Dict[str, Any]], None]) -> None:
    """Edit a copied pilot's sweep summary (a fresh copy) and re-bind the receipt to it."""
    source = sweep_of(pilot)
    target = pilot.parent / f"{pilot.name}-sweep-edited" / "summary.json"
    shutil.copytree(source.parent, target.parent)
    edit_json(target, mutate)
    edit_json(
        pilot / "receipt.json",
        lambda r: r["sweep_binding"].update(
            summary_path=str(target.resolve()), summary_sha256=sha(target)
        ),
    )


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pilot_deltas(root: Path) -> Dict[str, List[float]]:
    rows: Dict[tuple, Dict[str, Any]] = {}
    for path in (root / "records").glob("*.json"):
        entry = json.loads(path.read_text())
        if entry["arm"] in ("A", "B"):
            rows[(entry["arm"], entry["mix"], entry["world_index"])] = entry["record"]
    return {
        mix: [
            rows[("B", mix, i)]["mass_integral"] - rows[("A", mix, i)]["mass_integral"]
            for i in range(SA.PILOT_COUNT)
        ]
        for mix in MIXES
    }


CHAMPION = SA.CHAMPION_SHA256
SAVED = Path("/Users/josenunez/Projects/ml/snake-dqn/saved_snakes")
POOL_FILES = {sha: SAVED / name for name, sha in dev_screen.POOL}
HAVE_CHECKPOINTS = all(path.is_file() for path in POOL_FILES.values())
needs_checkpoints = pytest.mark.skipif(not HAVE_CHECKPOINTS, reason="pool checkpoints absent")
# Each arm's real diagnostics kind: v5 counters (A) and v7 counters with nested v5 (B).
DIAG = {arm: SCREEN[letter]["veto_diagnostics"] for letter, arm in ARM_NAME.items()}


def closure_files() -> Dict[str, str]:
    """The veto modules' source-closure entries (absolute path -> sha256)."""
    rels = {rel for arm in R.ARMS for rel in R.ARM_SOURCES[arm]}
    return {str((REPO / rel).resolve()): sha(REPO / rel) for rel in sorted(rels)}


def arm_intent_nodes() -> Dict[str, Dict[str, Any]]:
    identities = R.arm_identities()
    return {
        arm: {
            "checkpoint_sha256": CHAMPION,
            "name": R.CHAMPION[0],
            "wrapper": identities[arm]["method"],
            "wrapper_identity": identities[arm],
            "wrapper_source_sha256": identities[arm]["source_sha256"],
        }
        for arm in R.ARMS
    }


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
    """A run tree written through strict_run's own producer functions (run_stages layout)."""
    rng = random.Random(rng_seed)
    root = tmp / "run-v1"
    output = root / "output"
    seeds = R.namespace_seeds()
    deltas = pilot_deltas(pilot)
    sizing = R.frozen_final_n(deltas)
    n_final = sizing["final_worlds_per_mix"]
    final_seeds = seeds["final"][: min(n_final, R.N_MAX)]
    nodes = arm_intent_nodes()
    wrappers = {arm: node["wrapper_identity"] for arm, node in nodes.items()}
    intent = {
        "schema_version": R.SCHEMA,
        "output_root": str(root),
        "profile": {"name": R.PROFILE_NAME, "digest": SA.PROFILE_DIGEST, "horizon": R.HORIZON},
        "checkpoint_pool": [{"name": n, "sha256": s} for n, s in dev_screen.POOL],
        **nodes,
        "namespaces": {
            name: {"domain": R.NAMESPACES[name][0], "namespace": "worlds", "seeds": values}
            for name, values in seeds.items()
        },
        "mixes": list(MIXES),
        "candidate_lambda": R.CANDIDATE_LAMBDA,
        "design": {"mde_absolute_per_mix": R.MDE_ABSOLUTE, "candidate_lambda": R.CANDIDATE_LAMBDA},
        "pilot": {
            "screen_run": str(pilot),
            "deltas_by_mix": deltas,
            "summary_deltas_match": True,
            "screen_source_parity": R.screen_source_parity(pilot, R.arm_identities()),
            "lambda_binding": R.lambda_binding_gate(pilot, sweep_of(pilot)),
        },
        "sizing": sizing,
        "final_seeds": final_seeds,
        "smoke_frames": None,
        "serving_path_qualified": False,
        "source_closure": {"files": closure_files()},
    }
    write_json(root / "intent.json", intent)
    if not sizing["feasible"]:
        write_json(output / "closeout.json", {"outcome": claim or "STOP_INFEASIBLE"})
        return root
    snapshots = {}
    for digest, source in POOL_FILES.items():
        target = output / "checkpoints" / f"{digest}.pth"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source)
        snapshots[digest] = str(target)
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
    calibration = decision = None
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
                arm, mix, seed = episode["arm"], episode["mix"], episode["world_seed"]
                row = by_key[(mix, seed)]
                index = index_maps[stage][seed]
                if stage == "serving":
                    record = move_record("B", mix, index, seed)
                else:
                    alive = None
                    if arm == "candidate":
                        base = SCREEN["A"]["record"]["denominators"]["alive_frames"]
                        alive = max(1, min(SA.HORIZON, base + candidate_alive_offset))
                    record = move_record(LETTER[arm], mix, index, seed, masses[(arm, mix, seed)])
                    if alive is not None:
                        record = move_record("B", mix, index, seed, record["mass_integral"], alive)
                diag = DIAG[arm]
                entry = R.episode_entry(episode, row, record, wrappers, shard, 1.0, index, diag)
                path = output / stage / "records" / f"{episode['episode_id']}.json"
                write_json(path, entry)
                done[episode["episode_id"]] = sha(path)
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
    command += ["--pilot-root", str(pilot), "--sweep-summary", str(sweep_of(pilot))]
    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    audit = json.loads((out / "audit.json").read_text()) if (out / "audit.json").is_file() else {}
    return proc.returncode, audit, proc


def run_inline(root: Path, pilot: Path) -> Dict[str, Any]:
    out = (root / "output" / "audit").resolve()
    return SA.run_audit(root.resolve(), out, pilot.resolve(), sweep_of(pilot).resolve())


def failed_rules(audit: Dict[str, Any]) -> List[str]:
    return [row["rule"] for row in audit.get("failures", [])]


# ----------------------------------------------------------------------------- statistics


@pytest.mark.parametrize("df", [1, 3, 15, 39, 120, 299])
def test_student_t_matches_eval_stats(df: int) -> None:
    for t in (-6.0, -1.5, -0.01, 0.001, 0.3, 1.0, 1.7, 2.5, 4.0, 7.7, 12.0):
        ours, ref = SA.t_sf(t, df), eval_stats.student_t_sf(t, df)
        assert abs(ours - ref) <= 1e-11 * ref
    for p in (0.2, 0.05, 0.05 / 3, 0.025):
        assert SA.close(SA.t_isf(p, df), eval_stats.student_t_isf(p, df))


@pytest.mark.parametrize("case", range(20))
def test_strict_decision_matches_eval_stats(case: int) -> None:
    rng = random.Random(case)
    n = rng.choice([2, 3, 5, 40, 41, 235, 300])
    shift = rng.choice([-20.0, 0.0, 15.0, 40.0])
    deltas = {m: [shift + rng.gauss(0, 50) for _ in range(n)] for m in MIXES}
    if case % 10 == 0:
        deltas["frozen"] = [3.0] * n  # zero variance, positive mean
    margin = rng.uniform(1.0, 6.0)
    ref = eval_stats.strict_promotion_decision(deltas, "scripted", margin)
    ours = SA.strict_decision(deltas, margin)
    assert SA._compare_statistics("case", ref, ours) == []
    assert ours["passes"] is ref["passes"]


@pytest.mark.parametrize("case", range(8))
def test_pilot_sizing_matches_eval_stats(case: int) -> None:
    rng = random.Random(100 + case)
    deltas = {m: [rng.gauss(10, rng.choice([5, 40, 90, 150])) for _ in range(40)] for m in MIXES}
    ref = eval_stats.paired_delta_pilot_size(deltas, {m: 40.0 for m in MIXES})
    ours = SA.pilot_sizing(deltas)
    assert ours["required_final_worlds"] == ref["required_final_worlds"]
    assert R.frozen_final_n(deltas)["final_worlds_per_mix"] == max(40, ref["required_final_worlds"])


# ----------------------------------------------------------------------------- worlds


def test_anchor_hashes_and_rosters_match_strict_promotion() -> None:
    from src.evaluation.strict_promotion import _expected_world_identity

    seeds = R.namespace_seeds()
    rosters = R.build_rosters(seeds["dev"], seeds["final"], seeds["serving"])
    for stage, bank in (("calibration", "dev"), ("final", "final"), ("serving", "serving")):
        index = {seed: i for i, seed in enumerate(seeds[bank])}
        for row in rosters[stage]:
            expected = SA.expected_world_identity(
                row["mix"], row["world_seed"], index[row["world_seed"]]
            )
            assert expected == _expected_world_identity(row)


def test_namespaces_match_producer_and_exclude_every_earlier_domain() -> None:
    from research.apex_veto_v5_strict_20261001 import strict_run as v5strict
    from research.apex_veto_v7_screen_20261002 import screen as v7screen

    assert SA.EARLIER_DOMAINS == {k: tuple(v) for k, v in R.EARLIER_DOMAINS.items()}
    # Every domain/purpose any earlier package excluded, plus the v5 strict banks and smoke.
    for earlier in (v7screen.EARLIER_DOMAINS, v5strict.EARLIER_DOMAINS):
        for domain, purposes in earlier.items():
            assert set(purposes) <= set(SA.EARLIER_DOMAINS.get(domain, ())), domain
    for domain, _ in [*v5strict.NAMESPACES.values(), (v5strict.SMOKE_DOMAIN, 0)]:
        assert "worlds" in SA.EARLIER_DOMAINS[domain]
    assert {s: SA.NAMESPACES[s] for s in SA.STAGES} == {
        "calibration": R.NAMESPACES["dev"],
        "final": R.NAMESPACES["final"],
        "serving": R.NAMESPACES["serving"],
    }
    assert SA.SMOKE_DOMAIN == R.SMOKE_DOMAIN and SA.PILOT_DOMAIN == R.PILOT_DOMAIN
    earlier = SA.earlier_namespaces()
    for name, seeds in dev_screen.challenger_namespaces().items():
        assert earlier[f"challenger/{name}"] == seeds
    producer = R.earlier_namespaces()
    for domain, purposes in SA.EARLIER_DOMAINS.items():
        for purpose in purposes:
            assert earlier[f"{domain}/{purpose}"] == producer[f"{domain}/{purpose}[0:1000]"]
    for domain in (
        "apex-safety-screen-v1",
        "apex-veto-strict-final-v3",
        "apex-veto-strict-dev-v1",
        "apex-veto-strict-smoke-v1",
        "apex-veto-web-serving-v1",
        "apex-veto-v3-screen-v2",
        "apex-veto-v4-screen-v1",
        "apex-veto-v5-screen-v1",
        "apex-veto-v5-screen-smoke-v1",
        "trap-horizon-dev-v1",
        "trap-horizon-v5-dev-v1",
        "apex-veto-v5-strict-final-v1",
        "apex-veto-v5-strict-smoke-v1",
        "apex-veto-v5-web-serving-v1",
        "apex-veto-v6-screen-v1",
        "apex-veto-v7-dev-v1",
        "apex-veto-v7-dev-smoke-v1",
        "apex-veto-v7-screen-v1",
        "apex-veto-v7-screen-smoke-v1",
        "apex-veto-v7-strict-smoke-v1",
    ):
        assert domain in SA.EARLIER_DOMAINS
    audit = SA.Audit()
    SA.audit_namespaces(audit)
    assert audit.failures() == []
    assert set(SA.stage_bank("final")).isdisjoint(SA.pilot_bank())


def test_namespace_overlap_is_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(SA.NAMESPACES, "serving", ("apex-veto-v7-screen-v1", 50))
    audit = SA.Audit()
    SA.audit_namespaces(audit)
    assert [row["rule"] for row in audit.failures()] == ["namespaces.fresh_and_disjoint"]


# ----------------------------------------------------------------------------- per-record rules


def strict_entry(letter: str, stage: str = "final", index: int = 3) -> Dict[str, Any]:
    """A real record in the producer's envelope on the strict bank (shape unchanged)."""
    bank = SA.stage_bank(stage)
    arm = ARM_NAME[letter]
    seed, mix = bank[index], ("frozen" if stage != "serving" else SA.SERVING_MIX)
    row = {"mix": mix, "world_seed": seed, "slots": []}
    identity = SA.expected_world_identity(mix, seed, index)
    row["slots"] = [{"member_sha256": h} for h in identity["ordered_slot_content_hashes"]]
    unit = {
        "episode_id": f"{stage}-{arm}-{mix}-{seed}",
        "stage": stage,
        "arm": arm,
        "mix": mix,
        "world_seed": seed,
    }
    wrappers = {a: n["wrapper_identity"] for a, n in arm_intent_nodes().items()}
    record = move_record(letter, mix, index, seed)
    diag = DIAG[arm]
    return json.loads(json.dumps(R.episode_entry(unit, row, record, wrappers, 0, 1.0, index, diag)))


def check(entry: Dict[str, Any], stage: str = "final") -> List[str]:
    return SA.validate_entry(stage, "x", entry)[1]


def test_real_records_pass_every_record_rule() -> None:
    for letter in ("A", "B"):
        assert check(strict_entry(letter)) == []
    assert check(strict_entry("A", "calibration"), "calibration") == []
    assert check(strict_entry("B", "serving"), "serving") == []
    assert check(strict_entry("B", "calibration"), "calibration")  # incumbent-only stage
    assert check(strict_entry("A", "serving"), "serving")  # candidate-only stage


def test_fixture_provenance_is_intact() -> None:
    provenance = json.loads((FIXTURES / "provenance.json").read_text())
    assert len(provenance["sha256"]) == 6
    for name, digest in provenance["sha256"].items():
        assert sha(FIXTURES / name) == digest
    for fixtures in (SCREEN, SMOKE):
        assert fixtures["A"]["safety_veto_method"] == SA.INCUMBENT_METHOD
        assert fixtures["B"]["safety_veto_method"] == SA.CANDIDATE_METHOD
        probe = fixtures["B"]["record"]["probes"]["safety_veto"]
        assert probe["space_preference_lambda"] == 4.0
        assert "rerank_changes" in fixtures["B"]["veto_diagnostics"]
        assert "boost_landing_vetoes" in fixtures["A"]["veto_diagnostics"]
    for arm, entry in STRICT_SMOKE.items():
        assert entry["wrapper"] == SA.ARM_METHODS[arm] and entry["schema_version"] == R.SCHEMA


def test_real_strict_smoke_records_pass_the_smoke_rules() -> None:
    """The package's own plumbing-smoke records (real producer envelopes, 500 frames)."""
    seed = SA.smoke_seed()
    for arm, entry in STRICT_SMOKE.items():
        assert SA.validate_smoke_entry(copy.deepcopy(entry), seed, 500) == [], arm
        bad = copy.deepcopy(entry)
        bad["veto_diagnostics"] = STRICT_SMOKE["candidate" if arm == "incumbent" else "incumbent"][
            "veto_diagnostics"
        ]
        assert SA.validate_smoke_entry(bad, seed, 500), arm
    audit = SA.Audit()
    intent = {arm: arm_intent_nodes()[arm] for arm in R.ARMS}
    intent["source_closure"] = {"files": closure_files()}
    SA.audit_arm_binding(audit, intent, list(STRICT_SMOKE.values()))
    assert audit.failures() == []


def _mutations() -> Dict[str, Callable[[Dict[str, Any]], None]]:
    def rec(e: Dict[str, Any]) -> Dict[str, Any]:
        return e["record"]

    def veto(e: Dict[str, Any]) -> Dict[str, Any]:
        return e["record"]["probes"]["safety_veto"]

    return {
        "optimizer_updates": lambda e: e.__setitem__("optimizer_updates", 1),
        "frames_completed_short": lambda e: e.__setitem__("frames_completed", 4999),
        "scored_frames": lambda e: rec(e)["denominators"].__setitem__("scored_frames", 4000),
        "decision_over_cap": lambda e: rec(e)["denominators"].__setitem__("decision_frames", 5001),
        "survival_mismatch": lambda e: rec(e).__setitem__("survival_fraction", 0.99),
        "negative_mass": lambda e: rec(e).__setitem__("mass_integral", -1.0),
        "hero": lambda e: e.__setitem__("hero_sha256", SA.POOL_SHA256S[1]),
        "profile_digest": lambda e: rec(e).__setitem__("evaluation_profile_digest", "0" * 64),
        "learn": lambda e: rec(e)["evaluation_profile"].__setitem__("learn", True),
        "roster": lambda e: rec(e)["world_identity"]["ordered_slot_content_hashes"].reverse(),
        "seed": lambda e: rec(e).__setitem__("seed", 12345),
        "index": lambda e: e.__setitem__("world_index", e["world_index"] + 1),
        "arm": lambda e: e.__setitem__("arm", "C"),
        "stage_field": lambda e: e.__setitem__("stage", "serving"),
        "veto_flag_off": lambda e: e.__setitem__("safety_veto", False),
        "wrapper_name": lambda e: e.__setitem__("wrapper", "free-space-veto/v4-x"),
        "wrapper_sha_missing": lambda e: e.pop("wrapper_source_sha256"),
        "wrapper_shas_short": lambda e: e["wrapper_source_sha256s"].popitem(),
        "probe_missing": lambda e: rec(e)["probes"].pop("safety_veto"),
        "probe_method": lambda e: veto(e).__setitem__("method", "free-space-veto/v1"),
        "counter_extra": lambda e: veto(e)["counters"].__setitem__("extra", 1),
        "counter_missing": lambda e: veto(e)["counters"].pop("vetoes_to_boost"),
        "counter_inconsistent": lambda e: veto(e)["counters"].__setitem__("kept_base", 0),
        "counter_over_cap": lambda e: veto(e)["counters"].__setitem__("vetoes_to_boost", 5001),
    }


@pytest.mark.parametrize("mutation", sorted(_mutations()))
@pytest.mark.parametrize("letter", ["A", "B"])
def test_mutated_real_records_fail(mutation: str, letter: str) -> None:
    entry = strict_entry(letter)
    _mutations()[mutation](entry)
    assert check(entry)


def test_arm_veto_rules_are_arm_specific() -> None:
    incumbent, candidate = strict_entry("A"), strict_entry("B")
    swapped = copy.deepcopy(incumbent)
    swapped["record"]["probes"]["safety_veto"] = candidate["record"]["probes"]["safety_veto"]
    assert "safety_veto.method is not free-space-veto/v5-boost-aware" in check(swapped)
    diag = copy.deepcopy(incumbent)
    diag["veto_diagnostics"] = DIAG["candidate"]
    assert "veto_diagnostics are not the incumbent's kind" in check(diag)
    for arm, entry in (("candidate", candidate), ("incumbent", incumbent)):
        no_diag = copy.deepcopy(entry)
        no_diag.pop("veto_diagnostics")
        assert f"veto_diagnostics are not the {arm}'s kind" in check(no_diag)
    wrong_shas = copy.deepcopy(candidate)
    wrong_shas["wrapper_source_sha256s"] = incumbent["wrapper_source_sha256s"]
    assert "wrapper_source_sha256s" in check(wrong_shas)


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
    assert code == 0, (proc.stdout, proc.stderr, failed_rules(audit))
    assert audit["status"] == "PASS" and audit["producer_outcome"] == "STRICT_PASS"
    assert audit["audit_expected_outcomes"] == ["STRICT_PASS"]
    assert audit["final"]["n_final"] == 40 and audit["serving"]["complete"]
    reported = audit["reported_not_gated"]
    assert reported["veto_decisions_equal_decision_frames_by_arm"]["incumbent"] == 48 + 120
    diagnostics = reported["veto_diagnostics"]
    assert (
        diagnostics["candidate_rerank_changes"] == (120 + 50) * DIAG["candidate"]["rerank_changes"]
    )
    assert set(diagnostics) == {
        "incumbent_boost_landing_vetoes",
        "candidate_rerank_changes",
        "candidate_rerank_changes_tail_release_driven",
        "candidate_v5_boost_landing_vetoes",
    }
    rules = {row["rule"] for row in audit["rules"]}
    assert {"pilot.screen_receipt", "pilot.lambda_binding", "identity.arm_records_bound"} <= rules
    code, _, proc = run_cli(run, small_pilot)  # audit.json already exists
    assert code == 1 and "cannot write audit" in proc.stdout


@needs_checkpoints
def test_strict_fail_and_band_failure_are_consistent(small_pilot: Path, tmp_path: Path) -> None:
    run = build_run(tmp_path / "a", small_pilot, final_shift=-5.0)
    audit = run_inline(run, small_pilot)
    assert audit["status"] == "PASS", failed_rules(audit)
    assert audit["producer_outcome"] == "STRICT_FAIL"
    run = build_run(tmp_path / "b", small_pilot, candidate_alive_offset=-900)
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
    pilot = build_pilot(tmp_path / "wide", delta_sd=300.0, shift=900.0)
    run = build_run(tmp_path / "a", pilot)
    audit = run_inline(run, pilot)
    assert audit["status"] == "PASS", failed_rules(audit)
    assert audit["producer_outcome"] == "STOP_INFEASIBLE"
    assert audit["sizing"]["required_final_worlds"] > 300
    stray = run / "output" / "final" / "records" / "stray.json"
    write_json(stray, strict_entry("A"))
    assert "final.absent_when_infeasible" in failed_rules(run_inline(run, pilot))


def _set(path: List[Any], value: Any) -> Callable[[Any], None]:
    def mutate(doc: Any) -> None:
        for key in path[:-1]:
            doc = doc[key]
        doc[path[-1]] = value(doc[path[-1]]) if callable(value) else value

    return mutate


SPD = "strict_promotion_decision"
V3_KEY = str((REPO / "src/evaluation/safety_veto_v3.py").resolve())
V2_KEY = str((REPO / "src/evaluation/safety_veto.py").resolve())
V5_KEY = str((REPO / "src/evaluation/safety_veto_v5.py").resolve())
V7_KEY = str((REPO / "src/evaluation/safety_veto_v7.py").resolve())
TAMPERS: Dict[str, tuple] = {
    "p_value": (
        "output/decision.json",
        _set([SPD, "superiority", "per_mix", "frozen", "p_value"], 0.2),
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
    "required_n": (
        "intent.json",
        _set(["sizing", "pilot_sizing", "required_final_worlds"], 41),
        "claims.required_final_worlds",
    ),
    "final_seeds": (
        "intent.json",
        _set(["final_seeds"], lambda v: v[1:] + v[:1]),
        "claims.design_and_calibration",
    ),
    "pilot_deltas": (
        "intent.json",
        _set(["pilot", "deltas_by_mix", "scripted", 3], 0.0),
        "claims.design_and_calibration",
    ),
    "incumbent_not_released": (
        "intent.json",
        _set(["incumbent", "wrapper_source_sha256"], "e" * 64),
        "identity.wrapper_source_sha256",
    ),
    "incumbent_wrapper_none": (
        "intent.json",
        _set(["incumbent", "wrapper"], None),
        "identity.checkpoint_and_wrapper",
    ),
    "candidate_wrapper_v5": (
        "intent.json",
        _set(["candidate", "wrapper"], SA.INCUMBENT_METHOD),
        "identity.checkpoint_and_wrapper",
    ),
    "incumbent_v3_not_released": (
        "intent.json",
        _set(["incumbent", "wrapper_identity", "source_sha256s", SA.V3_SOURCE], "e" * 64),
        "identity.wrapper_source_sha256",
    ),
    "candidate_lambda": (
        "intent.json",
        _set(["candidate", "wrapper_identity", "descriptor", "space_preference_lambda"], 2.0),
        "identity.wrapper_source_sha256",
    ),
    "candidate_lambda_claim": (
        "intent.json",
        _set(["design", "candidate_lambda"], 2.0),
        "claims.sizing",
    ),
    "mde_claim": (
        "intent.json",
        _set(["design", "mde_absolute_per_mix"], 20.0),
        "claims.sizing",
    ),
    "lambda_binding_record": (
        "intent.json",
        _set(["pilot", "lambda_binding", "passes"], False),
        "pilot.lambda_binding",
    ),
    "closure_v5_dependency": (
        "intent.json",
        _set(["source_closure", "files", V5_KEY], "d" * 64),
        "identity.wrapper_source_bound_to_closure",
    ),
    "closure_v7": (
        "intent.json",
        _set(["source_closure", "files", V7_KEY], "d" * 64),
        "identity.wrapper_source_bound_to_closure",
    ),
    "identity_method": (
        "intent.json",
        _set(["candidate", "wrapper_identity", "method"], "free-space-veto/v1"),
        "identity.wrapper_source_sha256",
    ),
    "closure_v3_dependency": (
        "intent.json",
        _set(["source_closure", "files", V3_KEY], "d" * 64),
        "identity.wrapper_source_bound_to_closure",
    ),
    "closure_v2": (
        "intent.json",
        _set(["source_closure", "files", V2_KEY], "d" * 64),
        "identity.wrapper_source_bound_to_closure",
    ),
    "candidate_descriptor": (
        "intent.json",
        _set(["candidate", "wrapper_identity", "descriptor", "landing_rule"], "x"),
        "identity.arm_records_bound",
    ),
    "incumbent_descriptor": (
        "intent.json",
        _set(["incumbent", "wrapper_identity", "descriptor", "free_space_min_cap"], 31),
        "identity.arm_records_bound",
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


RECORD_BINDING_TAMPERS = {
    "candidate_sha_is_v2": ("candidate", lambda e: e.update(wrapper_source_sha256="1" * 64)),
    "candidate_shas_dropped_v3": (
        "candidate",
        lambda e: e["wrapper_source_sha256s"].update(
            {"src/evaluation/safety_veto_v3.py": "2" * 64}
        ),
    ),
    "candidate_landing_rule": (
        "candidate",
        lambda e: e["record"]["probes"]["safety_veto"].update(landing_rule="none"),
    ),
    "incumbent_bfs_cap": (
        "incumbent",
        lambda e: e["record"]["probes"]["safety_veto"].update(free_space_bfs_cap=159),
    ),
    "incumbent_sha": ("incumbent", lambda e: e.update(wrapper_source_sha256="3" * 64)),
    "candidate_lambda": (
        "candidate",
        lambda e: e["record"]["probes"]["safety_veto"].update(space_preference_lambda=2.0),
    ),
    "candidate_shas_dropped_v5": (
        "candidate",
        lambda e: e["wrapper_source_sha256s"].update({SA.V5_SOURCE: "4" * 64}),
    ),
}


@pytest.mark.parametrize("tamper", sorted(RECORD_BINDING_TAMPERS))
def test_records_are_bound_to_their_arm_identity(
    tamper: str, pass_run: Path, small_pilot: Path, tmp_path: Path
) -> None:
    arm, fn = RECORD_BINDING_TAMPERS[tamper]
    run = clone(pass_run, tmp_path)
    edit_json(final_record(run, arm, "mixed", 5), fn)
    assert "identity.arm_records_bound" in failed_rules(run_inline(run, small_pilot))


def test_record_bytes_extra_records_and_shards(
    pass_run: Path, small_pilot: Path, tmp_path: Path
) -> None:
    run = clone(pass_run, tmp_path / "a")
    edit_json(
        final_record(run, "candidate", "frozen", 2),
        _set(["record", "mass_integral"], lambda v: v + 50.0),
    )
    rules = failed_rules(run_inline(run, small_pilot))
    assert "shards.plan_assignment_and_hashes" in rules and "claims.statistics" in rules
    run = clone(pass_run, tmp_path / "b")
    source = final_record(run, "incumbent", "frozen", 3)
    shutil.copyfile(source, source.with_name("extra.json"))
    rules = failed_rules(run_inline(run, small_pilot))
    assert "final.records_valid" in rules and "shards.plan_assignment_and_hashes" in rules
    run = clone(pass_run, tmp_path / "c")
    reports = [run / "output" / "final" / f"shard-{k}" / "report.json" for k in (0, 1)]
    docs = [json.loads(path.read_text()) for path in reports]
    moved = docs[0]["planned_episode_ids"].pop()
    docs[1]["planned_episode_ids"].append(moved)
    docs[1]["records_sha256"][moved] = docs[0]["records_sha256"].pop(moved)
    for path, doc in zip(reports, docs):
        write_json(path, doc)
    assert "shards.plan_assignment_and_hashes" in failed_rules(run_inline(run, small_pilot))
    run = clone(pass_run, tmp_path / "d")
    final_record(run, "candidate", "scripted", 7).unlink()
    rules = set(failed_rules(run_inline(run, small_pilot)))
    assert {"claims.outcome", "claims.statistics", "shards.plan_assignment_and_hashes"} <= rules


def test_calibration_artifacts_and_stray_records(
    pass_run: Path, small_pilot: Path, tmp_path: Path
) -> None:
    run = clone(pass_run, tmp_path / "a")
    seed = SA.stage_bank("calibration")[4]
    name = f"calibration-incumbent-scripted-{seed}.json"
    (run / "output" / "calibration" / "records" / name).unlink()
    rules = failed_rules(run_inline(run, small_pilot))
    assert "claims.absolute_delta_ni" in rules and "final.requires_calibration" in rules
    run = clone(pass_run, tmp_path / "b")
    snapshot = run / "output" / "checkpoints" / f"{SA.POOL_SHA256S[1]}.pth"
    snapshot.unlink()
    assert "artifacts.hashes" in failed_rules(run_inline(run, small_pilot))
    run = clone(pass_run, tmp_path / "c")
    source = final_record(run, "incumbent", "frozen", 0)
    write_json(run / "output" / "records" / "stray.json", json.loads(source.read_text()))
    assert "records.inside_stage_dirs" in failed_rules(run_inline(run, small_pilot))
    run = clone(pass_run, tmp_path / "d")
    claim = run / "output" / "producer-outcome.json"
    edit_json(claim, lambda d: d.__setitem__("outcome", "INVALID_STOP"))
    assert "claims.outcome" in failed_rules(run_inline(run, small_pilot))
    edit_json(claim, lambda d: d.__setitem__("stop_reason", "RSS cap exceeded on worker 2"))
    assert run_inline(run, small_pilot)["status"] == "PASS"


# ----------------------------------------------------------------------------- pilot gate


def test_missing_pilot_fails_closed(pass_run: Path, tmp_path: Path) -> None:
    run = clone(pass_run, tmp_path)
    code, audit, _ = run_cli(run, tmp_path / "no-pilot")
    assert code == 1 and "pilot.available" in failed_rules(audit)


PILOT_GATE_TAMPERS: Dict[str, Callable[[Path], None]] = {
    "not_advanced": lambda p: edit_json(p / "receipt.json", _set(["decision"], "NOT_ADVANCED")),
    "before_self_check": lambda p: edit_json(
        p / "receipt.json", _set(["screen_decision_before_self_check"], "INCOMPLETE")
    ),
    "self_check_failed": lambda p: edit_json(
        p / "receipt.json", _set(["self_check", "passes"], False)
    ),
    "summary_sha": lambda p: edit_json(p / "receipt.json", _set(["summary_sha256"], "0" * 64)),
    "unbound_record": lambda p: edit_json(p / "receipt.json", lambda d: d["records"].pop(0)),
    "receipt_missing": lambda p: (p / "receipt.json").unlink(),
}


@pytest.mark.parametrize("tamper", sorted(PILOT_GATE_TAMPERS))
def test_pilot_gate_requires_a_recommending_screen(
    tamper: str, pass_run: Path, small_pilot: Path, tmp_path: Path
) -> None:
    pilot = tmp_path / "screen"
    shutil.copytree(small_pilot, pilot)
    PILOT_GATE_TAMPERS[tamper](pilot)
    rules = failed_rules(run_inline(clone(pass_run, tmp_path), pilot))
    assert "pilot.screen_receipt" in rules


@pytest.mark.parametrize(
    "mutate",
    [
        lambda e: e.update(safety_veto_method=SA.INCUMBENT_METHOD),
        lambda e: e["record"]["probes"]["safety_veto"].update(space_preference_lambda=2.0),
    ],
)
def test_pilot_arm_identity_is_checked(small_pilot: Path, tmp_path: Path, mutate) -> None:
    pilot = tmp_path / "screen"
    shutil.copytree(small_pilot, pilot)
    path = next((pilot / "records").glob("B-mixed-*.json"))
    edit_json(path, mutate)
    audit = SA.Audit()
    assert SA.load_pilot(audit, pilot) is None
    assert "pilot.pairs" in [row["rule"] for row in audit.failures()]


def _edit_screen_intent(pilot: Path) -> None:
    edit_json(
        pilot / "intent.json",
        lambda i: i["arms"].update(B=i["arms"]["B"].replace("lambda=4.0", "lambda=2.0")),
    )
    edit_json(pilot / "receipt.json", _set(["intent_sha256"], sha(pilot / "intent.json")))


# Each makes the screen's lambda binding wrong in one place (producer gate and audit rule).
LAMBDA_TAMPERS: Dict[str, Callable[[Path], None]] = {
    "receipt_lambda": lambda p: edit_json(p / "receipt.json", _set(["lambda"], 2.0)),
    "binding_selected": lambda p: edit_json(
        p / "receipt.json", _set(["sweep_binding", "selected_lambda"], 2.0)
    ),
    "binding_sha": lambda p: edit_json(
        p / "receipt.json", _set(["sweep_binding", "summary_sha256"], "0" * 64)
    ),
    "binding_commit": lambda p: edit_json(
        p / "receipt.json", _set(["sweep_binding", "source_commit"], "0" * 40)
    ),
    "sweep_selected": lambda p: rebind_sweep(p, _set(["selection", "selected_lambda"], 2.0)),
    "sweep_status": lambda p: rebind_sweep(p, _set(["selection", "status"], "NONE_QUALIFIES")),
    "sweep_smoke": lambda p: rebind_sweep(p, _set(["smoke"], True)),
    "sweep_commit": lambda p: rebind_sweep(p, _set(["source", "commit"], "0" * 40)),
    "sweep_dirty": lambda p: rebind_sweep(p, _set(["source", "dirty_paths"], " M src/x.py")),
    "screen_intent_arm_b": _edit_screen_intent,
}


@pytest.mark.parametrize("tamper", sorted(LAMBDA_TAMPERS))
def test_lambda_binding_rule(tamper: str, pass_run: Path, small_pilot: Path, tmp_path: Path):
    assert (
        SA.lambda_binding_problems(
            small_pilot, sweep_of(small_pilot), json.loads((pass_run / "intent.json").read_text())
        )
        == []
    )
    pilot = tmp_path / "screen"
    shutil.copytree(small_pilot, pilot)
    LAMBDA_TAMPERS[tamper](pilot)
    rules = failed_rules(run_inline(clone(pass_run, tmp_path), pilot))
    assert "pilot.lambda_binding" in rules
