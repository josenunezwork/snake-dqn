"""T2 identity, diagnostic accounting, restart and synthetic decision contracts."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from research.t2_screen_20261015 import probe_t2, watch_t2

from research.t2_screen_20261015.analyze_t2 import analyze, extension_requirements

ARTIFACTS = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts")


def _synthetic_probe_context(tmp_path: Path):
    """Build a real local GridBatchSim context with only synthetic checkpoints."""
    from types import SimpleNamespace

    from research.redesign_m2_20261008 import ni_spec
    from src.core.config_loader import load_and_initialize_config
    from src.scripts.tournament_eval import evaluation_profile_for_name
    from src.training.apex_policy import ApexPolicy
    import torch

    seeds = [901, 902]
    carrier = ni_spec.FRP3_S12[1]
    carrier_path = tmp_path / "carrier.pth"
    load_and_initialize_config("research/apex_safety_20260926/deployment.yaml")
    torch.manual_seed(1717)
    carrier_policy = ApexPolicy(61, 64, 6, training=False)
    torch.save(carrier_policy.get_state_dict(), carrier_path)
    return seeds, {
        "profile": evaluation_profile_for_name("promotion-v2-watch-rect", 5000),
        "lookup": {
            carrier: ("checkpoint", str(carrier_path)),
            "synthetic-anchor": ("scripted", "greedy_food"),
        },
        "ds": SimpleNamespace(
            _design_rows=lambda requested: [
                {
                    "world_seed": seed,
                    "mix": "scripted",
                    "slots": [{"member_sha256": "synthetic-anchor"}] * 5,
                }
                for seed in requested
            ]
        ),
    }


def test_two_world_h5000_dev_probe_identity(monkeypatch, tmp_path: Path) -> None:
    """A valid local checkpoint and full synthetic H5000 preserve adapter identity."""
    import torch

    from research.redesign_m2_20261008 import dev_probe
    from src.model.ego2s_network import Ego2sNet, load_ego2s_checkpoint, save_ego2s_checkpoint

    checkpoint = tmp_path / "student.pth"
    torch.manual_seed(1717)
    save_ego2s_checkpoint(checkpoint, Ego2sNet())
    assert isinstance(load_ego2s_checkpoint(str(checkpoint)), Ego2sNet)
    seeds, ctx = _synthetic_probe_context(tmp_path)
    monkeypatch.setattr(dev_probe, "probe_seeds", lambda worlds, round_index: seeds)
    torch.set_num_threads(1)
    args = probe_t2.build_parser().parse_args(
        [
            "--student",
            str(checkpoint),
            "--arm",
            "student",
            "--mix",
            "scripted",
            "--worlds",
            "2",
            "--out",
            "/tmp/t2-unused.json",
            "--name",
            "CTRL_s10",
        ]
    )
    records, original = dev_probe.probe(args, ctx)
    diagnostic = probe_t2.probe(args, ctx)
    assert diagnostic["mass"] == original["mass"]
    assert diagnostic["deaths"] == original["deaths"]
    assert diagnostic["causes"] == {rec["seed"]: rec["probes"]["death_cause"] for rec in records}
    assert diagnostic["alive_frames"] == {
        rec["seed"]: rec["denominators"]["alive_frames"] for rec in records
    }
    assert any(value == 1 for value in diagnostic["deaths"].values())
    for seed, frame in diagnostic["death_frame"].items():
        if frame is None:
            assert diagnostic["alive_frames"][seed] == 5000
            assert diagnostic["length_at_death"][seed] is None
        else:
            assert diagnostic["alive_frames"][seed] == frame - 1
            assert diagnostic["length_at_death"][seed] > 0


def test_observer_death_frame_is_excluded_and_length_precedes_fatal_action() -> None:
    observer = probe_t2.SurvivalObserver([17, 18])
    observer(
        {
            "frame": 0,
            "seeds": (17, 18),
            "pre": {"alive": [True, True], "logical_mass": [800, 900]},
            "post": {"alive": [True, True]},
        }
    )
    observer(
        {
            "frame": 1,
            "seeds": (17, 18),
            "pre": {"alive": [True, True], "logical_mass": [801, 901]},
            "post": {"alive": [False, True]},
        }
    )
    observer(
        {
            "frame": 2,
            "seeds": (17, 18),
            "pre": {"alive": [False, True], "logical_mass": [0, 902]},
            "post": {"alive": [False, True]},
        }
    )
    assert observer.alive_frames == {17: 1, 18: 3}
    assert observer.death_frame == {17: 2, 18: None}
    assert observer.length_at_death == {17: 801, 18: None}


def test_watcher_waits_for_successful_exit_and_checkpoint(tmp_path: Path) -> None:
    train, ctrl = tmp_path / "train", tmp_path / "ctrl"
    directory = train / "HUBER_s10"
    directory.mkdir(parents=True)
    (directory / "final.pth").write_bytes(b"weights")
    (directory / "summary.json").write_text("{}")
    assert not watch_t2.ready("HUBER_s10", train, ctrl)
    (train / "status.txt").write_text("HUBER_s10 exit=1\n")
    assert not watch_t2.ready("HUBER_s10", train, ctrl)
    (train / "status.txt").write_text("HUBER_s10 exit=0\n")
    assert watch_t2.ready("HUBER_s10", train, ctrl)


def test_watcher_existing_rows_must_be_complete_and_have_same_settings(tmp_path: Path) -> None:
    from research.redesign_m2_20261008.dev_probe import probe_seeds

    job = {"name": "CTRL_s10", "mix": "scripted", "worlds": 2, "round_index": 60}
    row = dict(
        job,
        schema="t2-screen-probe/v1",
        frames=5000,
        veto="none",
        ego2s_forward="rowwise",
        ego2s_mask="advisory",
        v61_backend="numba",
        anchors="fast",
    )
    for key in ("mass", "deaths", "alive_frames", "death_frame", "length_at_death", "causes"):
        row[key] = dict.fromkeys(map(str, probe_seeds(2, 60)), 0)
    path = tmp_path / "row.json"
    path.write_text(json.dumps(row))
    assert watch_t2.valid_row(path, job, None)
    row["ego2s_forward"] = "batched"
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="incompatible"):
        watch_t2.valid_row(path, job, None)


def test_watcher_lock_survives_in_owned_child(tmp_path: Path) -> None:
    """A restarted watcher cannot exceed the resource cap after its parent dies."""
    import fcntl
    import subprocess
    import sys

    lock_path = tmp_path / "singleton.lock"
    owner = lock_path.open("w")
    fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    child = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.read(1)"],
        pass_fds=(owner.fileno(),),
        stdin=subprocess.PIPE,
    )
    owner.close()
    try:
        with lock_path.open("w") as restarted:
            with pytest.raises(BlockingIOError):
                fcntl.flock(restarted, fcntl.LOCK_EX | fcntl.LOCK_NB)
        child.communicate(b"x", timeout=10)
        with lock_path.open("w") as restarted:
            fcntl.flock(restarted, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


def _row(name, mix, deaths, mass=100.0, alive=None, round_index=60, offset=0):
    worlds = [str(number) for number in range(offset, offset + len(deaths))]
    return {
        "name": name,
        "round_index": round_index,
        "mix": mix,
        "frames": 5000,
        "mass": {world: mass for world in worlds},
        "deaths": {world: death for world, death in zip(worlds, deaths)},
        "alive_frames": {
            world: 5000 if death == 0 else (alive if alive is not None else 2500)
            for world, death in zip(worlds, deaths)
        },
    }


def _dataset(overrides=None, round_index=60):
    overrides = overrides or {}
    rows = []
    names = ["CTRL_s%d" % seed for seed in range(10, 15)]
    names += [
        "%s_s%d" % (arm, seed)
        for arm in ("HUBER", "ANCHOR0", "HUBER_ANCHOR0")
        for seed in range(10, 15)
    ]
    for name in names:
        arm = name.rsplit("_s", 1)[0]
        for mix in ("scripted", "frozen", "mixed"):
            default = ([1, 1], 100.0, 2500) if arm == "CTRL" else ([1, 1], 100.0, 2500)
            death, mass, alive = overrides.get((arm, mix), default)
            rows.append(_row(name, mix, death, mass, alive, round_index))
    for mix in ("scripted", "frozen", "mixed"):
        rows.append(_row("baseline", mix, [0, 0], 50.0, round_index=round_index))
    telemetry = {name: {"holdout_agreement": 0.7, "greedy_boost_rate": 0.1} for name in names}
    return rows, telemetry


def test_go_reports_seed_and_pooled_world_se():
    rows, telemetry = _dataset({("HUBER", "scripted"): ([0, 0], 80.0, None)})
    result = analyze(rows, telemetry)
    arm = result["arms"]["HUBER"]
    assert arm["decision"] == "GO"
    primary = arm["vs_ctrl"]["scripted"]["episode_survival"]
    assert primary["mean"] == 1.0
    assert primary["seed_se"] == 0.0
    assert primary["pooled_world_se"] == 0.0
    assert arm["vs_ctrl"]["scripted"]["mass"]["mean"] == -20.0


def test_kill_on_nonpositive_primary_and_tripwire():
    rows, telemetry = _dataset({("HUBER", "scripted"): ([1, 1], 100.0, 1000)})
    assert analyze(rows, telemetry)["arms"]["HUBER"]["decision"] == "KILL"
    rows, telemetry = _dataset({("HUBER", "scripted"): ([0, 0], 30.0, None)})
    assert analyze(rows, telemetry)["arms"]["HUBER"]["decision"] == "KILL"


def test_extend_and_round_61_exhaustion():
    rows, telemetry = _dataset({("HUBER", "scripted"): ([0, 1], 60.0, 3000)})
    initial = analyze(rows, telemetry)
    assert initial["arms"]["HUBER"]["decision"] == "EXTEND"
    requested = extension_requirements(initial, rows)
    assert len(requested) == 10
    assert {entry["name"] for entry in requested} == {"HUBER_s%d" % s for s in range(10, 15)} | {
        "CTRL_s%d" % s for s in range(10, 15)
    }
    rows.extend(
        _row(name, "scripted", [0, 1], 20.0 if name.startswith("HUBER") else 100.0, 3000, 61, 2)
        for name in ["HUBER_s%d" % s for s in range(10, 15)]
        + ["CTRL_s%d" % s for s in range(10, 15)]
    )
    final = analyze(rows, telemetry)
    assert final["arms"]["HUBER"]["decision"] == "EXTEND"
    assert final["arms"]["HUBER"]["extension_exhausted"] is True
    assert extension_requirements(final, rows) == []


def test_interaction_is_difference_in_differences():
    rows, telemetry = _dataset(
        {
            ("HUBER_ANCHOR0", "scripted"): ([0, 0], 100.0, None),
        }
    )
    interaction = analyze(rows, telemetry)["interaction"]
    assert interaction["mean"] == 1.0
    assert interaction["seed_se"] == 0.0
    assert not math.isnan(interaction["pooled_world_se"])


def test_nonzero_seed_and_pooled_world_se() -> None:
    rows, values = _dataset()
    for seed, survivors in zip(range(10, 15), (3, 2, 1, 0, 0)):
        rows = [
            r
            for r in rows
            if not (r["mix"] == "scripted" and r["name"] in (f"HUBER_s{seed}", f"CTRL_s{seed}"))
        ]
        rows.append(_row(f"CTRL_s{seed}", "scripted", [1] * 4))
        rows.append(_row(f"HUBER_s{seed}", "scripted", [0] * survivors + [1] * (4 - survivors)))
    result = analyze(rows, values)["arms"]["HUBER"]["vs_ctrl"]["scripted"]
    assert result["episode_survival"]["mean"] == pytest.approx(0.3)
    assert result["episode_survival"]["seed_se"] == pytest.approx(0.1457737973711325)
    assert result["alive_survival"]["mean"] == pytest.approx(0.15)
    assert result["alive_survival"]["seed_se"] == pytest.approx(0.1457737973711325 / 2)
    assert result["episode_survival"]["pooled_world_se"] == pytest.approx(math.sqrt(0.3 * 0.7 / 19))


def test_missing_holdout_blocks_go_and_writes_valid_json(tmp_path: Path) -> None:
    from research.t2_screen_20261015.analyze_t2 import write_results

    rows, values = _dataset({("HUBER", "scripted"): ([0, 0], 100.0, None)})
    values["HUBER_s10"]["holdout_agreement"] = None
    partial = analyze(rows, values)
    assert partial["arms"]["HUBER"]["decision"] == "EXTEND"
    for seed in range(10, 15):
        values[f"HUBER_s{seed}"]["holdout_agreement"] = None
    result = analyze(rows, values)
    path = tmp_path / "analysis.json"
    write_results(result, path, tmp_path / "result.md")
    saved = json.loads(path.read_text())
    assert saved["arms"]["HUBER"]["telemetry"]["holdout_agreement_min"] is None
    assert "NaN" not in path.read_text()


def test_two_extended_arms_share_ctrl_jobs_once() -> None:
    rows, values = _dataset(
        {
            ("HUBER", "scripted"): ([0] + [1] * 19, 100.0, None),
            ("ANCHOR0", "scripted"): ([0] + [1] * 19, 100.0, None),
            ("CTRL", "scripted"): ([1] * 20, 100.0, None),
        }
    )
    result = analyze(rows, values)
    requests = extension_requirements(result, rows)
    assert len(requests) == 15
    assert len({watch_t2.job_key(job) for job in requests}) == 15


def test_analysis_refuses_incomplete_initial_design() -> None:
    rows, values = _dataset()
    with pytest.raises(ValueError, match="21 x 3"):
        analyze(rows[:-1], values)
    rows[0]["deaths"].pop("0")
    with pytest.raises(ValueError, match="missing or inconsistent"):
        analyze(rows, values)
