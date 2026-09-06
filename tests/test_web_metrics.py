"""Tests for the dashboard data module (web/backend/metrics.py).

Covers the dual-schema eval-leaderboard parsing (legacy ``summaries`` files vs
the repaired gate's ``candidates``/``baseline_runs`` format) and the
obs_spec-enriched checkpoint listing with its (path, mtime) cache.
"""

import json
import os

import pytest
import torch

pytest.importorskip("fastapi")

from src.model.obs_spec import OBS_SPEC_KEY, RASTER31V2, VECTOR61  # noqa: E402
from web.backend import metrics  # noqa: E402


@pytest.fixture()
def dirs(tmp_path, monkeypatch):
    """Point the module's artifact roots at a temp tree and clear the cache."""
    saved = tmp_path / "saved_snakes"
    runs = tmp_path / "runs"
    logs = tmp_path / "logs"
    for d in (saved, runs, logs):
        d.mkdir()
    monkeypatch.setattr(metrics, "REPO_ROOT", str(tmp_path))
    monkeypatch.setattr(metrics, "SAVED_DIR", str(saved))
    monkeypatch.setattr(metrics, "RUNS_DIR", str(runs))
    monkeypatch.setattr(metrics, "LOGS_DIR", str(logs))
    metrics._OBS_SPEC_CACHE.clear()
    yield {"saved": saved, "runs": runs, "logs": logs}
    metrics._OBS_SPEC_CACHE.clear()


def _legacy_run(mean_mass: float, survival: float = 1.0) -> dict:
    return {
        "seed": 0,
        "max_mass": mean_mass * 2,
        "mean_mass": mean_mass,
        "kills": 0.0,
        "deaths": 0.0,
        "survival": survival,
    }


def _gate_run(mass_integral: float, survival: float = 0.9) -> dict:
    return {
        "seed": 0,
        "mass_integral": mass_integral,
        "max_mass": mass_integral * 3,
        "mean_mass_alive": mass_integral * 1.5,
        "kills": 1.0,
        "deaths": 0.0,
        "survival_fraction": survival,
        "probes": {"death_cause": None},
    }


def _write_legacy_file(logs, name="eval_legacy.json", mean_mass=100.0):
    payload = {
        "opponent": "saved_snakes/best_apex.pth",
        "frames": 1500,
        "summaries": [
            {
                "candidate": "/abs/path/cand_old.pth",
                "n": 2,
                "runs": [_legacy_run(mean_mass), _legacy_run(mean_mass + 10)],
            }
        ],
    }
    (logs / name).write_text(json.dumps(payload))


def _write_gate_file(logs, name="eval_gate.json"):
    payload = {
        "mode": "gate",
        "engine": "simd",
        "frames": 3000,
        "baseline": "saved_snakes/champion.pth",
        "mixes": ["scripted", "frozen"],
        "baseline_summaries": {"scripted": {"n": 2, "mass_integral": 5.0, "mass_integral_ci": 0.4}},
        "baseline_runs": {"scripted": [_gate_run(4.0), _gate_run(6.0)]},
        "candidates": [
            {
                "candidate": "runs/pqn_local/latest_pqn.pth",
                "per_mix": {
                    "scripted": {
                        "runs": [_gate_run(8.0), _gate_run(10.0)],
                        "summary": {"n": 2, "mass_integral": 9.0, "mass_integral_ci": 1.2},
                        "paired": {"significant": True},
                    },
                    "frozen": {
                        "runs": [_gate_run(7.0)],
                        "summary": {"n": 1, "mass_integral": 7.0},
                        "paired": {"significant": False},
                    },
                },
                "decision": {"promote": False},
            },
            {
                "candidate": "broken.pth",
                "error": "shape mismatch",
                "decision": {"promote": False, "reasons": ["shape mismatch"]},
            },
        ],
    }
    (logs / name).write_text(json.dumps(payload))


class TestEvalLeaderboard:
    def test_legacy_file_parses_with_metric_label(self, dirs):
        _write_legacy_file(dirs["logs"])
        rows = metrics.eval_leaderboard()
        assert len(rows) == 1
        row = rows[0]
        assert row["metric"] == "legacy_mean_mass"
        assert row["candidate"] == "cand_old.pth"
        assert row["opponent"] == "best_apex.pth"
        assert row["mean_mass"] == 105.0
        assert row["survival"] == 1.0
        assert row["frames"] == 1500
        assert row["date"] is not None  # file mtime, ISO

    def test_gate_file_parses_candidates_and_baseline(self, dirs):
        _write_gate_file(dirs["logs"])
        rows = metrics.eval_leaderboard()
        # 2 candidate mixes + 1 baseline mix; the errored candidate yields none.
        assert len(rows) == 3
        assert all(r["metric"] == "mass_integral" for r in rows)
        assert not any(r["candidate"] == "broken.pth" for r in rows)

        cand_scripted = next(
            r
            for r in rows
            if r["candidate"] == "latest_pqn.pth" and r["opponent"] == "scripted mix"
        )
        # Headline mass comes from mass_integral, survival from survival_fraction.
        assert cand_scripted["mean_mass"] == 9.0
        assert cand_scripted["survival"] == 0.9
        assert cand_scripted["mean_mass_alive"] == 13.5
        assert cand_scripted["mass_ci"] == 1.2
        assert cand_scripted["n"] == 2

        baseline = next(r for r in rows if r["candidate"] == "champion.pth")
        assert baseline["opponent"] == "scripted mix"
        assert baseline["mean_mass"] == 5.0
        assert baseline["mass_ci"] == 0.4

    def test_gate_rows_rank_before_legacy_rows(self, dirs):
        """mass_integral and legacy mean_mass are incomparable: never interleave."""
        # Legacy mass (100+) dwarfs the gate mass integral (<10) numerically.
        _write_legacy_file(dirs["logs"])
        _write_gate_file(dirs["logs"])
        rows = metrics.eval_leaderboard()
        metrics_seq = [r["metric"] for r in rows]
        assert metrics_seq == ["mass_integral"] * 3 + ["legacy_mean_mass"]
        # Within the gate block, sorted by headline mass desc.
        gate_masses = [r["mean_mass"] for r in rows[:3]]
        assert gate_masses == sorted(gate_masses, reverse=True)

    def test_unparseable_and_alien_files_skipped(self, dirs):
        (dirs["logs"] / "eval_junk.json").write_text("{not json")
        (dirs["logs"] / "eval_other.json").write_text(json.dumps({"something": 1}))
        assert metrics.eval_leaderboard() == []


def _save_ckpt(path, spec=None):
    blob = {"dqn_state_dict": {}}
    if spec is not None:
        blob[OBS_SPEC_KEY] = spec
    torch.save(blob, str(path))


class TestListCheckpoints:
    def test_obs_spec_enrichment(self, dirs):
        _save_ckpt(dirs["saved"] / "vec.pth")  # no key -> vector61 default
        _save_ckpt(dirs["saved"] / "ras.pth", RASTER31V2)
        (dirs["saved"] / "junk.pth").write_text("not a checkpoint")
        out = {c["name"]: c for c in metrics.list_checkpoints()}
        assert out["vec.pth"]["obs_spec"] == VECTOR61
        assert out["ras.pth"]["obs_spec"] == RASTER31V2
        assert out["junk.pth"]["obs_spec"] == "unknown"
        assert all("size_mb" in c and "path" in c for c in out.values())

    def test_scans_runs_latest_pqn(self, dirs):
        exp = dirs["runs"] / "pqn_local"
        exp.mkdir()
        _save_ckpt(exp / "latest_pqn.pth", RASTER31V2)
        # Other run artifacts are not swept in.
        _save_ckpt(exp / "opponent_003.pth", RASTER31V2)
        out = metrics.list_checkpoints()
        names = [c["name"] for c in out]
        assert os.path.join("runs", "pqn_local", "latest_pqn.pth") in names
        assert not any("opponent_003" in n for n in names)
        entry = next(c for c in out if "latest_pqn" in c["name"])
        assert entry["obs_spec"] == RASTER31V2

    def test_obs_spec_cached_by_path_and_mtime(self, dirs, monkeypatch):
        _save_ckpt(dirs["saved"] / "vec.pth")
        calls = {"n": 0}
        real_load = torch.load

        def counting_load(*args, **kwargs):
            calls["n"] += 1
            return real_load(*args, **kwargs)

        monkeypatch.setattr(torch, "load", counting_load)
        metrics.list_checkpoints()
        assert calls["n"] == 1
        metrics.list_checkpoints()  # same mtime -> cache hit, no re-read
        assert calls["n"] == 1
        # Touch the file with a new mtime -> re-probed once.
        path = dirs["saved"] / "vec.pth"
        st = os.stat(path)
        os.utime(path, (st.st_atime, st.st_mtime + 10))
        metrics.list_checkpoints()
        assert calls["n"] == 2
