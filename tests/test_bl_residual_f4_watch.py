"""Bounded synthetic proof of the actual F4 Watch6 session runner wiring."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.bl_ppo.exports import digest, export_json
from research.bl_residual import analyze, gameplay


@pytest.mark.parametrize("fault", [None, "controller", "infrastructure"])
def test_watch6_runner_clones_and_binds_hero_before_constructor(
    monkeypatch, tmp_path: Path, fault: str | None
) -> None:
    """Run 128 tiny fake pairs through real admission, bank and aggregation code."""
    fixture_path = Path(__file__).with_name("test_bl_residual_f4_analyzer.py")
    spec = importlib.util.spec_from_file_location("f4_watch_fixture", fixture_path)
    assert spec and spec.loader
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    admission = analyze.h5000_admission(fixture._payload())
    assert admission["status"] == "PASS_WATCH6"
    admission_path = tmp_path / "h5000-admission.json"
    seal = export_json(admission_path, admission)
    handoff = {**admission, "receipt_path": str(admission_path), "receipt_sha256": seal["sha256"]}
    b8, top = tmp_path / "b8.pth", tmp_path / "top.pth"
    b8.write_bytes(b"synthetic baseline placeholder")
    top.write_bytes(b"synthetic candidate placeholder")
    original_sha = gameplay.sha256_file
    identities = {b8: gameplay.B8_SHA256, top: admission["controller_sha256"]["D4096"]}
    monkeypatch.setattr(
        gameplay, "sha256_file", lambda path: identities.get(Path(path)) or original_sha(path)
    )
    monkeypatch.setattr(gameplay, "_require_mac", lambda: None)
    monkeypatch.setattr(gameplay, "_approved_successor", lambda _: {})
    monkeypatch.setattr(gameplay, "_validate_candidates", lambda *_: None)
    monkeypatch.setattr(gameplay, "WATCH6_HORIZON", 3)
    rows = [
        {
            "session_id": f"synthetic-{i}",
            "seed": 50000 + i,
            "parent_world_id": f"synthetic-parent-{i}",
            "mix": "mixed",
        }
        for i in range(128)
    ]
    audited_rows = [
        {
            "world_id": row["session_id"],
            "world_seed": row["seed"],
            "parent_world_id": row["parent_world_id"],
            "mix": row["mix"],
        }
        for row in rows
    ]
    allocation = gameplay.allocation_sha256(audited_rows)
    collision_path = tmp_path / "collision.json"
    export_json(
        collision_path,
        {"status": "CLEAR", "namespace": "synthetic-watch/v1", "allocation_sha256": allocation},
    )
    bank = {
        "namespace": "synthetic-watch/v1",
        "rows": rows,
        "allocation_sha256": allocation,
        "collision_receipt": {
            "status": "CLEAR",
            "namespace": "synthetic-watch/v1",
            "path": str(collision_path),
            "sha256": digest(collision_path),
            "allocation_sha256": allocation,
        },
    }
    from research.t3_rollout_labels_20261015 import bank_engine
    from research.t34_pathb import adapter_v2
    from research.t34_shadow import shadow

    constructed, sources, closed = [], [], []

    def open_watch(row, fps, arm):
        assert fps == 24 and arm == "bare"
        source = SimpleNamespace(
            world=SimpleNamespace(hero=SimpleNamespace(checkpoint_path=str(b8))), specs=[]
        )
        source.close = lambda: sources.append(row["session_id"])
        return source

    def build_world(name, plan, *, hero_checkpoint):
        assert name == "b8_s3" and plan["mix"] == "mixed" and hero_checkpoint == top
        return SimpleNamespace(hero=SimpleNamespace(checkpoint_path=str(top)))

    class Session:
        def __init__(self, world, arm, *, specs, checkpoint_sha):
            assert arm == "bare"
            # This assertion fails if the candidate is installed only after construction.
            assert identities[Path(world.hero.checkpoint_path)] == checkpoint_sha
            constructed.append((id(world), checkpoint_sha))
            self.world = world
            self.decisions = self.boost_decisions = self.decisions_delta = 0
            self.executed_action, self.hero_length = 1, 10
            self.hero_death, self.death_length, self.ticks = False, 0, 0

        def step(self):
            if fault == "controller":
                raise gameplay.ControllerFault("nonfinite_or_malformed_q")
            if fault == "infrastructure":
                raise RuntimeError("synthetic infrastructure failure")
            self.ticks += 1
            self.hero_death = self.ticks == 2
            self.death_length = 10 if self.hero_death else 0
            self.hero_length = 10 if self.ticks == 1 else 0
            self.decisions_delta = int(self.ticks <= 2)
            self.decisions += self.decisions_delta
            self.executed_action = 4 if self.ticks == 1 else 1
            self.boost_decisions += int(self.ticks == 1)

        def close(self):
            closed.append(self.world)

    monkeypatch.setattr(adapter_v2, "open_watch", open_watch)
    monkeypatch.setattr(adapter_v2, "RunSession", Session)
    monkeypatch.setattr(bank_engine, "build_world", build_world)
    monkeypatch.setattr(shadow, "clone_world_exact", copy.deepcopy)
    monkeypatch.setattr(shadow, "full_digest", lambda *_args, **_kwargs: "d" * 64)

    def run():
        return gameplay.run_watch6_sessions(
            bank,
            b8_path=b8,
            d4096_path=top,
            review_evidence={},
            candidate_metadata={},
            h5000_go_receipt=handoff,
        )

    if fault == "infrastructure":
        with pytest.raises(RuntimeError, match="synthetic infrastructure failure"):
            run()
        assert len(closed) == len(sources) == 1
        return
    result = run()
    if fault == "controller":
        assert result["known_controller_fault"]
        assert len(closed) == len(sources) == 1
        payload = fixture._payload()
        payload["watch6"] = result
        fixture._reseal(payload)
        assert analyze.analyze(payload)["label"] == "KILL_FIT"
        return
    assert len(result["sessions"]) == 128
    assert len(constructed) == len(closed) == 256
    assert len(sources) == 128
    # Every session retains its cloned world until close, making object identities distinct.
    assert len({world_id for world_id, _ in constructed}) == 256
    for session in result["sessions"]:
        for arm in session["arms"].values():
            assert arm["hero_decisions"] == 2
            assert len(arm["action_trace"]) == 2
            assert arm["deaths"] == 1
            assert arm["mass_integral"] == pytest.approx(10 / 3)
            assert arm["boost_share"] == 0.5
            assert arm["death_events"] == [{"frame": 1, "pre_death_length": 10}]
            assert arm["median_length"] == 0
            assert arm["p95_length"] == pytest.approx(9)


@pytest.mark.parametrize("fault", [None, "controller", "infrastructure"])
def test_probe_known_fault_preserves_actual_controller_identity(
    monkeypatch, tmp_path: Path, fault: str | None
) -> None:
    """Use the real sealed retention consumer and terminal probe fault receipt."""
    fixture_path = Path(__file__).with_name("test_bl_residual_f4_analyzer.py")
    spec = importlib.util.spec_from_file_location("f4_probe_fixture", fixture_path)
    assert spec and spec.loader
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    payload = fixture._payload()
    b8, top = tmp_path / "b8.pth", tmp_path / "top.pth"
    b8.write_bytes(b"synthetic baseline")
    top.write_bytes(b"synthetic candidate")
    original_sha = gameplay.sha256_file
    identities = {b8: gameplay.B8_SHA256, top: payload["identity"]["controller_sha256"]["D4096"]}
    monkeypatch.setattr(
        gameplay, "sha256_file", lambda path: identities.get(Path(path)) or original_sha(path)
    )
    monkeypatch.setattr(gameplay, "_require_mac", lambda: None)
    monkeypatch.setattr(gameplay, "_approved_successor", lambda _: {})
    monkeypatch.setattr(gameplay, "_validate_candidates", lambda *_: None)
    monkeypatch.setattr(
        gameplay,
        "validate_probe_bank",
        lambda _: [{"plan": {}, "world_id": "synthetic-probe-world", "mix": "scripted"}],
    )
    retention_path = tmp_path / "retention.json"
    export_json(
        retention_path,
        {
            "schema": "bl-residual-heldout/v2",
            "dose": 4096,
            "registration_sha256": gameplay.F4_REGISTRATION_SHA256,
            "checkpoint_sha256": identities[top],
            "metrics": {"life_macro_rescue_retention": 0.8},
        },
    )
    retention = gameplay._sealed_receipt_wrapper(retention_path)

    def fail(*_args, **_kwargs):
        if fault is None:
            return object(), {
                "B8": SimpleNamespace(hero="B8"),
                "D4096": SimpleNamespace(hero="D4096"),
            }
        if fault == "controller":
            raise gameplay.ControllerFault("nonfinite_or_malformed_q")
        raise RuntimeError("synthetic infrastructure failure")

    monkeypatch.setattr(gameplay, "_build_exact_roots", fail)
    monkeypatch.setattr(
        gameplay,
        "_run_exact_arm_from_root",
        lambda *_args, **_kwargs: {
            "decisions": 3,
            "action_trace": [
                {
                    "decision": index,
                    "hero_length": 7 + index,
                    "served_action": action,
                    "b8_served_action": action,
                }
                for index, action in enumerate((0, 3, 0))
            ],
        },
    )

    def run():
        return gameplay.run_probe(
            {"namespace": "synthetic-probe/v1"},
            b8_path=b8,
            d4096_path=top,
            heldout_receipt=retention,
            review_evidence={},
            candidate_metadata={},
        )

    if fault == "infrastructure":
        with pytest.raises(RuntimeError, match="synthetic infrastructure failure"):
            run()
        return
    result = run()
    if fault is not None:
        assert result["known_controller_fault"]
    else:
        assert result["pass"] is True
        assert result["action_collapse"]["D4096"]["pass"] is True
    assert result["controller_sha256"] == {
        "B8": identities[b8],
        "D4096": identities[top],
    }
    payload["probe"] = result
    fixture._reseal(payload)
    assert analyze.analyze(payload)["label"] == (
        "KILL_FIT" if fault is not None else "GO_TOP_DOSE_DEV"
    )
