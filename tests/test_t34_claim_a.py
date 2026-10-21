"""Claim A pipeline: student fan-out labels, payload/ReplaySource, probe, manifests."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from research.t34_claim_a import label as claim_label
from research.t34_claim_a import make_manifest as claim_manifest
from research.t34_claim_a import pipeline, replay_source
from research.t34_claim_a.probe import DevProbe, probe_rows, validate_smoke_probe
from research.t34_exit_b8 import label as ratified_label
from research.t34_exit_b8_fit.diagnostics import ProbeResult

FACTORY = "research.t34_claim_a.student:student_policy"
MAIN_SAVED = Path("/Users/josenunez/Projects/ml/snake-dqn/saved_snakes")


def _b8() -> Path:
    inventory = json.loads(Path("research/t34_exit_b8/INPUTS_REQUIRED.json").read_text())
    for row in inventory["inputs"]:
        path = Path(row["local_path"])
        if row["name"] == "B8_s3_final.pth" and path.is_file():
            return path
    pytest.skip("the pinned B8 checkpoint is unavailable")


@pytest.fixture
def frozen_dir(monkeypatch):
    """Worktrees have no saved_snakes/; point the frozen-opponent loader at main's."""
    if not Path("saved_snakes").is_dir():
        if not MAIN_SAVED.is_dir():
            pytest.skip("frozen opponent checkpoints are unavailable")
        monkeypatch.setenv("SNAKE_CKPT_DIR", str(MAIN_SAVED))
    for key in pipeline.THREAD_ENV:
        monkeypatch.setenv(key, "1")
    monkeypatch.setenv("SNAKE_DQN_DEVICE", "cpu")
    torch.set_num_threads(1)


def _world_files(root: Path) -> dict[str, bytes]:
    files = {}
    for path in sorted(root.rglob("world-*.json*")):
        files[path.name] = path.read_bytes()
    return files


# ---------------------------------------------------------------------------
# student label path == B8 label path (byte identity on a tiny real fixture)
# ---------------------------------------------------------------------------
def test_student_fanout_with_b8_student_reproduces_b8_labels_and_replays(
    tmp_path: Path, frozen_dir
) -> None:
    """B8 as the student under process fan-out writes byte-identical exports.

    The two-world, 500-frame development namespace reaches a real searched doom,
    so the search rollouts (``evaluate_rows``) and root/shadow forwards are all
    exercised.  The exports then feed the production ReplaySource unchanged.
    """
    b8 = _b8()
    base = dict(
        checkpoint=b8,
        round="r1",
        shard=0,
        attempt=0,
        worlds=2,
        max_worlds=2,
        frames=500,
        seconds=600,
        horizon=300,
        threads=1,
        nproc=2,
        device="cpu",
        kernel="v2",
        namespace="t34-v2-amend2-preflight/benchmark/03",
        mixes=("scripted", "frozen", "mixed"),
    )
    reference = ratified_label.run_label_chunk({**base, "output_dir": tmp_path / "b8"})
    student = claim_label.run_label_chunk(
        {
            **base,
            "output_dir": tmp_path / "student",
            "policy_factory": FACTORY,
            "student_checkpoint": b8,
        }
    )
    assert reference["complete"] and student["complete"]
    assert sum(mix["searched"] for mix in student["mixes"].values()) >= 1
    assert student["fanout"]["nproc"] == 2
    assert student["student"]["rollout_is_carrier"] is True
    expected, actual = _world_files(tmp_path / "b8"), _world_files(tmp_path / "student")
    assert len([name for name in expected if name.endswith(".json")]) == 2
    assert actual == expected

    # The same exports regenerate through the production source with q checks.
    from research.t34_claim_a.student import load_student_model

    exports = tmp_path / "payload"
    replay_source.build_payload(
        [(tmp_path / "student", 0, "r1-s0-a0")], exports, heldout_fraction=0.0
    )
    previous = load_student_model(b8)
    source = replay_source.ExportReplaySource(
        exports,
        carrier_checkpoint=b8,
        previous_model=previous,
        previous_sha256=replay_source.sha256_file(b8),
    )
    worlds = list(source.replay_epoch(1))
    assert all(world.verified for world in worlds), [w.reason for w in worlds]
    assert source.training_row_counts(1)[0] >= 1
    rows = [row for world in worlds for row in world.rows]
    assert len(rows) == 1000
    assert all(len(world.hash_expected) == 64 for world in worlds)


def _scripted_world(b8: Path, namespace: str = "t34-claim-a/test/policy") -> object:
    from research.t34_exit_b8.trajectory import build_world, make_world_rows, prepare_world

    row = make_world_rows({"namespace": namespace, "mixes": ("scripted",)}, 1)[0]
    return prepare_world(build_world(row, b8))


def test_student_policy_matches_b8_policy_for_plain_and_zero_tower(
    tmp_path: Path, frozen_dir
) -> None:
    from research.t34_claim_a.student import StudentPolicy
    from research.t34_exit_b8.policy import B8Policy
    from research.t34_exit_b8_fit.network import ExitFitTowerNet, save_exit_fit_tower_checkpoint

    b8 = _b8()
    tower_path = tmp_path / "tower.pth"
    save_exit_fit_tower_checkpoint(tower_path, ExitFitTowerNet.from_b8_checkpoint(str(b8)))
    worlds = [_scripted_world(b8), _scripted_world(b8, "t34-claim-a/test/policy2")]
    reference = B8Policy("cpu")
    rows = np.asarray([[0, 0], [0, 1]], dtype=np.int64)
    expected_root = reference.evaluate(worlds[0])
    expected_rows = reference.evaluate_rows(worlds[0].sim, rows, worlds[0].hero.net)
    expected_batch = reference.evaluate_worlds(worlds, bucket_size=2)
    for path, exact in ((b8, True), (tower_path, False)):
        policy = StudentPolicy("cpu", checkpoint=path)
        assert policy.requires_full_sight is (not exact)
        root = policy.evaluate(worlds[0])
        got_rows = policy.evaluate_rows(worlds[0].sim, rows, worlds[0].hero.net)
        batch = policy.evaluate_worlds(worlds, bucket_size=2)
        compare = np.testing.assert_array_equal if exact else np.testing.assert_allclose
        compare(root.q, expected_root.q)
        compare(got_rows, expected_rows)
        for left, right in zip(batch, expected_batch):
            compare(left.q, right.q)
            np.testing.assert_array_equal(left.select, right.select)
        assert root.greedy_action() == expected_root.greedy_action()


# ---------------------------------------------------------------------------
# label entrypoint guards
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "override, message",
    [
        ({"threads": 2}, "two total CPU threads|one thread per process"),
        ({"device": "cuda"}, "CPU and one thread"),
        ({"policy_factory": None}, "policy-factory"),
        ({"horizon": 200}, "H=300"),
        ({"preflight": True}, "production CPU recipe"),
    ],
)
def test_student_fanout_keeps_ratified_bounds(tmp_path: Path, override, message) -> None:
    student = tmp_path / "student.pth"
    student.write_bytes(b"x")
    config = {
        "checkpoint": student,
        "student_checkpoint": student,
        "policy_factory": FACTORY,
        "output_dir": tmp_path / "out",
        "round": "r1",
        "shard": 0,
        "attempt": 0,
        "max_worlds": 2,
        "nproc": 2,
        "threads": 1,
        "device": "cpu",
        "kernel": "v2",
        **override,
    }
    config = {key: value for key, value in config.items() if value is not None}
    with pytest.raises(ValueError, match=message):
        claim_label.run_label_chunk(config)


def test_without_student_the_ratified_entrypoint_runs_verbatim(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        ratified_label, "run_label_chunk", lambda config: calls.append(config) or {"ok": 1}
    )
    assert claim_label.run_label_chunk({"nproc": 4, "kernel": "v2"}) == {"ok": 1}
    assert calls == [{"nproc": 4, "kernel": "v2"}]


def test_student_fanout_sends_only_references_to_workers(monkeypatch, tmp_path: Path) -> None:
    from research.t34_exit_b8 import process_fanout, trajectory

    student = tmp_path / "student.pth"
    student.write_bytes(b"weights")
    seen = {}

    def fake_fanout(config, *, runner, rows, nproc):
        seen.update(config=config, runner=runner, rows=rows, nproc=nproc)
        return {"complete": True, "worlds": len(rows), "fanout": {"nproc": nproc}}

    monkeypatch.setattr(process_fanout, "run_fanout", fake_fanout)
    monkeypatch.setattr(
        trajectory, "make_world_rows", lambda config, count: [{"world_id": i} for i in range(count)]
    )
    monkeypatch.setattr(claim_label.os, "uname", lambda: SimpleNamespace(sysname="Linux"))
    result = claim_label.run_label_chunk(
        {
            "checkpoint": student,
            "student_checkpoint": student,
            "policy_factory": FACTORY,
            "output_dir": tmp_path / "out",
            "round": "r2",
            "shard": 1,
            "attempt": 0,
            "max_worlds": 5,
            "nproc": 12,
            "threads": 1,
            "device": "cpu",
            "kernel": "v2",
        }
    )
    assert seen["runner"] == claim_label.WORKER and seen["nproc"] == 5
    assert seen["config"]["beta"] == 0.5
    assert isinstance(seen["config"]["student_checkpoint"], str)
    assert "policy" not in seen["config"]
    assert result["recipe"]["nproc"] == 5
    assert result["student"]["rollout_is_carrier"] is True


# ---------------------------------------------------------------------------
# payload + ReplaySource (synthetic replay; real replay is covered above)
# ---------------------------------------------------------------------------
def _write_export(directory: Path, index: int, *, rollout: str, carrier: str, decisive: bool):
    from research.t34_exit_b8.exports import SCHEMA

    labels = []
    for decision in range(8):
        label = {
            "decision": decision,
            "label_kind": "shadow_confirm",
            "q": [float(decision + a) for a in range(6)],
            "shadow_id": 0,
        }
        labels.append(label)
    if decisive:
        outcomes = {
            str(a): {"survives": a != 5, "alive_steps": 300 if a != 5 else 4, "tested": True}
            for a in range(6)
        }
        labels[3] = {
            "decision": 3,
            "label_kind": "transition",
            "q": [float(3 + a) for a in range(6)],
            "shadow_id": 2,
            "t_star": 3,
            "a_star": 1,
            "F": [5],
            "per_action": outcomes,
            "student_action": 5,
            "outcome_kind": "horizon_rescue",
            "decisive": True,
            "searched": True,
        }
    value = {
        "schema": SCHEMA,
        "world_id": f"test/{index:03d}",
        "row": {"world_id": f"test/{index:03d}"},
        "checkpoint_sha256": carrier,
        "rollout_policy_sha256": rollout,
        "actions": [[1] * 6 for _ in range(8)],
        "hashes": {"0": "a", "8": "b"},
        "labels": labels,
        "teacher_v2": {"kernel": replay_source.TEACHER_KERNEL},
    }
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"world-{index:06d}.json"
    encoded = json.dumps(value).encode()
    path.write_bytes(encoded)
    path.with_name(path.name + ".sha256").write_text(
        f"{hashlib.sha256(encoded).hexdigest()}  {path.name}\n"
    )
    path.with_name(path.name + ".complete").write_text("complete\n")
    return path


class _ConstantQ(torch.nn.Module):
    """Previous-student double: Q = decision index + action, read from scalars[0]."""

    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, local, glob, scalars):
        del local, glob
        return scalars[:, :1].float() + torch.arange(6, dtype=torch.float32) + 0 * self.anchor


def _fake_replay(exported, checkpoint, *, build_world, observe):
    from research.t34_exit_b8.replay import ReplayResult

    world = build_world(exported["row"], checkpoint)
    observations = []
    for label in exported["labels"]:
        scalars = np.zeros(60, dtype=np.float32)
        scalars[0] = float(label["decision"])
        observations.append(
            {
                "decision": label["decision"],
                "local": np.zeros((7, 31, 31), dtype=np.uint8),
                "global": np.zeros((4, 37, 37), dtype=np.uint8),
                "scalars": scalars,
                "length": 10 + int(label["decision"]),
                "label": dict(label),
            }
        )
    del world
    return ReplayResult(str(exported["world_id"]), True, tuple(observations))


@pytest.fixture
def payload(tmp_path: Path, monkeypatch):
    carrier = tmp_path / "carrier.pth"
    carrier.write_bytes(b"carrier")
    carrier_sha = replay_source.sha256_file(carrier)
    previous_sha = "f" * 64
    raw = tmp_path / "raw"
    for index in range(4):
        _write_export(
            raw, index, rollout=previous_sha, carrier=carrier_sha, decisive=index in (1, 2, 3)
        )
    exports = tmp_path / "payload"
    replay_source.build_payload([(raw, 0, "chunk-a")], exports, heldout_decisive=1)
    # The synthetic exports record "b" as their final-frame state hash.
    monkeypatch.setattr("research.t34_exit_b8.exports.state_hash", lambda sim: "b")
    return SimpleNamespace(
        exports=exports, carrier=carrier, previous_sha=previous_sha, raw=raw, tmp=tmp_path
    )


def _source(payload, **kwargs):
    return replay_source.ExportReplaySource(
        payload.exports,
        carrier_checkpoint=payload.carrier,
        previous_model=_ConstantQ(),
        previous_sha256=payload.previous_sha,
        build_world=lambda row, checkpoint: SimpleNamespace(sim=row["world_id"]),
        replay_fn=_fake_replay,
        **kwargs,
    )


def test_payload_is_self_contained_hash_bound_and_splits_on_decisive_worlds(payload) -> None:
    value = replay_source.load_payload(payload.exports)
    splits = {w["world_id"]: w["split"] for w in value["worlds"]}
    assert sorted(splits.values()).count("heldout") == 1
    held = next(w for w, s in splits.items() if s == "heldout")
    assert held != "test/000"  # world 0 carries no decisive label
    for world in value["worlds"]:
        path = payload.exports / world["path"]
        assert path.is_file() and path.resolve().is_relative_to(payload.exports.resolve())
        assert replay_source.sha256_file(path) == world["sha256"]
    with pytest.raises(FileExistsError):
        replay_source.build_payload([(payload.raw, 0, "chunk-a")], payload.exports)
    (payload.exports / "payload.json").write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        replay_source.load_payload(payload.exports)


def test_payload_refuses_mixed_carriers_and_duplicate_worlds(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _write_export(raw / "a", 0, rollout="r" * 64, carrier="a" * 64, decisive=True)
    _write_export(raw / "b", 1, rollout="r" * 64, carrier="b" * 64, decisive=True)
    with pytest.raises(ValueError, match="one carrier"):
        replay_source.build_payload([(raw, 0, "c")], tmp_path / "p1")
    _write_export(raw / "c", 0, rollout="r" * 64, carrier="a" * 64, decisive=True)
    with pytest.raises(ValueError, match="duplicate world id"):
        replay_source.build_payload([(raw / "a", 0, "c"), (raw / "c", 0, "d")], tmp_path / "p2")


def test_source_rows_satisfy_the_fit_contract_and_feed_cache_and_loader(payload) -> None:
    from research.t34_exit_b8_fit.cache import build_cache
    from research.t34_exit_b8_fit.data import StreamingReplayLoader
    from research.t34_exit_b8_fit.interface import normalize_row, validate_row

    source = _source(payload)
    assert source.training_row_counts(1) == (2, 22)
    worlds = list(source.replay_epoch(1))
    assert len(worlds) == 3 and all(world.verified for world in worlds)
    for world in worlds:
        assert world.hash_expected == world.hash_actual and len(world.hash_actual) == 64
        for row in world.rows:
            normalized = normalize_row(row)
            validate_row(normalized)
            assert normalized["q_prev"].dtype == np.float32
            if normalized["searched"]:
                assert normalized["death_window_id"] == 2 and normalized["decisive"]
                assert normalized["forbidden_mask"].tolist() == [False] * 5 + [True]
    assert source.training_row_counts(1) == (2, 22)
    held = list(source.replay_heldout(1))
    assert len(held) == 1 and sum(r.get("label_kind") == "transition" for r in held[0].rows) == 1

    cached = build_cache(source, payload.tmp / "cache", identity={"test": 1})
    assert cached.training_row_counts(1) == (2, 22)
    batches = list(StreamingReplayLoader(cached, batch_size=3, seed=0).epoch(1))
    assert sum(len(batch["action_star"]) for batch in batches) == 4  # 2 searched + 2 anchors


def test_q_prev_mismatch_against_the_bound_rollout_student_drops_the_world(payload) -> None:
    class Shifted(_ConstantQ):
        def forward(self, local, glob, scalars):
            return super().forward(local, glob, scalars) + 1.0

    source = _source(payload)
    source.previous_model = Shifted()
    worlds = list(source.replay_epoch(1))
    assert not any(world.verified for world in worlds)
    assert all("q_prev mismatch" in world.reason for world in worlds)
    # An older-round export (different rollout student) is not compared.
    source = _source(payload)
    source.previous_model = Shifted()
    source.previous_sha256 = "0" * 64
    assert all(world.verified for world in source.replay_epoch(1))


def test_replay_divergence_is_reported_not_raised(payload) -> None:
    from research.t34_exit_b8.replay import ReplayResult

    source = _source(payload)
    source.replay_fn = lambda exported, checkpoint, **_: ReplayResult(
        exported["world_id"], False, (), "frame 250: x != y"
    )
    worlds = list(source.replay_epoch(1))
    assert [world.reason for world in worlds] == ["frame 250: x != y"] * 3
    assert source.timings.dropped["train"] == 3


def test_source_factory_reads_the_claim_a_binding(payload) -> None:
    binding = {
        "arm": "S1b",
        "claim_a": {"carrier_checkpoint": str(payload.carrier), "previous_sha256": "f" * 64},
    }
    source = replay_source.source_factory(
        exports=payload.exports, binding=binding, previous_model=_ConstantQ()
    )
    assert source is replay_source.last_source() and not source.include_full_sight
    with pytest.raises(ValueError, match="carrier_checkpoint"):
        replay_source.source_factory(
            exports=payload.exports, binding={"claim_a": {}}, previous_model=_ConstantQ()
        )
    other = payload.tmp / "other.pth"
    other.write_bytes(b"not the carrier")
    with pytest.raises(ValueError, match="carrier checkpoint differs"):
        replay_source.source_factory(
            exports=payload.exports,
            binding={"claim_a": {"carrier_checkpoint": str(other)}},
            previous_model=_ConstantQ(),
        )


# ---------------------------------------------------------------------------
# probe boundary
# ---------------------------------------------------------------------------
def test_dev_probe_records_complete_aligned_traces(frozen_dir) -> None:
    from research.t34_claim_a.student import load_student_model

    b8 = _b8()
    rows = probe_rows("t34-claim-a/test/probe", 2, ["scripted"])
    probe = DevProbe(
        carrier_checkpoint=b8, previous_model=load_student_model(b8), rows=rows, horizon=12
    )
    result = probe(load_student_model(b8), 1)
    assert not result.is_gate_data and len(result.trajectories) == 2
    for record in result.trajectories:
        assert record["horizon"] == 12 and record["complete"]
        assert record["actions"] == record["previous_actions"]
        assert len(record["actions"]) == len(record["lengths"]) <= 12
    metrics = validate_smoke_probe(result, worlds=2, horizon=12)
    assert metrics["max_decile_disagreement"] == 0.0
    assert probe.calls[0]["decisions"] == sum(len(t["actions"]) for t in result.trajectories)
    with pytest.raises(ValueError, match="probe namespace"):
        probe_rows("t34-claim-a/test/label", 2)


def test_smoke_shape_and_tripwire_recording_restore_the_ratified_functions() -> None:
    from research.t34_exit_b8_fit import train

    validate, enforce = train.validate_probe, train.enforce_tripwires
    with pipeline.smoke_probe_shape(32, 5000):
        assert train.validate_probe is validate
    log: list = []
    with pipeline.smoke_probe_shape(2, 300), pipeline.record_tripwires(True, log):
        assert train.validate_probe is not validate
        train.enforce_tripwires(
            loss=1.0,
            heldout_decisive_agreement=0.1,
            probe={
                "max_action_share": 0.5,
                "max_decile_disagreement": 0.0,
                "boost_share": 0.0,
                "previous_boost_share": 0.0,
            },
        )
        with pytest.raises(ValueError, match="smoke probe worlds"):
            train.validate_probe(ProbeResult(1.0, ()))
    assert train.validate_probe is validate and train.enforce_tripwires is enforce
    assert log == [{"fired": "held-out decisive agreement below 70%", "heldout": 0.1}]


# ---------------------------------------------------------------------------
# manifests and shell entrypoints
# ---------------------------------------------------------------------------
@pytest.fixture
def qualified(monkeypatch):
    from research.t34_exit_b8 import make_manifest as ratified

    monkeypatch.setattr(ratified, "require_qualified_transport", lambda *_args: {})
    monkeypatch.setattr(claim_manifest, "require_qualified_transport", lambda *_args: {})


def test_label_manifest_binds_a_nondefault_student(tmp_path: Path, qualified) -> None:
    student = tmp_path / "student_r1.pth"
    student.write_bytes(b"student weights")
    sha = replay_source.sha256_file(student)
    manifest = claim_manifest.make_label_manifest(
        commit="a" * 40,
        output_dir=tmp_path / "out",
        student={"path": student, "sha256": sha, "size_bytes": 15, "name": "student_r1.pth"},
        round_id="r2",
        shard=3,
        attempt=1,
        worlds=24,
        measured_export_bytes_per_world=9_000_000,
    )
    assert manifest["script"] == "research/t34_claim_a/label.sh"
    assert manifest["job_id"] == f"t34-claim-a-r2-s3-a1-{sha[:8]}"
    assert manifest["env"]["T34_STUDENT_SHA256"] == sha
    assert manifest["env"]["T34_STUDENT_CHECKPOINT"] == "student/student_r1.pth"
    assert manifest["env"]["T34_POLICY_FACTORY"] == FACTORY
    assert manifest["env"]["T34_BETA"] == "0.5" and manifest["env"]["T34_DEVICE"] == "cpu"
    assert {"path": str(student.resolve()), "sha256": sha, "dest": "student/student_r1.pth"} in (
        manifest["inputs"]
    )
    assert "research/t34_claim_a" in manifest["source_paths"]
    with pytest.raises(claim_manifest.ManifestRefused, match="differ"):
        claim_manifest.make_label_manifest(
            commit="a" * 40,
            output_dir=tmp_path / "out",
            student={"path": student, "sha256": "0" * 64, "size_bytes": 15, "name": "s.pth"},
            round_id="r2",
            shard=3,
            attempt=1,
            worlds=24,
            measured_export_bytes_per_world=9_000_000,
        )


def test_student_input_rows_are_exclusive_and_well_formed(tmp_path: Path) -> None:
    row = claim_manifest.student_input(
        sha256="b" * 64, size_bytes=10, name="s.pth", url="https://example.invalid/s.pth"
    )
    assert row == {
        "url": "https://example.invalid/s.pth",
        "sha256": "b" * 64,
        "size_bytes": 10,
        "dest": "student/s.pth",
    }
    for bad in (
        {"sha256": "b" * 64, "size_bytes": 10, "name": "s.pth"},
        {"sha256": "b" * 64, "size_bytes": 10, "name": "../s.pth", "url": "https://x/s.pth"},
        {"sha256": "B" * 64, "size_bytes": 10, "name": "s.pth", "url": "https://x/s.pth"},
        {"sha256": "b" * 64, "size_bytes": 10, "name": "s.pth", "url": "http://x/s.pth"},
    ):
        with pytest.raises(claim_manifest.ManifestRefused):
            claim_manifest.student_input(**bad)


def test_pipeline_manifest_is_l4_public_only_and_priced(tmp_path: Path, qualified) -> None:
    manifest = claim_manifest.make_pipeline_manifest(commit="c" * 40, output_dir=tmp_path / "o")
    assert manifest["script"] == "research/t34_claim_a/pipeline.sh"
    assert manifest["requirements"]["allowed_gpus"] == ["L4"]
    assert manifest["timeout_minutes"] == 2 * manifest["requirements"]["est_minutes"]
    budget = manifest["budget"]
    assert manifest["timeout_minutes"] * budget["rate_units_per_hour"] / 60 <= budget["unit_cap"]
    assert [item["dest"] for item in manifest["inputs"]] == ["public-inputs.json"]
    assert manifest["env"]["T34_CA_DEVICE"] == "cuda"
    with pytest.raises(claim_manifest.ManifestRefused, match="unit cap"):
        claim_manifest.make_pipeline_manifest(
            commit="c" * 40, output_dir=tmp_path / "o", unit_cap=1.0
        )


@pytest.mark.parametrize("script", ["label.sh", "pipeline.sh"])
def test_shell_entrypoints_parse(script: str) -> None:
    subprocess.run(["bash", "-n", f"research/t34_claim_a/{script}"], check=True)
    text = Path(f"research/t34_claim_a/{script}").read_text()
    assert "research.t34_claim_a." in text


def test_full_fp32_restores_tf32_switches() -> None:
    from research.t34_claim_a.student import full_fp32

    matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    with full_fp32():
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.allow_tf32 is False
    assert torch.backends.cuda.matmul.allow_tf32 == matmul
    assert torch.backends.cudnn.allow_tf32 == cudnn
