"""Integrity, leak prevention, clustered inference and exact continuation tests."""

from __future__ import annotations

import copy
import json
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from research.doom_auc_smoke.aggregate import AUC, aggregate, analyze, bars
from research.doom_auc_smoke.data import load_gzip, save_gzip
from research.doom_auc_smoke.data_mac import audit_split, compact_snapshot
from research.doom_auc_smoke.evidence import (
    build_ledger,
    digest,
    empty_ledger,
    latest,
    load_unit,
    publish,
    validate_ledger,
    verified,
)
from research.doom_auc_smoke.fixtures import synthetic_dataset
from research.doom_auc_smoke.model import make_model, masked_bce
from research.doom_auc_smoke.prepare import manifest
from research.doom_auc_smoke.run import train_arm
from research.spatial_student_pilot.observation import (
    build_observation,
    build_observation_cpu_reference,
)


@pytest.mark.parametrize("values,expected", [([0, 1], 1), ([1, 0], 0), ([0, 0], 0.5)])
def test_auc_direction_and_ties(values: list, expected: float) -> None:
    metric = AUC(np.array(values), np.array([0, 1]), np.array([0, 1]))
    assert metric.value(np.ones(2)) == expected


def test_parent_world_weights_keep_all_events() -> None:
    scores = np.array([0.1, 0.9, 0.2, 0.5, 0.3, 0.7])
    labels = np.array([0, 1, 1, 0, 0, 1])
    worlds = np.array([0, 0, 1, 1, 2, 2])
    metric = AUC(scores, labels, worlds)
    ids = np.repeat(np.arange(6), np.array([2, 2, 0, 0, 1, 1]))
    explicit = AUC(scores[ids], labels[ids], worlds[ids])
    assert metric.value(np.array([2, 0, 1])) == explicit.value(np.ones(3))


@pytest.mark.parametrize(
    "auc,expected", [(0.749999, "KILL"), (0.75, "GRAY"), (0.849999, "GRAY"), (0.85, "GO")]
)
def test_frozen_bar_boundaries(auc: float, expected: str) -> None:
    assert bars(0.99, auc) == expected


def test_loss_masks_illegal_targets_and_is_suffix_bce() -> None:
    logits = torch.tensor([[100.0, -100.0, 9.0, 0.0, 0.0, 0.0]], requires_grad=True)
    target = torch.tensor([[0.25, 0.75, float("nan"), 0.0, 0.0, 0.0]])
    legal = torch.tensor([[True, True, False, False, False, False]])
    loss = masked_bce(logits, target, legal)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    assert not logits.grad[~legal].any()
    suffix = torch.tensor([[[0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 0.0]]])
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[:, :2, None].expand_as(suffix), suffix
    )
    assert loss == expected


def test_world_partition_mismatch_refused() -> None:
    meta = [
        SimpleNamespace(root_index=0, root_uid="a", world_id=0, partition="train"),
        SimpleNamespace(root_index=1, root_uid="b", world_id=0, partition="assessment"),
    ]
    with pytest.raises(ValueError, match="leakage"):
        audit_split(meta, ["train", "assessment"])


def test_compact_conversion_body_order_timers_food() -> None:
    fields = dict(
        length=torch.tensor([700, 1, 1, 1, 1, 1]),
        alive=torch.ones(6, dtype=torch.bool),
        direction=torch.arange(6) % 4,
        boost_frames=torch.arange(6),
        frames_since_food=torch.arange(6) * 10,
        frame=torch.tensor(100),
    )
    snapshot = SimpleNamespace(
        body_offsets=np.arange(7),
        body_segments_xy=np.array([[i, 3] for i in range(6)], np.int16),
        small_tensors=fields,
        prepared=True,
        num_snakes=6,
        food_xy=np.array([[7, 5]], np.int16),
        food_corpse=np.array([1], np.uint8),
    )
    compact = compact_snapshot(snapshot, 83, 145)
    assert compact["boost_cost_frames"] == 3
    assert compact["heads"].tolist() == [[0, 3]]
    assert compact["hunger"].tolist() == [[0, 10, 20, 30, 40, 50]]
    assert compact["food_kind"].tolist() == [[2]]
    actual = build_observation(compact)
    expected = build_observation_cpu_reference(compact)
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a.numpy(), b, atol=1e-7)


def test_lossless_input_io(tmp_path: Path) -> None:
    data = synthetic_dataset(8, 8)
    path = tmp_path / "dataset.pt.gz"
    save_gzip(path, data)
    restored = load_gzip(path)
    assert torch.equal(restored["phi"], data["phi"])
    assert torch.equal(restored["compact"][0]["body_cells"], data["compact"][0]["body_cells"])
    with pytest.raises(FileExistsError):
        save_gzip(path, data)


def test_units_foreign_duplicates_partial_and_corruption(tmp_path: Path) -> None:
    row = publish(tmp_path, "spatial-fit-0250", "id", dict(step=250))
    assert load_unit(tmp_path, row, "id")["step"] == 250
    with pytest.raises(ValueError, match="foreign"):
        verified(tmp_path, row, "other")
    ledger = dict(empty_ledger("id"), units=[row, row])
    with pytest.raises(ValueError, match="duplicate"):
        validate_ledger(ledger, "id")
    with pytest.raises(FileExistsError):
        publish(tmp_path, "spatial-fit-0250", "id", {})
    marker = Path(str(tmp_path / row["path"]) + ".complete")
    marker.unlink()
    with pytest.raises(FileNotFoundError):
        verified(tmp_path, row, "id")


def receipt_for(root: Path) -> None:
    exports = [
        dict(path=str(p.relative_to(root / "exports")), size=p.stat().st_size, sha256=digest(p))
        for p in sorted((root / "exports").rglob("*"))
        if p.is_file()
    ]
    (root / "receipt.json").write_text(json.dumps(dict(stop_verified=True, exports=exports)))


def test_outcome_blind_ledger_receipt_binding(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "attempt"
    publish(output / "exports/doom", "b8-fit-0250", "id", dict(secret_outcome="unread"))
    receipt_for(output)
    monkeypatch.setattr(torch, "load", lambda *_a, **_kw: pytest.fail("ledger opened outcomes"))
    ledger = build_ledger([output], "id")
    assert len(ledger["units"]) == 1
    assert latest(ledger, "b8")["unit"] == "b8-fit-0250"
    with pytest.raises(ValueError, match="duplicate"):
        build_ledger([output, output], "id")
    (output / "exports/doom" / ledger["units"][0]["path"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="bytes differ"):
        build_ledger([output], "id")


@pytest.mark.parametrize("arm", ["spatial", "b8"])
def test_exact_interrupted_resume_and_assessment_not_fit(
    tmp_path: Path, monkeypatch, arm: str
) -> None:
    from research.doom_auc_smoke import run as runner

    torch.set_num_threads(1)
    data = synthetic_dataset(8, 8)
    full = tmp_path / "full"
    start = tmp_path / "first"
    resumed = tmp_path / "resumed"
    inputs = tmp_path / "inputs"
    for p in (full, start, resumed, inputs):
        p.mkdir()
    kwargs = dict(
        identity="id",
        device="cpu",
        updates=6,
        batch=2,
        deadline=time.monotonic() + 60,
        checkpoint_every=2,
    )
    assert train_arm(arm, data, full, inputs, empty_ledger("id"), stopped=[False], **kwargs)
    original = runner.forward
    stopped = [False]
    calls = [0]

    def interrupt(*args, **kw):
        calls[0] += 1
        # This closure receives only the physical train view during fitting.
        assert len(args[2]["compact"]) == 8
        result = original(*args, **kw)
        if calls[0] == 2:
            stopped[0] = True
        return result

    monkeypatch.setattr(runner, "forward", interrupt)
    assert not train_arm(arm, data, start, inputs, empty_ledger("id"), stopped=stopped, **kwargs)
    monkeypatch.setattr(runner, "forward", original)
    row = json.loads(next(start.glob("units/*.complete")).read_text())
    row = {k: row[k] for k in ("unit", "path", "size", "sha256")}
    for source in start.glob("units/*"):
        dest = inputs / source.relative_to(start)
        dest.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, dest)
    changed = copy.deepcopy(data)
    changed["phi"][8:] *= 100
    ledger = dict(empty_ledger("id"), units=[row])
    assert train_arm(arm, changed, resumed, inputs, ledger, stopped=[False], **kwargs)
    final_name = f"units/a0000-{arm}-fit-0006.pt"
    left = torch.load(full / final_name, weights_only=True)
    right = torch.load(resumed / final_name, weights_only=True)
    assert all(torch.equal(v, right["model"][k]) for k, v in left["model"].items())
    assert torch.equal(left["sampler"], right["sampler"])
    # A verified final prediction is skipped, even with a past deadline.
    pred = json.loads((full / f"units/a0000-{arm}-prediction.pt.complete").read_text())
    pred = {k: pred[k] for k in ("unit", "path", "size", "sha256")}
    skipped = tmp_path / "skip"
    assert train_arm(
        arm,
        data,
        skipped,
        full,
        dict(empty_ledger("id"), units=[pred]),
        stopped=[False],
        **dict(kwargs, deadline=0),
    )
    assert not skipped.exists()


def test_bootstrap_support_paired_ci_and_held_labels() -> None:
    n = 24
    doom = (torch.arange(n) % 2).bool()[:, None, None].expand(n, 6, 4)
    held = dict(
        world=torch.arange(n),
        length=torch.full((n,), 600),
        doom=doom,
        legal=torch.ones((n, 6), dtype=torch.bool),
        action=torch.zeros(n, dtype=torch.long),
    )
    scores = doom[:, :, 0].float()
    report = analyze(held, dict(spatial=scores, b8=scores), resamples=100)
    assert report["verdict"] == "GO"
    primary = report["endpoints"]["late/b8_action"]
    assert primary["support"]["events"] == 4 * n
    assert primary["spatial_minus_b8"]["ci95"] == [0.0, 0.0]
    held["world"] = torch.zeros(n, dtype=torch.long)
    assert (
        analyze(held, dict(spatial=scores, b8=scores), resamples=100)["verdict"]
        == "INCOMPLETE_SUPPORT"
    )


def test_aggregate_missing_units_prevents_look(tmp_path: Path) -> None:
    packet = tmp_path / "packet"
    packet.mkdir()
    (packet / "config.json").write_text(json.dumps(dict(identity="id")))
    result = tmp_path / "result"
    with pytest.raises(ValueError, match="INCOMPLETE"):
        aggregate(packet, [], result)
    assert not result.exists()


def test_hardware_envelopes_and_no_assessment_upload(tmp_path: Path) -> None:
    config = dict(commit="a" * 40, registration_sha256="b" * 64)
    item = manifest(config, [], [], tmp_path / "out", "test-doom", "fit")
    assert item["timeout_minutes"] == 120
    assert item["budget"]["unit_cap"] == 24
    assert item["resume"]["max_outcome_blind_retries"] == 2
    from research.compute.gpu_preference import variants

    fallback = variants(item)[1]
    assert fallback["budget"]["unit_cap"] == 16
    assert fallback["requirements"]["gpu"] == "A100"
    assert "canary" not in item  # Genuine receipt required by supervisor, no invented admission.
    assert sum(p.numel() for p in make_model("spatial").parameters()) < 700000


def test_aggregate_one_look_even_with_new_result_path(tmp_path: Path) -> None:
    packet, output = tmp_path / "packet", tmp_path / "out"
    packet.mkdir()
    n = 24
    doom = (torch.arange(n) % 2).bool()[:, None, None].expand(n, 6, 4)
    ids = [str(i) for i in range(n)]
    held = dict(
        root_uids=ids,
        world=torch.arange(n),
        length=torch.full((n,), 600),
        doom=doom,
        legal=torch.ones((n, 6), dtype=torch.bool),
        action=torch.zeros(n, dtype=torch.long),
    )
    save_gzip(packet / "assessment-MAC-ONLY.pt.gz", held)
    config = dict(
        identity="id",
        assessment_sha256=digest(packet / "assessment-MAC-ONLY.pt.gz"),
        registration_sha256="reg",
        mac_look_file=str(tmp_path / "look.json"),
    )
    (packet / "config.json").write_text(json.dumps(config))
    for arm in ("spatial", "b8"):
        publish(
            output / "exports/doom",
            arm + "-prediction",
            "id",
            dict(steps=3000, root_uids=ids, logits=doom[:, :, 0].float()),
        )
    receipt_for(output)
    assert aggregate(packet, [output], tmp_path / "first")["verdict"] == "GO"
    with pytest.raises(FileExistsError):
        aggregate(packet, [output], tmp_path / "second")


def test_completion_eligibility_mechanical_only() -> None:
    from research.doom_auc_smoke.resume import eligible_failure

    eligible_failure(dict(status="INCOMPLETE", stop_verified=True, error="lease timeout"))
    for receipt in (
        dict(status="PASS", stop_verified=True, error="timeout"),
        dict(status="INCOMPLETE", stop_verified=False, error="timeout"),
        dict(status="INCOMPLETE", stop_verified=True, error="nonfinite timeout"),
    ):
        with pytest.raises(ValueError, match="mechanical"):
            eligible_failure(receipt)


def test_completion_uses_latest_and_requires_full_history(tmp_path: Path) -> None:
    from research.doom_auc_smoke.evidence import json_once
    from research.doom_auc_smoke.resume import completion

    prior, output = tmp_path / "prior", tmp_path / "out"
    prior.mkdir()
    config = dict(identity="id", commit="a" * 40, registration_sha256="b" * 64)
    for name in ("config.json", "REGISTRATION.md", "dataset.pt.gz", "assessment-MAC-ONLY.pt.gz"):
        (prior / name).write_text(json.dumps(config) if name == "config.json" else "fixture")
    for mode in ("canary", "fit"):
        item = manifest(config, [], [], prior / (mode + "-output"), "test-" + mode, mode)
        json_once(prior / ("canary.json" if mode == "canary" else "fit.pending.json"), item)
    for step in (250, 500):
        publish(output / "exports/doom", f"spatial-fit-{step:04d}", "id", dict(step=step))
    receipt_for(output)
    receipt = json.loads((output / "receipt.json").read_text())
    receipt.update(status="INCOMPLETE", error="lease timeout")
    (output / "receipt.json").write_text(json.dumps(receipt))
    dest = tmp_path / "completion"
    ledger = completion(prior, [output], dest)
    assert [r["unit"] for r in ledger["units"]] == ["spatial-fit-0500"]
    assert len(json.loads((dest / "prior-union-MAC-ONLY.json").read_text())["units"]) == 2
    with pytest.raises(ValueError, match="history"):
        completion(dest, [output], tmp_path / "bad-next")
