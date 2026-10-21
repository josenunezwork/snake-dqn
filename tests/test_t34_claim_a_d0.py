"""Tiny-fixture tests for the L1 D0 rule, input tarball, runner wiring and manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from research.t34_claim_a_loss import decide as D
from research.t34_claim_a_loss import harness as H
from research.t34_claim_a_loss import inputs as dinputs
from research.t34_claim_a_loss import make_manifest as M

torch.set_num_threads(1)


# ---------------------------------------------------------------------------
# synthetic report for the frozen rule
# ---------------------------------------------------------------------------
def _cell(name, *, train=0.95, held=lambda u: 0.5, bad_from=None, folds=3):
    spec = H.DEFAULT_CELLS[name]
    points = [u for u in spec.checkpoints]
    if spec.rule == "ratified":
        points = [0, 1, 2, 3, 4, 5]

    def anchor(u):
        bad = bad_from is not None and u >= bad_from
        return {
            "n": 100,
            "agree": 0.99,
            "max_decile_disagree": 0.08 if bad else 0.02,
            "boost_ratio": 1.0,
            "max_action_share": 0.4,
        }

    def flip(u):
        return 0.0 if u == 0 else (train(u) if callable(train) else train)

    fold_rows = [
        {
            "fold": f,
            "curve": [
                {
                    "updates": u,
                    "train_decisive": {"n": 50, "flip": flip(u)},
                    "heldout_anchor": anchor(u),
                }
                for u in points
            ],
        }
        for f in range(folds)
    ]
    pooled = [
        {
            "updates": u,
            "heldout_decisive": {"flip": 0.0 if u == 0 else held(u)},
            "decisive_passes_mean": u * 128 / 150,
        }
        for u in points
    ]
    cell = {
        "name": name,
        "loss": spec.loss,
        "trunk": spec.trunk,
        "lr": spec.lr,
        "rule": spec.rule,
        "checkpoints": points,
    }
    return {"cell": cell, "folds": fold_rows, "pooled": pooled}


def _report(overrides=None, free=None):
    overrides = overrides or {}
    free = {"frozen": 0.004, "frozen_tau2": 0.5, "split": 1.0, "split_tau2": 1.0} | (free or {})
    return {
        "free_q": [{"loss": k, "flip": v} for k, v in free.items()],
        "cells": [_cell(name, **overrides.get(name, {})) for name in H.D0_CELLS],
    }


def test_d0_rule_passes_with_default_dose_and_tie_breaks() -> None:
    out = D.decide(_report())
    assert out["decision"] == "PASS"
    for trunk in ("trainable", "residual"):
        sel = out["selection"][trunk]
        assert sel["status"] == "SELECTED" and sel["updates"] == 2048
        assert sel["updates_per_block"] == 512 and sel["batch"]["decisive"] == 128
    # all candidates tie on held-out agreement -> smaller lr, then split before split_tau2
    assert out["selection"]["trainable"]["cell"] == "split_trainable"
    assert out["selection"]["residual"]["cell"] == "split_residual"
    assert set(out["controls"]) == {"frozen_recipe", "frozen_trainable", "frozen_residual"}


def test_d0_rule_raises_dose_only_on_a_guarded_five_point_gain() -> None:
    gain = {"held": lambda u: 0.5 + (0.06 if u >= 4096 else 0.0)}
    out = D.decide(_report({"split_trainable": gain}))
    assert out["selection"]["trainable"]["updates"] == 4096
    small = {"held": lambda u: 0.5 + (0.04 if u >= 4096 else 0.0)}
    assert D.decide(_report({"split_trainable": small}))["selection"]["trainable"]["updates"] == (
        2048
    )


def test_d0_rule_falls_back_when_guards_break() -> None:
    bad = {name: {"bad_from": 2048} for name in H.D0_CELLS if "residual" in name}
    out = D.decide(_report(bad))
    assert out["selection"]["residual"]["updates"] == 1024
    worse = {name: {"bad_from": 512} for name in H.D0_CELLS if "split" in name}
    out = D.decide(_report(worse))
    assert out["decision"] == "KILL" and "guards" in out["reason"]


def test_d0_rule_kills_on_free_q_and_permissive_bars() -> None:
    out = D.decide(_report(free={"split": 0.6, "split_tau2": 0.65}))
    assert out["decision"] == "KILL" and "free Q" in out["reason"]
    # free-Q between 70% and 99%: losses are ineligible, so no candidate survives
    out = D.decide(_report(free={"split": 0.9, "split_tau2": 0.9}))
    assert out["decision"] == "KILL" and "permissive" in out["reason"]
    low = {name: {"train": 0.69} for name in H.D0_CELLS if "split" in name}
    out = D.decide(_report(low))
    assert out["decision"] == "KILL" and "permissive" in out["reason"]
    # one trunk failing permissively leaves the other selected
    low_res = {name: {"train": 0.5} for name in H.D0_CELLS if name.startswith("split_res")}
    low_res["split_tau2_residual"] = {"train": 0.5}
    out = D.decide(_report(low_res))
    assert out["decision"] == "PASS" and out["selection"]["residual"]["status"] == "INELIGIBLE"


def test_d0_rule_refuses_an_incomplete_report() -> None:
    report = _report()
    report["cells"] = report["cells"][:-1]
    with pytest.raises(D.DecisionRefused, match="split_tau2_residual"):
        D.decide(report)


def test_d0_rule_handles_empty_training_metrics_without_relaxing_guards() -> None:
    report = _report()
    for cell in report["cells"]:
        for point in cell["folds"][0]["curve"]:
            point["train_decisive"] = {"n": 0}
            point["heldout_anchor"] = {"n": 0}
        if cell["cell"]["rule"] == "streams":
            assert D.train_agreement(cell, 512) == pytest.approx(0.95)
            assert not D.guards_hold(cell, 512)
            assert D.dose_for(cell) is None
    decision = D.decide(report)
    assert decision["decision"] == "KILL" and "guards" in decision["reason"]
    cell = report["cells"][1]
    for fold in cell["folds"]:
        for point in fold["curve"]:
            point["train_decisive"] = {"n": 0}
    assert D.train_agreement(cell, 512) == 0.0


# ---------------------------------------------------------------------------
# input tarball
# ---------------------------------------------------------------------------
def _export(root: Path, n: int, world_id: str, b8: str = dinputs.B8_SHA256) -> None:
    path = root / f"process-{n:03d}" / "worlds" / f"world-{n:06d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(
        {"world_id": world_id, "checkpoint_sha256": b8, "rollout_policy_sha256": b8}
    ).encode()
    path.write_bytes(data)
    path.with_name(path.name + ".sha256").write_text(
        f"{hashlib.sha256(data).hexdigest()}  {path.name}\n"
    )
    path.with_name(path.name + ".complete").write_text("complete\n")


def test_inputs_pack_is_deterministic_and_unpack_verifies(tmp_path: Path) -> None:
    roots = [tmp_path / "s0", tmp_path / "s1"]
    for r, root in enumerate(roots):
        for n in range(2):
            _export(root, n, f"t34-exit-a5/v2/s{r}/a0/scripted/{n:06d}")
    a = dinputs.pack(roots, tmp_path / "a.tar.gz", expect_worlds=4)
    b = dinputs.pack(roots, tmp_path / "b.tar.gz", expect_worlds=4)
    assert a["sha256"] == b["sha256"] and a["worlds"] == 4
    paths = dinputs.unpack(tmp_path / "a.tar.gz", a["sha256"], tmp_path / "x")
    assert len(paths) == 4 and all(p.is_file() for p in paths)
    with pytest.raises(dinputs.InputsRefused, match="SHA-256"):
        dinputs.unpack(tmp_path / "a.tar.gz", "0" * 64, tmp_path / "y")
    with pytest.raises(dinputs.InputsRefused, match="expected 5"):
        dinputs.pack(roots, tmp_path / "c.tar.gz", expect_worlds=5)
    _export(tmp_path / "s2", 0, "t34-claim-a/r1-labels-v4/train/s0/a0/scripted/000000")
    with pytest.raises(dinputs.InputsRefused, match="not an excluded A5"):
        dinputs.pack([tmp_path / "s2"], tmp_path / "d.tar.gz", expect_worlds=1)
    _export(tmp_path / "s3", 0, "t34-exit-a5/v2/s9/a0/scripted/000000", b8="c" * 64)
    with pytest.raises(dinputs.InputsRefused, match="B8 s3"):
        dinputs.pack([tmp_path / "s3"], tmp_path / "e.tar.gz", expect_worlds=1)


# ---------------------------------------------------------------------------
# runner wiring on a tiny cache (residual trunk, CPU)
# ---------------------------------------------------------------------------
def _write_cache(tmp_path: Path, bank: H.Bank | None = None) -> Path:
    from tests.test_t34_claim_a_loss import tiny_bank

    bank = tiny_bank() if bank is None else bank
    cache = tmp_path / "cache"
    cache.mkdir()
    receipts = []
    for w, world_id in enumerate(bank.world_ids):
        rows = bank.world == w
        name = f"w{w}.npz"
        np.savez(
            cache / name,
            world_id=np.asarray(world_id),
            **{k: v[rows] for k, v in bank.arrays.items()},
        )
        receipts.append(
            {
                "world_id": world_id,
                "accepted": True,
                "npz": name,
                "npz_sha256": dinputs.sha256_file(cache / name),
            }
        )
    (cache / "manifest.json").write_text(json.dumps({"worlds": receipts, "rejected": []}))
    return cache


def test_run_report_pools_sparse_folds_and_preserves_absent_metrics(
    tmp_path: Path, monkeypatch
) -> None:
    from tests.test_t34_claim_a_loss import tiny_bank

    bank = tiny_bank(worlds=4)
    # Both retained worlds have decisive rows; the third held-out fold is empty.
    rows = bank.world < 2
    bank = H.Bank(
        bank.world_ids[:2], bank.world[rows], {k: v[rows] for k, v in bank.arrays.items()}
    )
    cache = _write_cache(tmp_path, bank)
    ckpt = tmp_path / "b8.pth"
    ckpt.write_bytes(b"residual-does-not-load-this")
    tiny = H.Cell("tiny", "split", "residual", n_dec=4, n_anchor=12, checkpoints=(0, 1))
    monkeypatch.setitem(H.DEFAULT_CELLS, "tiny", tiny)
    report = H.run_report(
        cache, ckpt, tmp_path / "report.json", cells=["tiny"], free_q=[], folds_k=3, threads=1
    )
    cell = report["cells"][0]
    empty = cell["folds"][2]
    assert empty["heldout_worlds"] == []
    assert empty["curve"][0]["heldout_anchor"] == {"n": 0}
    assert cell["pooled"][0]["heldout_anchor_agree"] == 1.0
    assert cell["pooled"][0]["heldout_anchor_max_decile_disagree"] == 0.0
    assert cell["pooled"][0]["heldout_boost_ratio_max"] == 1.0
    assert not D.guards_hold(cell, 1)

    # All-empty support stays absent, rather than manufacturing perfect agreement.
    result = H.FoldResult(0, 0, 0, 0, curve=[empty["curve"][0]])
    pooled = H.pooled(bank, [(result, {0: np.zeros((0, 6), np.float32)})], np.zeros(10))[0]
    assert pooled["heldout_decisive"] == {"n": 0}
    assert pooled["heldout_anchor_agree"] is None
    assert pooled["heldout_anchor_max_decile_disagree"] is None
    assert pooled["heldout_boost_ratio_max"] is None
    json.dumps(report, allow_nan=False)


def test_run_report_wires_cache_free_q_and_cells(tmp_path: Path, monkeypatch) -> None:
    cache = _write_cache(tmp_path)
    ckpt = tmp_path / "b8.pth"
    ckpt.write_bytes(b"not-a-checkpoint")  # residual cells never load it
    tiny = H.Cell("tiny", "split", "residual", lr=1e-2, n_dec=4, n_anchor=12, checkpoints=(0, 4))
    monkeypatch.setitem(H.DEFAULT_CELLS, "tiny", tiny)
    out = tmp_path / "out" / "d0_report.json"
    report = H.run_report(
        cache, ckpt, out, cells=["tiny"], free_q=["split"], free_q_steps=50, folds_k=2, threads=1
    )
    assert out.is_file() and not out.with_suffix(".partial.json").exists()
    assert report["bank"]["decisive_bearing_worlds"] == 4
    assert report["claim"] == "D0_DESIGN_DIAGNOSTIC_EXCLUDED_A5_NOT_EVIDENCE"
    with pytest.raises(D.DecisionRefused):
        D.decide(json.loads(out.read_text()))
    with pytest.raises(FileExistsError):
        H.run_report(cache, ckpt, out, cells=["tiny"], free_q=[], folds_k=2, threads=1)


def test_d0_runner_writes_refusal_for_a_two_world_fixture(tmp_path: Path, monkeypatch) -> None:
    from research.colab_cli.runtime import _exports
    from research.t34_claim_a_loss import run_d0 as runner
    from tests.test_t34_claim_a_loss import tiny_bank

    bank = tiny_bank(worlds=3)  # one decisive-bearing world, one anchor-only world
    rows = bank.world < 2
    bank = H.Bank(
        bank.world_ids[:2], bank.world[rows], {k: v[rows] for k, v in bank.arrays.items()}
    )
    cache = _write_cache(tmp_path, bank)
    ckpt = tmp_path / "b8.pth"
    ckpt.write_bytes(b"preflight-never-fits")
    monkeypatch.setattr(runner.inputs, "unpack", lambda *_: [Path("world0"), Path("world1")])

    def build_cache(_roots, _ckpt, out, **_kwargs):
        out.mkdir()
        for source in cache.iterdir():
            (out / source.name).write_bytes(source.read_bytes())
        return {"accepted": 2, "rejected": [], "totals": {"rows": bank.n}}

    def unexpected(*_args, **_kwargs):
        pytest.fail("unsupported folds must not run probes or fabricate cell updates")

    monkeypatch.setattr(runner, "build_cache", build_cache)
    monkeypatch.setattr(H, "run_cell", unexpected)
    monkeypatch.setattr(H, "free_q_probe", unexpected)
    out = tmp_path / "output"
    export_cache = {}
    progress_write = runner._write

    def observe_progress(path, payload):
        progress_write(path, payload)
        # Poll just as the runtime does between phases. Published receipts must
        # remain unchanged, and mutable timing must not become an export yet.
        records = _exports(tmp_path, 1024**2, export_cache)
        assert "d0_timing.json" not in {record["path"] for record in records}

    monkeypatch.setattr(runner, "_write", observe_progress)
    timing = runner.run(
        tarball=tmp_path / "tiny.tar.gz",
        tarball_sha256="a" * 64,
        checkpoint=ckpt,
        work=tmp_path / "work",
        out_dir=out,
        workers=2,
        device="cpu",
        threads=1,
        expect_worlds=2,
    )
    report = json.loads((out / "d0_report.json").read_text())
    decision = json.loads((out / "D0_DECISION.json").read_text())
    assert timing["status"] == "COMPLETE"
    assert report["status"] == "PRECONDITION_FAILED"
    assert report["unsupported_training_folds"] == [
        {"fold": 0, "train_decisive": 0, "train_anchor": 0}
    ]
    assert report["cells_requested"] == list(H.D0_CELLS)
    assert report["folds"] == H.D0_FOLDS
    assert report["cells"] == report["free_q"] == []
    assert decision["decision"] == "REFUSED"
    assert decision["report_sha256"] == dinputs.sha256_file(out / "d0_report.json")
    assert decision["registered_run"] is True
    expected = {"cache_manifest.json", "d0_report.json", "D0_DECISION.json", "d0_timing.json"}
    assert {path.name for path in out.glob("*.json")} == expected
    records = _exports(tmp_path, 1024**2, export_cache)
    assert {record["path"] for record in records} == expected
    for record in records:
        path = out / record["path"]
        assert Path(str(path) + ".complete").read_text() == "complete\n"
        assert (
            Path(str(path) + ".sha256").read_text().split()[0]
            == hashlib.sha256(path.read_bytes()).hexdigest()
        )
        assert record["size"] == path.stat().st_size
    assert _exports(tmp_path, 1024**2, export_cache) == records


@pytest.mark.parametrize("failure", ["unpack", "rejected"])
def test_d0_terminal_failure_receipts_are_exportable(tmp_path: Path, monkeypatch, failure) -> None:
    from research.colab_cli.runtime import _exports
    from research.t34_claim_a_loss import run_d0 as runner

    def unpack(*_args):
        if failure == "unpack":
            raise dinputs.InputsRefused("fixture unpack failure")
        return [Path("world0")]

    def build_cache(_roots, _ckpt, out, **_kwargs):
        out.mkdir()
        manifest = {"accepted": 0, "rejected": ["world0"], "totals": {"rows": 0}}
        (out / "manifest.json").write_text(json.dumps(manifest))
        return manifest

    monkeypatch.setattr(runner.inputs, "unpack", unpack)
    monkeypatch.setattr(runner, "build_cache", build_cache)
    kwargs = dict(
        tarball=tmp_path / "tiny.tar.gz",
        tarball_sha256="a" * 64,
        checkpoint=tmp_path / "b8.pth",
        work=tmp_path / "work",
        out_dir=tmp_path / "output",
        workers=1,
        device="cpu",
        expect_worlds=1,
    )
    if failure == "unpack":
        with pytest.raises(dinputs.InputsRefused, match="fixture unpack failure"):
            runner.run(**kwargs)
        expected = {"d0_timing.json"}
        status = "FAILED"
    else:
        assert runner.run(**kwargs)["status"] == "PRECONDITION_FAILED"
        expected = {"d0_timing.json", "cache_manifest.json"}
        status = "PRECONDITION_FAILED"
    cache = {}
    records = _exports(tmp_path, 1024**2, cache)
    assert {record["path"] for record in records} == expected
    assert json.loads((kwargs["out_dir"] / "d0_timing.json").read_text())["status"] == status
    assert _exports(tmp_path, 1024**2, cache) == records


# ---------------------------------------------------------------------------
# manifest (pure builder; never dispatched)
# ---------------------------------------------------------------------------
def test_d0_manifest_is_one_l4_job_inside_the_lane_cap(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(M, "require_qualified_transport", lambda *_a: {})
    tar = tmp_path / "d0.tar.gz"
    tar.write_bytes(b"x" * 100)
    m = M.make_d0_manifest(commit="a" * 40, output_dir=tmp_path / "o", tarball=tar)
    assert m["script"] == "research/t34_claim_a_loss/d0.sh"
    assert m["requirements"]["allowed_gpus"] == ["L4"]
    assert m["budget"]["unit_cap"] == 8.0 and m["budget"]["other_assignments"] == {}
    assert m["env"]["T34_D0_TARBALL_SHA256"] == dinputs.sha256_file(tar)
    assert {i["dest"] for i in m["inputs"]} == {"public-inputs.json", "d0-exports.tar.gz"}
    assert "research/t34_claim_a_loss" in m["source_paths"]
    with pytest.raises(M.ManifestRefused, match="lane cap"):
        M.make_d0_manifest(
            commit="a" * 40,
            output_dir=tmp_path / "o",
            tarball=tar,
            est_minutes=180,
            timeout_minutes=390,
        )
    with pytest.raises(M.ManifestRefused, match="twice"):
        M.make_d0_manifest(
            commit="a" * 40, output_dir=tmp_path / "o", tarball=tar, timeout_minutes=150
        )
