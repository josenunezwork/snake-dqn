"""Tiny-fixture tests for the Claim A loss diagnostics and the R1 labels-only lane."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from research.t34_claim_a_loss import cache as dcache
from research.t34_claim_a_loss import harness as H
from research.t34_claim_a_loss import losses as L
from research.t34_claim_a_r1 import make_manifest as r1m
from research.t34_claim_a_r1 import plan as r1p
from research.t34_claim_a_r1 import seal as r1s
from research.t34_exit_b8_fit.loss import exit_fit_loss

torch.set_num_threads(1)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def tiny_bank(worlds: int = 6, rows: int = 40, seed: int = 0) -> H.Bank:
    """Synthetic bank: B8-like Q with the bare action ahead by ~0.17 on decisive rows."""
    rng = np.random.default_rng(seed)
    n = worlds * rows
    q = rng.normal(8.4, 0.05, size=(n, 6)).astype(np.float32)
    kind = np.zeros(n, dtype=np.int8)
    a_star = q.argmax(1)
    bare = q.argmax(1)
    forbidden = np.zeros((n, 6), dtype=bool)
    tested = np.zeros((n, 6), dtype=bool)
    survive = np.zeros((n, 6), dtype=bool)
    shadow = np.full(n, -1, dtype=np.int64)
    world = np.repeat(np.arange(worlds), rows)
    for w in range(worlds - 2):  # last two worlds carry no decisive rows
        for j in range(3):
            i = w * rows + 5 + j
            kind[i] = dcache.KIND_DECISIVE
            bare[i] = 1
            q[i] = 8.4
            q[i, 1] = 8.57
            a_star[i] = 4
            forbidden[i] = True
            forbidden[i, 4] = False
            tested[i] = True
            survive[i] = False
            survive[i, 4] = True
            shadow[i] = 10 * w
        i = w * rows + 20
        kind[i] = dcache.KIND_DELAY
        shadow[i] = 10 * w + 1
    arrays = {
        "decision": np.tile(np.arange(rows), worlds).astype(np.int64),
        "length": rng.integers(5, 1500, size=n).astype(np.int64),
        "local": rng.integers(0, 255, (n, 7, 31, 31), dtype=np.uint8),
        "global": rng.integers(0, 255, (n, 4, 37, 37), dtype=np.uint8),
        "scalars": rng.normal(size=(n, 60)).astype(np.float32),
        "q_prev": q,
        "h": rng.normal(size=(n, 256)).astype(np.float32),
        "kind": kind,
        "a_star": a_star.astype(np.int64),
        "forbidden": forbidden,
        "tested": tested,
        "survive": survive,
        "student_action": bare.astype(np.int64),
        "shadow_id": shadow,
    }
    return H.Bank([f"fixture/w{w:02d}" for w in range(worlds)], world, arrays)


# ---------------------------------------------------------------------------
# losses
# ---------------------------------------------------------------------------
def test_frozen_candidate_is_the_ratified_loss() -> None:
    bank = tiny_bank()
    idx = np.arange(bank.n)
    batch = H.batch_tensors(bank, idx, np.ones(bank.n))
    q = batch["q_prev"] + 0.01 * torch.randn(bank.n, 6)
    assert torch.equal(L.frozen(q, batch), exit_fit_loss(q, batch)[0])


def test_split_has_zero_anchor_gradient_at_identity_and_ignores_zero_weight() -> None:
    bank = tiny_bank()
    anchors = np.flatnonzero(bank["kind"] == dcache.KIND_ANCHOR)[:64]
    batch = H.batch_tensors(bank, anchors, np.ones(len(anchors)))
    q = batch["q_prev"].clone().requires_grad_(True)
    L.split(q, batch).backward()
    assert float(q.grad.abs().max()) == 0.0
    delay = np.flatnonzero(bank["kind"] == dcache.KIND_DELAY)
    zero = H.batch_tensors(bank, delay, np.zeros(len(delay)))
    qd = zero["q_prev"].clone().requires_grad_(True)
    assert float(L.split(qd, zero).detach()) == 0.0


def test_free_q_probe_frozen_keeps_bare_split_flips() -> None:
    bank = tiny_bank()
    frozen = H.free_q_probe(bank, "frozen", steps=600, lr=1e-2)
    split = H.free_q_probe(bank, "split", steps=600, lr=1e-2)
    tau2 = H.free_q_probe(bank, "frozen_tau2", steps=600, lr=1e-2)
    assert frozen["flip"] == 0.0 and frozen["argmax_bare"] == 1.0
    assert split["flip"] == 1.0 and split["margin_median"] > 0
    assert tau2["flip"] == 1.0


# ---------------------------------------------------------------------------
# folds, samplers, metrics
# ---------------------------------------------------------------------------
def test_folds_are_world_level_and_stratified() -> None:
    bearing = [True, True, True, True, False, False]
    folds = H.assign_folds([f"w{i}" for i in range(6)], bearing, 2)
    assert sorted(np.bincount(folds[:4]).tolist()) == [2, 2]
    assert sorted(np.bincount(folds[4:], minlength=2).tolist()) == [1, 1]
    assert H.assign_folds([f"w{i}" for i in range(6)], bearing, 2).tolist() == folds.tolist()


def test_stream_sampler_draws_train_worlds_only_and_excludes_delay() -> None:
    bank = tiny_bank()
    sampler = H.StreamSampler(bank, np.array([0, 1, 4]), 4, 12, np.random.default_rng(0), 3.0)
    seen = set()
    for _ in range(10):
        idx, weight = sampler.next()
        assert len(idx) == 16 and set(bank.world[idx]) <= {0, 1, 4}
        assert (bank["kind"][idx] != dcache.KIND_DELAY).all()
        assert weight[:4].tolist() == [3.0] * 4 and weight[4:].tolist() == [1.0] * 12
        seen.update(idx[:4].tolist())
    assert seen == set(np.flatnonzero(np.isin(bank.world, [0, 1]) & (bank["kind"] == 1)))


def test_ratified_epoch_is_the_frozen_one_to_one_rule() -> None:
    bank = tiny_bank()
    batches = H.ratified_epoch(bank, np.array([0, 1, 2, 3]), np.random.default_rng(0), 1024)
    assert len(batches) == 1
    idx, weight = batches[0]
    searched = (bank["kind"][idx] != dcache.KIND_ANCHOR).sum()
    assert searched == len(idx) - searched == 16
    for w in range(4):  # each death window sums to 10, delay rows 0
        dec = (bank.world[idx] == w) & (bank["kind"][idx] == dcache.KIND_DECISIVE)
        assert weight[dec].sum() == pytest.approx(10.0)
    assert (weight[bank["kind"][idx] == dcache.KIND_DELAY] == 0).all()


def test_decisive_and_anchor_metrics() -> None:
    bank = tiny_bank()
    dec = np.flatnonzero(bank["kind"] == dcache.KIND_DECISIVE)
    q = bank["q_prev"][dec].copy()
    m = H.decisive_metrics(q, bank, dec)
    assert m["flip"] == 0.0 and m["argmax_bare"] == 1.0 and m["margin_pos"] == 0.0
    q[:, 4] = 9.0
    m = H.decisive_metrics(q, bank, dec)
    assert m["flip"] == 1.0 and m["argmax_known_survivor"] == 1.0 and m["gap_vs_bare_pos"] == 1.0
    anc = np.flatnonzero(bank["kind"] == dcache.KIND_ANCHOR)
    edges = np.quantile(bank["length"][anc], np.linspace(0, 1, 11)[:-1])
    a = H.anchor_metrics(bank["q_prev"][anc], bank, anc, edges)
    assert a["agree"] == 1.0 and a["max_decile_disagree"] == 0.0


def test_residual_cell_runs_world_level_folds_and_starts_at_b8() -> None:
    bank = tiny_bank()
    bearing = [bool(((bank.world == w) & (bank["kind"] == 1)).any()) for w in range(6)]
    folds = H.assign_folds(bank.world_ids, bearing, 2)
    cell = H.Cell("t", "split", "residual", lr=1e-2, n_dec=4, n_anchor=12, checkpoints=(0, 4, 8))
    out = H.run_cell(bank, folds, cell, None, 2)
    first = out["pooled"][0]
    assert first["updates"] == 0 and first["heldout_decisive"]["flip"] == 0.0
    assert first["heldout_anchor_agree"] == 1.0
    assert [p["updates"] for p in out["pooled"]] == [0, 4, 8]
    for fold in out["folds"]:
        assert set(fold["heldout_worlds"]) == set(np.flatnonzero(folds == fold["fold"]).tolist())
    json.dumps(out)  # JSON-serialisable receipt


# ---------------------------------------------------------------------------
# cache row semantics (ratified normalize_row)
# ---------------------------------------------------------------------------
def _label(kind: str, outcome: str | None = None) -> dict:
    per_action = {
        str(a): {"tested": True, "survives": a in (2, 4), "alive_steps": 300 if a in (2, 4) else 9}
        for a in range(6)
    }
    if outcome == "all_die_delay":
        per_action = {str(a): {"tested": True, "survives": False} for a in range(6)}
    base = {"decision": 7, "label_kind": kind, "student_action": 1, "shadow_id": 3}
    if kind == "shadow_confirm":
        return base
    return base | {
        "t_star": 7,
        "a_star": 4,
        "F": [1],
        "per_action": per_action,
        "outcome_kind": outcome,
    }


def test_row_fields_follow_ratified_semantics() -> None:
    q = np.array([8.4, 8.57, 8.4, 8.4, 8.4, 8.4], dtype=np.float32)
    anchor = dcache.row_fields(_label("shadow_confirm"), q)
    assert anchor["kind"] == dcache.KIND_ANCHOR and anchor["a_star"] == 1
    rescue = dcache.row_fields(_label("transition", "horizon_rescue"), q)
    assert rescue["kind"] == dcache.KIND_DECISIVE and rescue["a_star"] == 4
    assert rescue["forbidden"].tolist() == [True, True, False, True, False, True]
    delay = dcache.row_fields(_label("lad", "all_die_delay"), q)
    assert delay["kind"] == dcache.KIND_DELAY and not delay["forbidden"].any()
    assert dcache.keep_label({"label_kind": "shadow_confirm", "decision": 16}, 8)
    assert not dcache.keep_label({"label_kind": "shadow_confirm", "decision": 17}, 8)
    assert dcache.keep_label({"label_kind": "lad", "decision": 17}, 8)


# ---------------------------------------------------------------------------
# R1 labels-only lane
# ---------------------------------------------------------------------------
def test_r1_plan_is_800_plus_200_disjoint_and_audited() -> None:
    from research.t34_claim_a.seed_audit import namespace_seeds, verify_receipt

    chunks = r1p.plan()
    assert sum(c.worlds for c in chunks if c.split == "train") == 800
    assert sum(c.worlds for c in chunks if c.split == "heldout") == 200
    receipt = json.loads(r1m.DEFAULT_SEED_AUDIT.read_text())
    seeds: set[int] = set()
    for attempt in r1p.AUDITED_ATTEMPTS:
        for chunk in r1p.plan(attempt):
            verify_receipt(receipt, chunk.namespace, chunk.worlds)
            new = set(namespace_seeds(chunk.namespace, chunk.worlds))
            assert not new & seeds
            seeds |= new
    assert set(receipt["registered_banks"]) == {"round0", "round0d", "retention_v2"}
    with pytest.raises(ValueError):
        r1p.Chunk("train", 8)
    with pytest.raises(ValueError):
        r1p.Chunk("heldout", 0, attempt=2)


def test_r1_lane_admission_and_heldout_support() -> None:
    assert r1p.admissible(12.0, [3.1, 3.2], [12.0, 12.0])
    assert not r1p.admissible(12.0, [20.0], [12.0, 12.0, 12.0])
    ok = r1p.heldout_support({f"w{i}": 5 for i in range(60)})
    assert ok["status"] == "SUPPORTED"
    assert r1p.heldout_support({f"w{i}": 5 for i in range(59)})["status"] == ("PRECONDITION_FAILED")
    assert r1p.heldout_support({f"w{i}": 4 for i in range(70)})["status"] == ("PRECONDITION_FAILED")


def _governing(tmp_path: Path, **override) -> Path:
    root = tmp_path / "t34-round0-study-01"
    out = root / "amendment-boost-floor"
    out.mkdir(parents=True)
    decision = {
        "schema": r1m.AMEND_SCHEMA,
        "governing": True,
        "decision": "GO",
        "evidence_authority": "TIER_1_DEVELOPMENT",
    } | override
    (out / "decision.json").write_text(json.dumps(decision))
    terminal = {
        "schema": r1m.AMEND_TERMINAL_SCHEMA,
        "status": "COMPLETE",
        "files": {"decision.json": r1m.sha256_file(out / "decision.json")},
    }
    (out / "phase_terminal.json").write_text(json.dumps(terminal))
    return out / "decision.json"


def _ledger_for(path: Path):
    sha = r1m.sha256_file(path.parent / "phase_terminal.json")
    return lambda _repo, rel: (
        [{"phase": "amendment-boost-floor", "terminal_sha256": sha}]
        if rel.endswith("t34-round0-study-01.jsonl")
        else []
    )


def test_r1_binds_only_the_published_governing_round0_go(tmp_path: Path) -> None:
    path = _governing(tmp_path)
    bound = r1m.governing_round0(path, ledger=_ledger_for(path))
    assert bound["decision_sha256"] == r1m.sha256_file(path)
    assert bound["amendment_ledger"].endswith("ledger/t34-round0-study-01.jsonl")
    with pytest.raises(r1m.ManifestRefused, match="published amendment ledger"):
        r1m.governing_round0(path, ledger=lambda *_: [])
    for bad, match in (
        ({"decision": "STOP"}, "not GO"),
        ({"governing": False}, "governing"),
        ({"evidence_authority": "INVALID_OR_INCOMPLETE"}, "TIER_1"),
    ):
        sub = tmp_path / bad.__repr__().replace(" ", "")[:20].replace("'", "").replace(":", "")
        sub.mkdir()
        p = _governing(sub, **bad)
        with pytest.raises(r1m.ManifestRefused, match=match):
            r1m.governing_round0(p, ledger=_ledger_for(p))
    registered = tmp_path / "reg" / "analysis"
    registered.mkdir(parents=True)
    (registered / "decision.json").write_text(json.dumps({"decision": "GO"}))
    with pytest.raises(r1m.ManifestRefused, match="never governs"):
        r1m.governing_round0(registered / "decision.json", ledger=lambda *_: [])
    assert bound["decision"] == "GO"
    wrong_phase = lambda _r, _rel: [  # noqa: E731
        {
            "phase": "analysis",
            "terminal_sha256": r1m.sha256_file(path.parent / "phase_terminal.json"),
        }
    ]
    with pytest.raises(r1m.ManifestRefused, match="published amendment ledger"):
        r1m.governing_round0(path, ledger=wrong_phase)
    sub = tmp_path / "noschema"
    sub.mkdir()
    p = _governing(sub, schema="t34-round0-decision/v1")
    with pytest.raises(r1m.ManifestRefused, match="governing verdict"):
        r1m.governing_round0(p, ledger=_ledger_for(p))
    terminal = json.loads((path.parent / "phase_terminal.json").read_text())
    (path.parent / "phase_terminal.json").write_text(json.dumps(terminal | {"status": "KILL"}))
    with pytest.raises(r1m.ManifestRefused, match="COMPLETE"):
        r1m.governing_round0(path, ledger=_ledger_for(path))
    (path.parent / "phase_terminal.json").write_text(json.dumps(terminal | {"files": {}}))
    with pytest.raises(r1m.ManifestRefused, match="does not seal"):
        r1m.governing_round0(path, ledger=_ledger_for(path))


GO_BINDING = {
    "decision": "GO",
    "decision_sha256": "d" * 64,
    "phase_terminal_sha256": "e" * 64,
    "amendment_ledger": "research/t34_round0_amend/ledger/t34-round0-study-01.jsonl",
}


@pytest.fixture
def qualified(monkeypatch):
    from research.t34_exit_b8 import make_manifest as ratified

    monkeypatch.setattr(ratified, "require_qualified_transport", lambda *_args: {})


def test_r1_chunk_manifest_is_the_ratified_label_job_retargeted(tmp_path: Path, qualified) -> None:
    from research.colab_cli import accounting

    chunk = r1p.Chunk("heldout", 1)
    round0 = GO_BINDING
    for bad in ({}, dict(GO_BINDING, decision="STOP"), dict(GO_BINDING, decision_sha256="")):
        with pytest.raises(r1m.ManifestRefused, match="governing Round 0 GO"):
            r1m.make_chunk_manifest(
                commit="a" * 40, output_dir=tmp_path / "o", chunk=chunk, round0=bad
            )
    manifest = r1m.make_chunk_manifest(
        commit="a" * 40, output_dir=tmp_path / "o", chunk=chunk, round0=round0
    )
    assert manifest["script"] == "research/t34_exit_b8/label.sh"
    assert manifest["env"]["T34_NAMESPACE"] == chunk.namespace
    assert manifest["env"]["T34_BETA"] == "1" and manifest["env"]["T34_KERNEL"] == "v2"
    assert manifest["env"]["T34_WORLDS"] == "100" and manifest["env"]["T34_ROUND"] == "r1"
    assert manifest["job_id"] == "t34-claim-a-r1-heldout-s1-a0"
    assert manifest["claim"] == r1m.CLAIM and manifest["round0_governing"] == round0
    assert manifest["requirements"]["allowed_gpus"] == ["L4"]
    assert manifest["budget"]["other_assignments"] == {}
    assert accounting._manifest_rates(manifest["budget"]["other_assignments"]) == {}
    assert manifest["seed_audit"]["sha256"] == r1m.sha256_file(r1m.DEFAULT_SEED_AUDIT)


def _world(path: Path, world_id: str, *, decisive: int, b8: str, beta: float = 1.0) -> Path:
    import hashlib

    labels = [{"outcome_kind": "horizon_rescue", "decisive": True}] * decisive
    labels += [{"label_kind": "shadow_confirm"}] * 3
    body = {
        "schema": "t34-exit-world/v1",
        "world_id": world_id,
        "teacher_v2": {"kernel": "shadow-lad-v2"},
        "checkpoint_sha256": b8,
        "rollout_policy_sha256": b8,
        "beta": beta,
        "mix": "scripted",
        "labels": labels,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(body).encode()
    path.write_bytes(encoded)
    path.with_name(path.name + ".sha256").write_text(
        f"{hashlib.sha256(encoded).hexdigest()}  {path.name}\n"
    )
    path.with_name(path.name + ".complete").write_text("complete\n")
    return path


def test_seal_inventories_label_data_and_applies_support_minimum(tmp_path: Path) -> None:
    b8 = "b" * 64
    chunks = {}
    for chunk in r1p.plan():
        root = tmp_path / chunk.job_id
        n = 2 if chunk.split == "train" else 31
        for i in range(n):
            _world(
                root / f"world-{i:06d}.json",
                f"{chunk.namespace}/scripted/{i:06d}",
                decisive=5 if chunk.split == "heldout" else 1,
                b8=b8,
            )
        chunks[chunk] = root
    with pytest.raises(r1s.SealRefused, match="attest"):
        r1s.seal(
            chunks, tmp_path / "s0.json", b8_sha256=b8, round0={}, attest_no_retention_phase=False
        )
    with pytest.raises(r1s.SealRefused, match="governing Round 0 GO"):
        r1s.seal(
            chunks, tmp_path / "s1.json", b8_sha256=b8, round0={}, attest_no_retention_phase=True
        )
    sealed = r1s.seal(
        chunks,
        tmp_path / "sealed.json",
        b8_sha256=b8,
        round0=GO_BINDING,
        attest_no_retention_phase=True,
    )
    assert sealed["worlds"] == {"train": 16, "heldout": 62}
    assert sealed["heldout_support"]["status"] == "SUPPORTED" and sealed["status"] == "SUPPORTED"
    assert sealed["claim"] == "R1_LABEL_DATA_ONLY_NOT_EVIDENCE"
    with pytest.raises(FileExistsError):
        r1s.seal(
            chunks,
            tmp_path / "sealed.json",
            b8_sha256=b8,
            round0=GO_BINDING,
            attest_no_retention_phase=True,
        )
    stray = tmp_path / "stray"
    _world(stray / "world-000000.json", "t34-claim-a/other/x", decisive=0, b8=b8)
    with pytest.raises(r1s.SealRefused, match="outside the chunk namespace"):
        r1s.inventory_chunk(r1p.Chunk("train", 0), stray, b8_sha256=b8)
    wrong = tmp_path / "wrong"
    ns = r1p.Chunk("train", 0).namespace
    _world(wrong / "world-000000.json", f"{ns}/scripted/000000", decisive=0, b8=b8, beta=0.5)
    with pytest.raises(r1s.SealRefused, match="beta"):
        r1s.inventory_chunk(r1p.Chunk("train", 0), wrong, b8_sha256=b8)
