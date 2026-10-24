"""Synthetic-only F2 statistic, sealing, and publication adversaries."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from research.bl_ppo.exports import digest, export_json
from research.bl_residual import f2


def fixture_table(n: int = 1481) -> tuple[dict, dict]:
    worlds = np.asarray([f"world-{i % 269:03d}" for i in range(n)])
    legal = np.ones((n, 6), dtype=bool)
    died = np.ones_like(legal)
    died[:, 1] = False
    table = {
        "world": worlds,
        "legal": legal,
        "died": died,
        "horizon": np.where(died, 20, 300),
        "truncated_at_300": ~died,
        "boost_only_escape": np.ones(n, dtype=bool),
    }
    low = np.ones((n, 6), dtype=np.float32)
    high = low.copy()
    low[:, 0] = 0
    high[:, 1] = 0
    return table, {25: low, 100: high}


def test_pinned_statistic_and_zero_boundary() -> None:
    table, predictions = fixture_table()
    result = f2.statistic(table, predictions)
    assert result["paired_delta"] == 1
    assert result["world_cluster_bootstrap"] == {
        "resamples": 10000,
        "seed": 3403,
        "quantile": 0.1,
        "method": "linear",
        "lower90": 1.0,
    }
    assert result["volume_struck"] is False
    predictions[100] = predictions[25]
    equal = f2.statistic(table, predictions)
    assert equal["paired_delta"] == equal["world_cluster_bootstrap"]["lower90"] == 0
    assert equal["volume_struck"] is True


def test_cluster_bootstrap_matches_literal_world_resampling() -> None:
    table, predictions = fixture_table()
    predictions[100][::3] = predictions[25][::3]
    predictions[100][::11] = predictions[25][::11]
    actual = f2.statistic(table, predictions)
    worlds = table["world"]
    delta = f2.selected_survival(table, predictions[100]).astype(int)
    unique = sorted(set(worlds))
    rng = np.random.default_rng(3403)
    draws = []
    by_world = [delta[worlds == world] for world in unique]
    for _ in range(10000):
        picks = rng.integers(0, len(unique), size=len(unique))
        draws.append(np.concatenate([by_world[i] for i in picks]).mean())
    assert actual["paired_delta"] == float(delta.mean())
    assert actual["world_cluster_bootstrap"]["lower90"] == float(np.quantile(draws, 0.1))


def test_legal_argmin_lowest_tie_unknown_not_survived() -> None:
    table, predictions = fixture_table(4)
    p = predictions[100]
    p[0, 0] = p[0, 1]  # tie selects action 0, which dies
    table["legal"][1, 0] = False
    p[1, 0] = -0.0  # illegal minimum ignored
    table["status"] = np.full((4, 6), "EXACT", dtype="U20")
    table["status"][2, 1] = "UNKNOWN"
    table["status"][3, 1] = "NOT_EVALUATED"
    assert f2.selected_survival(table, p).tolist() == [False, True, False, False]
    del table["status"]
    table["horizon"][2, 1] = 299
    table["truncated_at_300"][3, 1] = False
    assert f2.selected_survival(table, p).tolist() == [False, True, False, False]


@pytest.mark.parametrize("bad", ["rows", "worlds", "nan", "illegal", "extra-dose"])
def test_adversarial_input_refusals(bad: str) -> None:
    table, predictions = fixture_table()
    if bad == "rows":
        table["boost_only_escape"][0] = False
    elif bad == "worlds":
        table["world"][:] = "same"
    elif bad == "nan":
        predictions[25][0, 0] = np.nan
    elif bad == "illegal":
        table["legal"][0] = False
    else:
        predictions[50] = predictions[25]
    with pytest.raises(ValueError):
        f2.statistic(table, predictions)


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def git_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "synthetic@example.test")
    _git(root, "config", "user.name", "Synthetic")
    monkeypatch.setattr(f2, "F2_ROOT", tmp_path / "artifacts")
    receipt = tmp_path / "artifacts/sealed-receipt.json"
    monkeypatch.setattr(f2, "RECEIPT", receipt)
    monkeypatch.setattr(f2, "_validate_plan", lambda *_: None)
    export_json(root / f2.INPUT_LEDGER, {"schema": "synthetic-input-ledger"})
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "synthetic input publication")
    _git(root, "update-ref", "refs/remotes/origin/main", "HEAD")
    return root, receipt


def seal_fixture(root: Path, receipt: Path) -> None:
    ref = f2._published(root / f2.INPUT_LEDGER, f2.INPUT_LEDGER, root)
    export_json(
        receipt,
        {
            "schema": f2.SCHEMA,
            "published": True,
            "published_ref": ref,
            "source_sha256": digest(root / f2.INPUT_LEDGER),
            "judgment_sha256": f2.JUDGMENT_SHA,
            "status": "INVALID",
            "volume_struck": True,
        },
    )


def test_publication_before_any_semantic_receipt_read(tmp_path, monkeypatch) -> None:
    root, receipt = git_fixture(tmp_path, monkeypatch)
    seal_fixture(root, receipt)
    original = Path.read_text
    sealed_bytes = receipt.read_bytes()
    original_loads = json.loads

    def no_semantic_read(payload, *args, **kwargs):
        if payload == sealed_bytes:
            pytest.fail("sealed receipt parsed before its ledger was published")
        return original_loads(payload, *args, **kwargs)

    def no_receipt_read(path, *args, **kwargs):
        if path == receipt:
            pytest.fail("sealed receipt opened before its ledger was published")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", no_receipt_read)
    monkeypatch.setattr(f2.json, "loads", no_semantic_read)
    f2.stage_publication(root=root)
    with pytest.raises(subprocess.CalledProcessError):
        f2.verify_published_receipt(receipt, root=root)
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "synthetic hash-only result publication")
    _git(root, "update-ref", "refs/remotes/origin/main", "HEAD")
    monkeypatch.setattr(Path, "read_text", original)
    monkeypatch.setattr(f2.json, "loads", original_loads)
    binding = f2.verify_published_receipt(receipt, root=root)
    assert binding["sha256"] == digest(receipt)
    assert "volume_struck" not in binding
    _git(root, "commit", "--allow-empty", "-qm", "unrelated later main commit")
    _git(root, "update-ref", "refs/remotes/origin/main", "HEAD")
    assert f2.verify_published_receipt(receipt, root=root) == binding


def test_local_publication_differs_refuses_before_read(tmp_path, monkeypatch) -> None:
    root, receipt = git_fixture(tmp_path, monkeypatch)
    seal_fixture(root, receipt)
    f2.stage_publication(root=root)
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "synthetic result publication")
    _git(root, "update-ref", "refs/remotes/origin/main", "HEAD")
    (root / f2.RESULT_LEDGER).write_text("{}")
    with pytest.raises(ValueError):
        f2.verify_published_receipt(receipt, root=root)


@pytest.mark.parametrize("failure", [ValueError("fixed"), RuntimeError("secret statistic")])
def test_sealed_runner_has_no_statistic_leak_and_never_retries(
    tmp_path, monkeypatch, capsys, failure
) -> None:
    from research.bl_residual import f2_inputs

    root, receipt = git_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(f2, "_validate_plan", lambda *_: None)
    for name in f2.THREAD_ENV:
        monkeypatch.setenv(name, "1")
    monkeypatch.setattr(f2.os, "getpriority", lambda *_: 15)
    # The input value used by the stub is metadata only.
    monkeypatch.setattr(f2_inputs, "load_inputs", lambda _: (_ for _ in ()).throw(failure))
    monkeypatch.setattr(f2.json, "loads", json.loads)
    original = Path.read_text

    def plan_read(path, *args, **kwargs):
        if path == root / f2.INPUT_LEDGER:
            return '{"inputs": {}}'
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", plan_read)
    result = f2.run_sealed(root=root)
    assert result == {"sealed_receipt": str(receipt), "sha256": digest(receipt)}
    assert capsys.readouterr().out == ""
    value = json.loads(original(receipt))
    assert value["status"] == "INVALID" and value["volume_struck"] is True
    assert "secret" not in original(receipt)
    with pytest.raises(FileExistsError):
        f2.run_sealed(root=root)


def test_success_writes_statistic_only_inside_sealed_receipt(tmp_path, monkeypatch, capsys) -> None:
    from research.bl_residual import f2_inputs

    root, receipt = git_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(f2, "_validate_plan", lambda *_: None)
    for name in f2.THREAD_ENV:
        monkeypatch.setenv(name, "1")
    monkeypatch.setattr(f2.os, "getpriority", lambda *_: 15)
    table, predictions = fixture_table()
    monkeypatch.setattr(f2_inputs, "load_inputs", lambda _: (table, None, predictions))
    original = Path.read_text
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda p, *a, **kw: (
            '{"inputs": {}}' if p == root / f2.INPUT_LEDGER else original(p, *a, **kw)
        ),
    )
    result = f2.run_sealed(root=root)
    assert capsys.readouterr().out == capsys.readouterr().err == ""
    value = json.loads(original(receipt))
    assert value["status"] == "COMPLETE" and value["paired_delta"] == 1
    assert "paired_delta" not in result
    assert set(p.name for p in receipt.parent.iterdir()) == {
        "one-look.lock.json",
        "one-look.lock.json.sha256",
        "one-look.lock.json.complete",
        "sealed-receipt.json",
        "sealed-receipt.json.sha256",
        "sealed-receipt.json.complete",
    }


@pytest.mark.parametrize("failure", ["lineage", "counts", "lock"])
def test_no_payload_or_fallback_after_refusal(tmp_path, monkeypatch, failure) -> None:
    from research.bl_residual import f2_c0, f2_inputs

    root, receipt = git_fixture(tmp_path, monkeypatch)
    for name in f2.THREAD_ENV:
        monkeypatch.setenv(name, "1")
    monkeypatch.setattr(f2.os, "getpriority", lambda *_: 15)
    original = Path.read_text
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda p, *a, **kw: (
            '{"inputs": {}}' if p == root / f2.INPUT_LEDGER else original(p, *a, **kw)
        ),
    )
    monkeypatch.setattr(f2_c0, "fit_c0", lambda *_: pytest.fail("refusal reached a fallback fit"))
    if failure == "lock":
        export_json(f2.F2_ROOT / "one-look.lock.json", {"prior": "synthetic interrupted look"})
        monkeypatch.setattr(f2_inputs, "load_inputs", lambda *_: pytest.fail("second payload look"))
        with pytest.raises(FileExistsError):
            f2.run_sealed(root=root)
        assert not receipt.exists()
        return
    if failure == "lineage":

        def refused(_):
            print("must never escape the seal")
            raise f2_inputs.LineageError("synthetic forbidden ancestor")

        monkeypatch.setattr(f2_inputs, "load_inputs", refused)
    else:
        table, _ = fixture_table(1480)
        monkeypatch.setattr(f2_inputs, "load_inputs", lambda _: (table, None, None))
    f2.run_sealed(root=root)
    value = json.loads(original(receipt))
    assert value["status"] == ("NOT_RUN" if failure == "lineage" else "INVALID")
    assert value["volume_struck"] is True
    assert "paired_delta" not in value


def test_unpublished_input_refused_before_claim_or_payload(tmp_path, monkeypatch) -> None:
    from research.bl_residual import f2_inputs

    root, receipt = git_fixture(tmp_path, monkeypatch)
    for name in f2.THREAD_ENV:
        monkeypatch.setenv(name, "1")
    monkeypatch.setattr(f2.os, "getpriority", lambda *_: 15)
    _git(root, "update-ref", "-d", "refs/remotes/origin/main")
    monkeypatch.setattr(f2_inputs, "load_inputs", lambda _: pytest.fail("unpublished data opened"))
    with pytest.raises(subprocess.CalledProcessError):
        f2.run_sealed(root=root)
    assert not receipt.exists()
    assert not receipt.parent.exists()
