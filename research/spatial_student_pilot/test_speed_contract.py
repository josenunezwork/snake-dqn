"""Mechanical speed controls fail closed without changing scientific checks."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from research.spatial_student_pilot.fit import finite_tensors
from research.spatial_student_pilot.native_cache import _validate, preload
from research.spatial_student_pilot.search import assess_roots, prepare_assessment
from research.spatial_student_pilot.test_search_readout import ZeroPolicy, fixture_root


def test_fused_predicate_retains_nonfinite_tripwires() -> None:
    assert finite_tensors([torch.ones(3), None, torch.zeros(2)])
    for value in (float("nan"), float("inf"), -float("inf")):
        assert not finite_tensors([torch.ones(3), None, torch.tensor([value])])


def test_memoization_exact_frozen_endpoint_and_closed_loop_separation() -> None:
    from research.spatial_student_pilot.test_batching import exact

    root, cfg, b8 = fixture_root()
    roots = [root]
    # Zero actor agrees with B8; all three frozen-action arms are duplicates,
    # but each scoring suffix and both closed-loop paths must still exist.
    prepared = prepare_assessment(roots, cfg=cfg, b8=b8, networks={}, device="cpu")
    assert len(prepared["scores"]) == 2
    assert {key[2] for key, _ in prepared["scores"]} == {4, 5}
    assert all(key[3] is False for key, _ in prepared["scores"])
    # Freeze dataclass module registration required by the old search module.
    import subprocess
    import sys
    import types

    old = types.ModuleType("spatial_search_oracle")
    sys.modules[old.__name__] = old
    raw = subprocess.check_output(
        [
            "git",
            "show",
            "e86fe37b573f6983fa44886e847ff092ca659f45:research/spatial_student_pilot/search.py",
        ]
    )
    exec(compile(raw, "<frozen-search>", "exec"), old.__dict__)
    kwargs = dict(cfg=cfg, b8=b8, networks={}, device="cpu")
    expected = old.assess_roots(roots, ZeroPolicy(), **kwargs)
    actual = assess_roots(roots, ZeroPolicy(), prepared=prepared, **kwargs)
    exact(actual, expected)
    from research.spatial_student_pilot.search import paired_gameplay

    worlds = [dict(world_id=0, roster_index=0, mix="scripted")]
    old_game = old.paired_gameplay(worlds, ZeroPolicy(), **kwargs)
    new_game = paired_gameplay(worlds, ZeroPolicy(), **kwargs)
    for arm in ("b8", "student"):
        for suffix in ("_mass", "_alive", "_action_counts", "_initial_state_sha256"):
            exact(new_game[0][arm + suffix], old_game[0][arm + suffix])
    bad = deepcopy(prepared)
    bad["scores"][0] = ((*bad["scores"][0][0][:2], 0, False), bad["scores"][0][1])
    with pytest.raises(ValueError, match="scoring keys"):
        assess_roots(roots, ZeroPolicy(), prepared=bad, **kwargs)


def test_missing_native_cache_preserves_lazy_compile(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("SPATIAL_NATIVE_CACHE_SHA256", raising=False)
    assert preload(tmp_path) is False
    with pytest.raises(ValueError, match="inventory differs"):
        _validate({"schema": "foreign"})


def test_ruler_probe_records_complete_checkpoint_cost() -> None:
    from dataclasses import replace

    from research.spatial_student_pilot.collect import collect_batch
    from research.spatial_student_pilot.config import PilotConfig
    from research.spatial_student_pilot.test_batching import FixedPolicy, networks

    cfg = replace(
        PilotConfig().development("preflight"),
        mode="ruler",
        namespace=PilotConfig().development("ruler").namespace,
        tiny=False,
        horizon=50,
        collection_checkpoint_every=500,
    )
    # Preserve small geometry while verifying ruler-only final probe boundary.
    from unittest.mock import Mock

    proxy = Mock(wraps=cfg)
    for name in (
        "namespace",
        "allocation_index",
        "worlds",
        "collection_unit_worlds",
        "demos",
        "horizon",
        "mode",
        "tiny",
        "body_capacity",
        "collection_checkpoint_every",
    ):
        setattr(proxy, name, getattr(cfg, name))
    proxy.sim_config = PilotConfig().development("preflight").sim_config
    seen = []
    collect_batch(
        proxy,
        range(6),
        b8=FixedPolicy(),
        networks=networks(),
        device="cpu",
        stop=lambda: False,
        progress=lambda _: None,
        ruler_ticks=2,
        checkpoint=seen.append,
    )
    assert len(seen) == 1 and seen[0]["next_tick"] == 2
    assert seen[0]["capture_seconds"] > 0


def test_cached_requirements_resolve_uploaded_wheels_without_network(tmp_path: Path) -> None:
    """Exercise pip's actual env expansion/find-links through the qualified bootstrap."""
    import os
    import subprocess
    import sys
    import zipfile

    inputs = tmp_path / "inputs"
    wheelhouse = inputs / "wheelhouse"
    wheelhouse.mkdir(parents=True)
    wheel = wheelhouse / "spatialcachefixture-1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as stream:
        stream.writestr(
            "spatialcachefixture-1.0.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: spatialcachefixture\nVersion: 1.0\n",
        )
        stream.writestr(
            "spatialcachefixture-1.0.dist-info/WHEEL",
            "Wheel-Version: 1.0\nGenerator: fixture\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        stream.writestr("spatialcachefixture-1.0.dist-info/RECORD", "")
    cached = Path("research/spatial_student_pilot/requirements-cached.txt").read_text()
    assert "-r requirements.txt" in cached
    requirements = tmp_path / "requirements-cached.txt"
    requirements.write_text(cached.replace("-r requirements.txt", "spatialcachefixture==1.0"))
    env = dict(os.environ, COLAB_INPUT_DIR=str(inputs))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "download",
            "--isolated",
            "--no-index",
            "--no-deps",
            "--dest",
            str(tmp_path / "download"),
            "-r",
            str(requirements),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "download" / wheel.name).read_bytes() == wheel.read_bytes()
    # The active qualified transport has not gained a cache hook or changed pins.
    baseline = subprocess.check_output(["git", "show", "e86fe37b:research/colab_cli/bootstrap.sh"])
    assert Path("research/colab_cli/bootstrap.sh").read_bytes() == baseline


def test_native_cache_binds_sources_toolchain_and_stopped_export_hashes(tmp_path: Path) -> None:
    """Synthetic binary bytes prove transport refusals, never native/CUDA execution."""
    import hashlib
    import json
    import subprocess

    from research.gpu_sim_spike.native_compile import ROOT, inventory
    from research.spatial_student_pilot.native_cache import NAME, SCHEMA, upload_inputs

    def sha(raw: bytes) -> str:
        return hashlib.sha256(raw).hexdigest()

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    hashes, specs = inventory(ROOT, commit)
    export = tmp_path / "exports/spatial"
    (export / "native").mkdir(parents=True)
    rows = []
    for spec in specs:
        if spec["name"] == "snake_dqn_v8_indexed_v1":
            continue
        name = spec["name"]
        path = export / "native" / (name + ".so")
        path.write_bytes(b"synthetic-nonexecutable-fixture-" + name.encode())
        rows.append(
            dict(
                name=name,
                path="native/" + path.name,
                size=path.stat().st_size,
                sha256=sha(path.read_bytes()),
            )
        )
    report = dict(
        schema=SCHEMA,
        source_sha256=hashes,
        extensions=specs,
        binaries=rows,
        platform="linux-x86_64",
        torch="2.9.1+cu126",
        cuda="12.6",
        architecture="8.0",
        source_commit=commit,
    )
    (export / NAME).write_text(json.dumps(report))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(dict(commit=commit)))
    inventory_rows = [
        dict(
            path="spatial/" + str(p.relative_to(export)),
            size=p.stat().st_size,
            sha256=sha(p.read_bytes()),
        )
        for p in export.rglob("*")
        if p.is_file()
    ]
    receipt_path = tmp_path / "receipt.json"
    receipt = dict(
        commit=commit,
        stop_verified=True,
        cleanup_error=None,
        manifest_sha256=sha(manifest.read_bytes()),
        exports=inventory_rows,
    )
    receipt_path.write_text(json.dumps(receipt))
    assert len(upload_inputs(receipt_path, commit)) == 5
    stale = deepcopy(report)
    stale["source_sha256"][next(iter(hashes))] = "0" * 64
    with pytest.raises(ValueError, match="inventory differs"):
        _validate(stale, commit=commit)
    wrong = dict(report, cuda="12.8")
    with pytest.raises(ValueError, match="inventory differs"):
        _validate(wrong, commit=commit)
    (export / rows[0]["path"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="export binding"):
        upload_inputs(receipt_path, commit)
    receipt["stop_verified"] = False
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="stopped supervisor"):
        upload_inputs(receipt_path, commit)
