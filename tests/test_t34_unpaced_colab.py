"""Focused, non-gameplay contracts for T34's descriptive Colab arm."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from research.t34_pathb import make_unpaced_manifest as manifests
from research.t34_pathb import retention_v2, unpaced_colab

REPO = Path(__file__).resolve().parents[1]
COMMIT = "a" * 40


def _rows() -> tuple[list[dict], list[dict], dict]:
    terminal, seed = [], 10_000
    for mix in retention_v2.MIXES:
        for _ in range(48):
            terminal.append(
                {
                    "seed": seed,
                    "mix": mix,
                    "kind": "terminal",
                    "roster": {"seed": seed, "mix": mix, "specs": []},
                }
            )
            seed += 1
    product = [{"seed": seed + index, "mix": "mixed", "kind": "product"} for index in range(32)]
    return terminal, product, {"seed": seed + 32, "mix": "mixed", "kind": "timing"}


def _registration() -> dict:
    terminal, product, timing = _rows()
    return {
        "schema": "t34-pathb-v2-registration/v1",
        "ratified": True,
        "ratified_by": "Claude",
        "near_boundary_precedence_acknowledged": True,
        "source_commit": COMMIT,
        "source_hashes": {"source.py": "c" * 64},
        "mac_arms": list(retention_v2.MAC_ARMS),
        "unpaced_colab": retention_v2.unpaced_colab_policy(),
        "horizon": 5000,
        "terminal_rows": terminal,
        "product_rows": product,
        "timing_row": timing,
        "bank": {"path": "bank.json", "sha256": "b" * 64},
        "seed_audit": {"path": "seed.json", "sha256": ""},
        "profile": {"path": "profile.json", "sha256": ""},
        "preregistration": {"path": "prereg.md", "sha256": ""},
    }


def _inputs(tmp_path: Path) -> tuple[Path, Path, dict]:
    value = _registration()
    registration = tmp_path / "registration.json"
    registration.write_text(json.dumps(value))
    return registration, REPO / "research/t34_exit_b8/PUBLIC_INPUTS.json", value


def test_manifest_cap_reserve_n_rate_commit_and_transport_refusals(tmp_path, monkeypatch):
    registration, catalog, value = _inputs(tmp_path)
    monkeypatch.setattr(manifests, "require_qualified_transport", lambda *_args: {})
    manifest = manifests.make_manifest(
        registration=registration,
        catalog=catalog,
        output_dir=tmp_path / "new",
        shard=0,
        shard_count=3,
        nproc=12,
        commit=COMMIT,
    )
    assert manifest["requirements"]["gpu"] == "L4"
    assert manifest["requirements"]["min_vcpu"] == 12
    assert manifest["budget"]["unit_cap"] * 3 == pytest.approx(unpaced_colab.GLOBAL_UNIT_CAP)
    assert manifest["budget"]["reserve_units"] * 3 == pytest.approx(
        unpaced_colab.GLOBAL_RESERVE_UNITS
    )
    assert manifest["timeout_minutes"] >= 2 * manifest["requirements"]["est_minutes"]
    assert manifest["unpaced_budget"]["worlds"] == 59
    value["source_commit"] = "b" * 40
    registration.write_text(json.dumps(value))
    with pytest.raises(unpaced_colab.UnpacedRefused, match="equal"):
        manifests.make_manifest(
            registration=registration,
            catalog=catalog,
            output_dir=tmp_path / "commit",
            shard=0,
            shard_count=3,
            nproc=12,
            commit=COMMIT,
        )
    value["source_commit"] = COMMIT
    registration.write_text(json.dumps(value))
    with pytest.raises(unpaced_colab.UnpacedRefused, match="positive"):
        manifests.make_manifest(
            registration=registration,
            catalog=catalog,
            output_dir=tmp_path / "rate",
            shard=0,
            shard_count=3,
            nproc=12,
            commit=COMMIT,
            cpu_rate_units_per_hour=0,
        )
    monkeypatch.setattr(
        manifests,
        "require_qualified_transport",
        lambda *_args: (_ for _ in ()).throw(manifests.QualificationRefused("not qualified")),
    )
    with pytest.raises(unpaced_colab.UnpacedRefused, match="not qualified"):
        manifests.make_manifest(
            registration=registration,
            catalog=catalog,
            output_dir=tmp_path / "qual",
            shard=0,
            shard_count=3,
            nproc=12,
            commit=COMMIT,
        )


def test_manifest_refuses_wrong_n_and_timeout_reserve_liability(tmp_path, monkeypatch):
    registration, catalog, value = _inputs(tmp_path)
    monkeypatch.setattr(manifests, "require_qualified_transport", lambda *_args: {})
    value["product_rows"].pop()
    registration.write_text(json.dumps(value))
    with pytest.raises(unpaced_colab.UnpacedRefused, match="176"):
        manifests.make_manifest(
            registration=registration,
            catalog=catalog,
            output_dir=tmp_path / "n",
            shard=0,
            shard_count=1,
            nproc=12,
            commit=COMMIT,
        )
    registration, catalog, _value = _inputs(tmp_path)
    with pytest.raises(unpaced_colab.UnpacedRefused, match="liability"):
        manifests.make_manifest(
            registration=registration,
            catalog=catalog,
            output_dir=tmp_path / "cap",
            shard=0,
            shard_count=3,
            nproc=2,
            commit=COMMIT,
            cpu_rate_units_per_hour=99,
        )


def test_runtime_registration_hash_bank_and_non_git_fallback(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    value = _registration()
    bank = {key: value[key] for key in ("terminal_rows", "product_rows", "timing_row")}
    (root / "bank.json").write_text(json.dumps(bank))
    value["bank"]["sha256"] = hashlib.sha256((root / "bank.json").read_bytes()).hexdigest()
    for filename, key, contents in (
        ("seed.json", "seed_audit", "{}"),
        ("profile.json", "profile", "{}"),
        ("prereg.md", "preregistration", "prereg"),
    ):
        path = root / filename
        path.write_text(contents)
        value[key]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(unpaced_colab, "ROOT", root)
    monkeypatch.setattr(retention_v2, "source_inventory", lambda _root: value["source_hashes"])
    monkeypatch.setenv("COLAB_CLI_ROOT", str(tmp_path))
    unpaced_colab._runtime_registration(value, COMMIT)
    bad = json.loads(json.dumps(value))
    bad["source_hashes"] = {"source.py": "d" * 64}
    with pytest.raises(unpaced_colab.UnpacedRefused, match="source inventory"):
        unpaced_colab._runtime_registration(bad, COMMIT)
    bad = json.loads(json.dumps(value))
    bad["terminal_rows"][0]["seed"] += 1
    with pytest.raises(unpaced_colab.UnpacedRefused, match="terminal_rows differs"):
        unpaced_colab._runtime_registration(bad, COMMIT)
    monkeypatch.delenv("COLAB_CLI_ROOT")
    with pytest.raises(unpaced_colab.UnpacedRefused, match="non-Git"):
        unpaced_colab._runtime_registration(value, COMMIT)


def test_preflight_requires_two_disjoint_distinct_terminal_product_rows(tmp_path, monkeypatch):
    registration, catalog, value = _inputs(tmp_path)
    monkeypatch.setattr(unpaced_colab, "_public_paths", lambda *_args: {})
    from research.t34_pathb import cost_smoke_v2

    monkeypatch.setattr(cost_smoke_v2, "_preflight", lambda: {})
    common = {
        "registration_path": registration,
        "catalog": catalog,
        "input_root": tmp_path / "input",
        "out": tmp_path / "out",
        "shard": 0,
        "shard_count": 1,
        "nproc": 1,
        "preflight": True,
        "source_commit": COMMIT,
        "frames": 4,
    }
    with pytest.raises(unpaced_colab.UnpacedRefused, match="exactly two"):
        unpaced_colab.run_shard(**common, development_rows=[])
    terminal = {"kind": "terminal", "seed": 900001, "mix": "scripted", "roster": {"specs": []}}
    duplicate = {"kind": "product", "seed": 900001, "mix": "mixed"}
    with pytest.raises(unpaced_colab.UnpacedRefused, match="one terminal"):
        unpaced_colab.run_shard(**common, development_rows=[terminal, duplicate])
    overlap = {"kind": "product", "seed": value["timing_row"]["seed"], "mix": "mixed"}
    with pytest.raises(unpaced_colab.UnpacedRefused, match="disjoint"):
        unpaced_colab.run_shard(**common, development_rows=[terminal, overlap])
    with pytest.raises(unpaced_colab.UnpacedRefused, match="four frames"):
        unpaced_colab.run_shard(
            **{**common, "frames": 5}, development_rows=[terminal, {**duplicate, "seed": 900002}]
        )


def test_streamed_modulo_payloads_are_sorted_and_preserve_original_rows(tmp_path, monkeypatch):
    registration, catalog, value = _inputs(tmp_path)
    monkeypatch.setattr(unpaced_colab, "_runtime_registration", lambda *_args: None)
    monkeypatch.setattr(unpaced_colab, "_public_paths", lambda *_args: {})
    monkeypatch.setattr(
        unpaced_colab, "_hardware", lambda nproc, workers: {"nproc": nproc, "workers": workers}
    )
    monkeypatch.setattr(
        unpaced_colab,
        "_collect_star",
        lambda args: {
            "bare_prefix_trace": {
                "initial_sha256": "0" * 64,
                "steps": [],
                "ended_by_intervention": False,
            }
        },
    )
    result = unpaced_colab.run_shard(
        registration_path=registration,
        catalog=catalog,
        input_root=tmp_path / "input",
        out=tmp_path / "out",
        shard=1,
        shard_count=2,
        nproc=2,
        source_commit=COMMIT,
    )
    shard = json.loads(result.read_text())
    rows = value["terminal_rows"] + value["product_rows"]
    assert len(shard["worlds"]) == 88
    assert shard["worlds"][0]["path"] == "worlds/terminal-001.json"
    assert shard["worlds"][-1]["path"] == "worlds/product-175.json"
    payload = json.loads((tmp_path / "out" / shard["worlds"][0]["path"]).read_text())
    assert payload["world"] == rows[1]
    world = tmp_path / "out" / shard["worlds"][0]["path"]
    assert world.with_name(world.name + ".complete").read_text() == "complete\n"
    assert (
        world.with_name(world.name + ".sha256").read_text().split()[0]
        == shard["worlds"][0]["sha256"]
    )


def test_failure_leaves_earlier_streamed_payload_but_no_complete_shard(tmp_path, monkeypatch):
    registration, catalog, _value = _inputs(tmp_path)
    monkeypatch.setattr(unpaced_colab, "_runtime_registration", lambda *_args: None)
    monkeypatch.setattr(unpaced_colab, "_public_paths", lambda *_args: {})
    calls = {"count": 0}

    def collector(_args):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("fake worker failure")
        return {
            "bare_prefix_trace": {
                "initial_sha256": "1" * 64,
                "steps": [],
                "ended_by_intervention": False,
            }
        }

    monkeypatch.setattr(unpaced_colab, "_collect_star", collector)
    with pytest.raises(RuntimeError, match="fake worker"):
        unpaced_colab.run_shard(
            registration_path=registration,
            catalog=catalog,
            input_root=tmp_path / "input",
            out=tmp_path / "out",
            shard=0,
            shard_count=88,
            nproc=2,
            source_commit=COMMIT,
        )
    assert (tmp_path / "out" / "worlds" / "terminal-000.json").is_file()
    assert not (tmp_path / "out" / "shard.json").exists()


def test_complete_sidecars_are_create_only_and_match_payload(tmp_path):
    output = tmp_path / "world.json"
    digest = unpaced_colab._write_complete(output, {"schema": "test", "value": 1})
    assert output.with_name("world.json.complete").read_text() == "complete\n"
    assert output.with_name("world.json.sha256").read_text().split()[0] == digest
    with pytest.raises(FileExistsError):
        unpaced_colab._write_complete(output, {"schema": "test", "value": 1})


def _deterministic_worker_fixture(args: tuple) -> dict:
    """Spawn-safe, single-thread fixture: no simulator, model or controller."""
    row, kind, _registration, _paths, horizon = args
    digest = hashlib.sha256(f"{kind}:{row['seed']}:{horizon}".encode()).hexdigest()
    return {
        "bare_prefix_trace": {
            "initial_sha256": digest,
            "steps": [digest],
            "ended_by_intervention": False,
        }
    }


def test_serial_equals_spawn_pool_world_bytes_and_order(tmp_path, monkeypatch):
    """Exercise the real spawn/export/merge boundary with two CPU fixture workers."""
    registration, catalog, _value = _inputs(tmp_path)
    monkeypatch.setattr(unpaced_colab, "_runtime_registration", lambda *_args: None)
    monkeypatch.setattr(unpaced_colab, "_public_paths", lambda *_args: {})
    monkeypatch.setattr(unpaced_colab, "_collect_star", _deterministic_worker_fixture)
    indices = []
    for name, nproc in (("serial", 2), ("parallel", 3)):
        index = unpaced_colab.run_shard(
            registration_path=registration,
            catalog=catalog,
            input_root=tmp_path / "input",
            out=tmp_path / name,
            shard=0,
            shard_count=88,
            nproc=nproc,
            source_commit=COMMIT,
        )
        indices.append(json.loads(index.read_text())["worlds"])
    assert indices[0] == indices[1]
    for world in indices[0]:
        relative = world["path"]
        assert (tmp_path / "serial" / relative).read_bytes() == (
            tmp_path / "parallel" / relative
        ).read_bytes()
