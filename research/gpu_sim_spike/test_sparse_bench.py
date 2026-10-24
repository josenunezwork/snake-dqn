"""CPU-only contracts for sparse benchmark ledgers, package caps, and kill rules."""

from __future__ import annotations

import pytest
import torch

from research.gpu_sim_spike import sparse_bench, sparse_pack, sparse_units

COMMIT = "a" * 40


def test_cpu_oracle_handles_distinct_worlds_in_repeated_template_batch(monkeypatch):
    """E>1 must compare every world despite the original single-world cfg."""
    from research.gpu_sim_spike.test_vector_device import _params, _world
    from research.gpu_sim_spike.vector_device import DeviceVectorRuntime

    monkeypatch.setenv("GPU_SIM_VECTOR_FLOOD", "static")
    sim = _world().repeat_interleave(2)
    sim.bodies[1, 1, 6] = torch.tensor((9, 6))
    runtime = DeviceVectorRuntime(_params())
    runtime.bind(sim)
    expected = sparse_bench._features_cpu_oracle(sim, runtime)
    actual = runtime.features(sim, update_memory=False).numpy()
    assert actual.shape == expected.shape == (4, 61)
    assert actual.tobytes() == expected.tobytes()


def _ledger(path, *, canary=False):
    sparse_units.publish_json(
        path,
        sparse_units.empty_ledger(
            sparse_units.namespace(COMMIT, canary=canary), sparse_units.allocation(canary=canary)
        ),
    )


def test_sparse_manifest_binds_q2_envelope_without_profiler_gate(tmp_path, monkeypatch):
    ledger, canary = tmp_path / "verified-ledger.json", tmp_path / "canary" / "verified-ledger.json"
    _ledger(ledger)
    _ledger(canary, canary=True)
    monkeypatch.setattr(sparse_pack, "checkpoint_inputs", lambda: [])
    monkeypatch.setattr(sparse_pack, "convert_public_checkpoints", lambda value, repo: value)
    monkeypatch.setattr(sparse_pack, "assert_upload_budget", lambda value, repo: None)
    preflight = {"path": "x", "sha256": "b" * 64}
    main = sparse_pack.manifest(
        commit=COMMIT,
        output_dir=tmp_path / "main",
        ledger=ledger,
        canary_ledger=canary,
        canary=False,
        preflight=preflight,
    )
    probe = sparse_pack.manifest(
        commit=COMMIT,
        output_dir=tmp_path / "probe",
        ledger=ledger,
        canary_ledger=canary,
        canary=True,
        preflight=preflight,
    )
    assert main["timeout_minutes"] == 192 and probe["timeout_minutes"] == 24
    assert main["budget"]["unit_cap"] + probe["budget"]["unit_cap"] == 36
    assert main["sparse_bench"]["a1_prof_v4_exports_optional"] is True
    assert main["canary"]["required"] == "same-pin sparse canary receipt"
    assert main["sparse_bench"]["kill_rule"].startswith("STOP_ROUTE")
    assert (
        main["resume"]["skip_verified"] is True and main["resume"]["max_outcome_blind_retries"] == 2
    )


def test_sparse_units_refuse_foreign_payload_and_keep_atomic_contract(tmp_path):
    ns, allocation = sparse_units.namespace(COMMIT), sparse_units.allocation()
    sparse_units.publish_unit(tmp_path, ns, allocation, allocation[0], {"sealed": True})
    ledger = tmp_path / "verified-ledger.json"
    sparse_units.build_ledger(tmp_path, ledger, commit=COMMIT)
    assert list(sparse_units.load_ledger(ledger, ns, allocation)) == [allocation[0]]
    payload = tmp_path / sparse_units.UNIT_DIRECTORY / f"{allocation[0]}.json"
    payload.write_text("foreign")
    try:
        sparse_units.load_ledger(ledger, ns, allocation)
    except ValueError as exc:
        assert "checksum" in str(exc)
    else:
        raise AssertionError("tampered complete payload was accepted")


def test_sparse_ledger_skips_only_partial_and_refuses_foreign_allocation(tmp_path):
    ns, allocation = sparse_units.namespace(COMMIT), sparse_units.allocation()
    partial = tmp_path / sparse_units.UNIT_DIRECTORY / f"{allocation[0]}.json"
    partial.parent.mkdir()
    partial.write_text("partial")
    ledger = tmp_path / "verified-ledger.json"
    built = sparse_units.build_ledger(tmp_path, ledger, commit=COMMIT)
    assert built["units"] == []
    sparse_units.publish_unit(tmp_path, ns, allocation, allocation[1], {"good": True})
    foreign = tmp_path / sparse_units.UNIT_DIRECTORY / "cell-scripted-e1.json"
    foreign.write_text("foreign")
    with pytest.raises(ValueError, match="foreign"):
        sparse_units.build_ledger(tmp_path, tmp_path / "ledger-2.json", commit=COMMIT)


def test_sparse_ledger_copies_verified_payload_for_portable_resume(tmp_path):
    ns, allocation = sparse_units.namespace(COMMIT), sparse_units.allocation()
    exports = tmp_path / "exports"
    sparse_units.publish_unit(exports, ns, allocation, allocation[0], {"sealed": True})
    portable = tmp_path / "portable" / "verified-ledger.json"
    sparse_units.build_ledger(exports, portable, commit=COMMIT)
    assert list(sparse_units.load_ledger(portable, ns, allocation)) == [allocation[0]]


def test_recovered_v4_exports_bind_prior_pin_without_success_or_traces(tmp_path):
    import json

    from research.gpu_sim_spike import a1_prof_units

    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "commit": COMMIT,
                "env": {"GPU_SIM_SPIKE_ENTRYPOINT": "a1-prof", "A1_PROF_CANARY": "1"},
                "a1_prof": {"version": 4},
            }
        )
    )
    exports = tmp_path / "exports"
    allocation = a1_prof_units.allocation(canary=True)
    payload = a1_prof_units.publish_unit(
        exports,
        a1_prof_units.namespace(COMMIT, canary=True),
        allocation,
        allocation[0],
        {"chrome_traces": [{"path": "missing-large-trace"}]},
    )
    binding = sparse_pack._a1_v4_exports_binding(exports)
    assert binding["commit"] == COMMIT and len(binding["units"]) == 1
    assert "not canary" in binding["role"]
    payload.write_text("tampered")
    with pytest.raises(ValueError, match="checksum"):
        sparse_pack._a1_v4_exports_binding(exports)


def test_recovered_v4_exports_refuse_foreign_identity(tmp_path):
    import json

    from research.gpu_sim_spike import a1_prof_units

    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "commit": COMMIT,
                "env": {"GPU_SIM_SPIKE_ENTRYPOINT": "a1-prof", "A1_PROF_CANARY": "1"},
                "a1_prof": {"version": 4},
            }
        )
    )
    exports = tmp_path / "exports"
    allocation = a1_prof_units.allocation(canary=True)
    a1_prof_units.publish_unit(
        exports,
        a1_prof_units.namespace("b" * 40, canary=True),
        allocation,
        allocation[0],
        {},
    )
    with pytest.raises(sparse_pack.ManifestRefused, match="foreign"):
        sparse_pack._a1_v4_exports_binding(exports)


def test_cell_stops_for_any_full_feature_or_speed_failure(monkeypatch):
    values = iter(
        (
            {
                "backend": "dense",
                "feature_oracle": {"match": True},
                "median_tick_seconds": 4.0,
                "samples": [{"tick_seconds": 4.0, "complete_world_steps_per_second": 256.0}],
            },
            {
                "backend": "static",
                "feature_oracle": {"match": True},
                "median_tick_seconds": 3.0,
                "samples": [{"tick_seconds": 3.0, "complete_world_steps_per_second": 341.0}],
            },
        )
    )
    monkeypatch.setattr(sparse_bench, "_measure_backend", lambda *a, **k: next(values))
    assert (
        sparse_bench._cell(None, "scripted", 1024, "cuda", 1, 3, 1, enforce_speed=True)["verdict"]
        == "STOP_ROUTE_SPEEDUP_LT_2X"
    )
    values = iter(
        (
            {
                "backend": "dense",
                "feature_oracle": {"match": False},
                "median_tick_seconds": 4.0,
                "samples": [],
            },
            {"backend": "static", "feature_oracle": {"match": False}, "samples": []},
        )
    )
    monkeypatch.setattr(sparse_bench, "_measure_backend", lambda *a, **k: next(values))
    assert (
        sparse_bench._cell(None, "scripted", 1024, "cuda", 1, 3, 1, enforce_speed=True)["verdict"]
        == "INVALID_FEATURE_MISMATCH"
    )


def test_smoke_does_not_apply_speed_rule_but_keeps_static_feature_rule(monkeypatch):
    values = iter(
        (
            {
                "backend": "dense",
                "feature_oracle": {"match": False},
                "median_tick_seconds": 4.0,
                "samples": [{"tick_seconds": 4.0, "complete_world_steps_per_second": 0.25}],
            },
            {
                "backend": "static",
                "feature_oracle": {"match": True},
                "median_tick_seconds": 3.0,
                "samples": [{"tick_seconds": 3.0, "complete_world_steps_per_second": 0.33}],
            },
        )
    )
    monkeypatch.setattr(sparse_bench, "_measure_backend", lambda *a, **k: next(values))
    row = sparse_bench._cell(None, "scripted", 1, "cpu", 1, 0, 1, enforce_speed=False)
    assert row["verdict"] == "PASS_CELL" and row["baseline_dense_cpu_diagnostic_only"] is True


def test_later_feature_failure_preserves_immutable_recorded_state():
    fixture = {"path": "recorded-states/scripted-static-e1024.npz", "sha256": "b" * 64}
    rows = [
        {"feature_oracle": {"match": True, "recorded_fixture": fixture}, "samples": []},
        {"feature_oracle": {"match": False}, "samples": []},
    ]
    merged = sparse_bench._merge_backend("static", rows)
    assert merged["feature_oracle"] == {"match": False, "recorded_fixture": fixture}


@pytest.mark.parametrize("bind_exports", [False, True])
def test_sparse_pack_canary_and_main_do_not_require_profiler_pass(
    tmp_path, monkeypatch, bind_exports
):
    import sys

    monkeypatch.setattr(sparse_pack, "require_source_pin", lambda commit: None)
    monkeypatch.setattr(sparse_pack, "_preflight_binding", lambda path, commit: {})
    monkeypatch.setattr(sparse_pack, "manifest", lambda **kwargs: {"canary": kwargs["canary"]})
    monkeypatch.setattr(sparse_pack, "_a1_v4_exports_binding", lambda path: {"version": 4})
    base = ["pack", "--commit", COMMIT, "--artifact-dir", str(tmp_path)]
    if bind_exports:
        base += ["--a1-prof-v4-exports", str(tmp_path / "recovered-exports")]
    monkeypatch.setattr(sys, "argv", base)
    sparse_pack.main()
    assert (tmp_path / "sparse-bench-canary.json").is_file()
    template = (tmp_path / "sparse-bench-main-template.json").read_bytes()
    monkeypatch.setattr(sys, "argv", base + ["--canary-receipt", str(tmp_path / "sparse.json")])
    sparse_pack.main()
    assert (tmp_path / "sparse-bench-main.json").is_file()
    assert (tmp_path / "sparse-bench-main-template.json").read_bytes() == template
    for name in ("dispatch-canary.sh", "dispatch-main-after-canary.sh"):
        script = (tmp_path / name).read_text()
        assert "A1_PROF_V4_RECEIPT" not in script
        assert ("--a1-prof-v4-exports" in script) == bind_exports


def test_generated_helpers_execute_without_profiler_receipt(tmp_path):
    import subprocess

    repo = tmp_path / "repo"
    (repo / "venv/bin").mkdir(parents=True)
    (repo / "tools/ops").mkdir(parents=True)
    # A tiny fake Python/dispatch boundary records exact arguments, while Bash
    # exercises the generated helpers' actual environment/receipt checks.
    python = repo / "venv/bin/python"
    python.write_text('#!/bin/sh\nprintf "%s\\n" "$@" >> "$ARG_LOG"\n')
    dispatch = repo / "tools/ops/dispatch.sh"
    dispatch.write_text('#!/bin/sh\nprintf "%s\\n" "$@" >> "$ARG_LOG"\n')
    python.chmod(0o755)
    dispatch.chmod(0o755)
    import os

    env = dict(os.environ, ARG_LOG=str(tmp_path / "args"))
    env.pop("A1_PROF_V4_RECEIPT", None)
    for generator in (sparse_pack._canary_script, sparse_pack._main_script):
        script = tmp_path / (generator.__name__ + ".sh")
        script.write_text(
            generator(tmp_path, COMMIT, tmp_path / "preflight.json", 10).replace(
                f"REPO={sparse_pack.ROOT}", f"REPO={repo}"
            )
        )
        if generator == sparse_pack._main_script:
            refused = subprocess.run(["bash", str(script)], env=env, capture_output=True)
            assert refused.returncode != 0 and b"missing sparse canary" in refused.stderr
            (tmp_path / "canary-output").mkdir()
            (tmp_path / "canary-output/receipt.json").write_text("{}")
        result = subprocess.run(["bash", str(script)], env=env, capture_output=True)
        assert result.returncode == 0, result.stderr
    assert "--canary-receipt" in (tmp_path / "args").read_text()
