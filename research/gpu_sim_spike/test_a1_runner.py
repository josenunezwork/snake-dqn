"""Fault-injection proof of cell-level A1 recovery, without CUDA or heavy jobs."""

from __future__ import annotations

import argparse

import pytest

from research.gpu_sim_spike import a1_analyze, a1_units, a100_ruler, make_a1_pack

COMMIT = "a" * 40
HARDWARE = "A100-40GB"


def _cell(**kw):
    value = {
        "mix": kw["mix"],
        "requested_worlds": kw["worlds"],
        "actual_concurrent_worlds": kw["worlds"],
        "bodies": [800, 900, 1000, 1100, 1300, 1500],
        "repaired_event_path": kw["repaired"],
        "warmed_repeats": kw["repeats"],
        "ticks_per_repeat": kw["ticks"],
        "median_tick_seconds": 1.0 if kw["repaired"] else 10.0,
        "median_complete_steps_per_second": 21_000.0,
        "setup_seconds": 0.1,
    }
    if kw["instrumented"]:
        value["instrumented"] = {"host_stages": dict.fromkeys(a100_ruler.STAGES, 0.1)}
    return value


def test_oom_preserves_first_cell_and_replacement_skips_it(tmp_path, monkeypatch):
    ns, alloc = a1_units.namespace(COMMIT, HARDWARE, False), a1_units.allocation(False)
    first = tmp_path / "first"
    ledger = tmp_path / "initial" / "verified-ledger.json"
    a1_units.publish_json(ledger, a1_units.empty_ledger(ns, alloc))
    args = argparse.Namespace(
        device="cpu",
        hardware=HARDWARE,
        smoke=False,
        warmup_ticks=3,
        seconds=120,
        ledger=ledger,
        checkpoint=tmp_path / "model.pth",
        output=first / "a1-ruler.json",
    )
    monkeypatch.setattr(a100_ruler, "source_commit", lambda: COMMIT)
    monkeypatch.setattr(a100_ruler, "source_hashes", lambda: {})
    calls = []

    def fail(**kw):
        calls.append((kw["mix"], kw["worlds"]))
        if kw["worlds"] == 256:
            raise RuntimeError("CUDA out of memory (injected, no hardware)")
        return _cell(**kw)

    monkeypatch.setattr(a100_ruler, "_measure_cell", fail)
    with pytest.raises(RuntimeError, match="out of memory"):
        a100_ruler.run(args)
    assert a1_units.payload_path(first, "cell-scripted-e6").with_suffix(".json.complete").is_file()
    with pytest.raises(ValueError, match="INCOMPLETE"):
        a1_analyze.analyze(first, commit=COMMIT, hardware=HARDWARE)
    recovery = tmp_path / "recovery" / "verified-ledger.json"
    a1_units.build_ledger(first, recovery, commit=COMMIT, hardware=HARDWARE, smoke=False)
    args.ledger, args.output = recovery, tmp_path / "second" / "a1-ruler.json"
    calls.clear()

    def succeed(**kw):
        calls.append((kw["mix"], kw["worlds"], kw["repaired"]))
        return _cell(**kw)

    monkeypatch.setattr(a100_ruler, "_measure_cell", succeed)
    monkeypatch.setattr(
        a100_ruler,
        "_parity_check",
        lambda *a: {"parity": {"classification": "float tie"}, "tape": {"independent": True}},
    )
    receipt = a100_ruler.run(args)
    assert receipt["verified_units_reused"] == ["cell-scripted-e6"]
    assert ("long-scripted", 6, True) not in calls
    assert receipt["kill_tier"]["tier"] == "SEALED_PENDING_MAC_AGGREGATION"
    decision = a1_analyze.analyze(args.output.parent, commit=COMMIT, hardware=HARDWARE)
    assert decision["kill_tier"]["tier"] == "FUND_WP1C_WP2"
    assert len(decision["verified_units"]) == 11


@pytest.mark.parametrize("hardware", ["A100-40GB", "A100-80GB", "H100-80GB"])
def test_variant_pack_has_separate_memory_guard_namespace_and_resumable_inputs(
    tmp_path, monkeypatch, hardware
):
    ledger = tmp_path / "verified-ledger.json"
    a1_units.publish_json(ledger, make_a1_pack.empty_ledger(COMMIT, hardware))
    monkeypatch.setattr(make_a1_pack, "checkpoint_inputs", lambda: [])
    monkeypatch.setattr(make_a1_pack, "convert_public_checkpoints", lambda value, repo: value)
    monkeypatch.setattr(make_a1_pack, "assert_upload_budget", lambda value, repo: None)
    smoke_ledger = tmp_path / "canary" / "verified-ledger.json"
    a1_units.publish_json(smoke_ledger, make_a1_pack.empty_ledger(COMMIT, hardware, smoke=True))
    manifest = make_a1_pack.manifest(
        commit=COMMIT,
        output_dir=tmp_path / "output",
        ledger=ledger,
        smoke_ledger=smoke_ledger,
        canary=False,
        hardware=hardware,
    )
    assert manifest["env"]["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert manifest["env"]["A1_RULER_EXPECTED_GPU_FAMILY"] == hardware
    assert manifest["requirements"]["allowed_gpus"] == [hardware.split("-")[0]]
    assert len(manifest["inputs"]) == 6
    assert manifest["resume"]["skip_verified"] is True


def test_canary_covers_identical_inputs_including_nonempty_resume_bundle(tmp_path, monkeypatch):
    from research.colab_cli.canary import input_inventory

    ns, alloc = a1_units.namespace(COMMIT, HARDWARE, False), a1_units.allocation(False)
    exports = tmp_path / "previous"
    a1_units.publish_unit(exports, ns, alloc, alloc[0], {"sealed": "do not inspect"})
    ledger = tmp_path / "recovery" / "verified-ledger.json"
    a1_units.build_ledger(exports, ledger, commit=COMMIT, hardware=HARDWARE, smoke=False)
    smoke = tmp_path / "smoke" / "verified-ledger.json"
    a1_units.publish_json(smoke, make_a1_pack.empty_ledger(COMMIT, smoke=True))
    monkeypatch.setattr(make_a1_pack, "checkpoint_inputs", lambda: [])
    monkeypatch.setattr(make_a1_pack, "convert_public_checkpoints", lambda value, repo: value)
    monkeypatch.setattr(make_a1_pack, "assert_upload_budget", lambda value, repo: None)
    shared = dict(
        commit=COMMIT,
        ledger=ledger,
        smoke_ledger=smoke,
        attempt=2,
        remaining_minutes=90,
        remaining_units=15,
    )
    main = make_a1_pack.manifest(**shared, output_dir=tmp_path / "main", canary=False)
    canary = make_a1_pack.manifest(**shared, output_dir=tmp_path / "canary", canary=True)
    assert input_inventory(main) == input_inventory(canary)
    assert len(main["inputs"]) == 9
    assert main["timeout_minutes"] == 90
    assert main["budget"]["unit_cap"] == 15
    assert main["env"]["A1_RULER_SECONDS"] == "4950"
    assert canary["env"]["A1_RULER_LEDGER_PATH"] == "canary-inputs/verified-ledger.json"
    assert main["env"]["A1_RULER_LEDGER_PATH"] == "inputs/verified-ledger.json"
    with pytest.raises(make_a1_pack.ManifestRefused, match="explicit remaining"):
        make_a1_pack.manifest(
            commit=COMMIT,
            ledger=ledger,
            smoke_ledger=smoke,
            attempt=2,
            output_dir=tmp_path / "unsafe",
            canary=False,
        )


def test_completion_refuses_wall_budget_that_cannot_fit_unit_cap(tmp_path, monkeypatch):
    ledger = tmp_path / "verified-ledger.json"
    smoke = tmp_path / "canary" / "verified-ledger.json"
    a1_units.publish_json(ledger, make_a1_pack.empty_ledger(COMMIT))
    a1_units.publish_json(smoke, make_a1_pack.empty_ledger(COMMIT, smoke=True))
    with pytest.raises(make_a1_pack.ManifestRefused, match="priced above"):
        make_a1_pack.manifest(
            commit=COMMIT,
            ledger=ledger,
            smoke_ledger=smoke,
            attempt=2,
            remaining_minutes=85,
            remaining_units=14,
            output_dir=tmp_path / "invalid",
            canary=False,
        )
