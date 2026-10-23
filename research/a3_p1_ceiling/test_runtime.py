"""Offline resume proofs for independent A3 assay units; no simulation."""

import json

import pytest

from research.a3_p1_ceiling.runtime import (
    SCHEMA,
    UNITS,
    aggregate,
    allocation,
    filename,
    make_ledger,
    read_ledger,
    run,
)
from research.bl_ppo.exports import export_json


def control():
    return {"namespace": "a3-test", "commit": "a" * 40, "assay_seed": 20271022}


def test_real_ledger_cli_accepts_payloads_after_options(monkeypatch, tmp_path):
    from research.a3_p1_ceiling.runtime import main

    c = control()
    cp, path, out = tmp_path / "control.json", tmp_path / filename(c, 0), tmp_path / "ledger.json"
    cp.write_text(json.dumps(c))
    export_json(path, {"sealed": True})
    monkeypatch.setattr(
        "sys.argv", ["runtime", "ledger", "--control", str(cp), "--output", str(out), str(path)]
    )
    main()
    assert set(read_ledger(out, c)) == {0}


def test_ledger_skips_every_verified_unit_without_launch(monkeypatch, tmp_path):
    c = control()
    paths = []
    for i in range(len(UNITS)):
        path = tmp_path / "prior" / filename(c, i)
        export_json(path, {"sealed": True})
        paths.append(path)
    ledger = tmp_path / "verified.json"
    export_json(ledger, make_ledger(c, paths))

    def forbidden(*args, **kwargs):
        pytest.fail("verified units were recomputed")

    monkeypatch.setattr("research.a3_p1_ceiling.runtime.subprocess.run", forbidden)
    run(c, ledger, tmp_path / "unused.pth", tmp_path / "output", tiny=False, device="cuda")
    receipt = json.loads((tmp_path / "output/receipt.json").read_text())
    assert receipt["verified_skipped"] == list(range(len(UNITS)))


def test_partial_foreign_duplicate_and_changed_units_refuse(tmp_path):
    c = control()
    path = tmp_path / filename(c, 0)
    path.write_text("incomplete")
    with pytest.raises(ValueError):
        make_ledger(c, [path])
    path.unlink()
    export_json(path, {"sealed": True})
    with pytest.raises(ValueError, match="duplicate"):
        make_ledger(c, [path, path])
    ledger = tmp_path / "ledger.json"
    export_json(ledger, make_ledger(c, [path]))
    with pytest.raises(ValueError, match="foreign"):
        read_ledger(ledger, {**c, "namespace": "other"})
    with pytest.raises(ValueError, match="INCOMPLETE"):
        aggregate(c, ledger)


def test_aggregator_requires_whole_unique_non_tiny_allocation(tmp_path):
    c = control()
    paths = []
    for i, (mode, variant) in enumerate(UNITS):
        path = tmp_path / filename(c, i)
        export_json(
            path,
            {
                "schema": SCHEMA,
                "namespace": c["namespace"],
                "allocation": allocation(c),
                "index": i,
                "mode": mode,
                "variant": variant,
                "tiny": i == 0,
                "result": {},
            },
        )
        paths.append(path)
    ledger = tmp_path / "ledger.json"
    export_json(ledger, make_ledger(c, paths))
    with pytest.raises(ValueError, match="tiny"):
        aggregate(c, ledger)


@pytest.mark.parametrize(
    "incomplete,grad_pass,seconds",
    [(True, True, 1.0), (False, False, 1.6384), (False, True, 1.63840001)],
)
def test_nested_incomplete_and_frozen_kill_precedence(tmp_path, incomplete, grad_pass, seconds):
    c = control()
    paths = []
    for i, (mode, variant) in enumerate(UNITS):
        path = tmp_path / filename(c, i)
        export_json(
            path,
            {
                "schema": SCHEMA,
                "namespace": c["namespace"],
                "allocation": allocation(c),
                "index": i,
                "mode": mode,
                "variant": variant,
                "tiny": False,
                "result": {
                    "schema": "a3-p1-learner-ceiling/v1",
                    "shape": {"worlds": 128, "rollout_steps": 128, "transitions": 16384},
                    "mode": mode,
                    "variant": variant,
                    "actor_enabled": True,
                    "arm": "F",
                    "device": "cuda",
                    "status": "INCOMPLETE(KL skip)" if incomplete else "COMPLETE",
                    "phase_timers_seconds": [{"adam_steps": 2}] * 3,
                    "cached_learning": {
                        "median_seconds": seconds if variant == "baseline" else 0.5
                    },
                    "comparison": {"passed": grad_pass},
                },
            },
        )
        paths.append(path)
    ledger = tmp_path / "ledger.json"
    export_json(ledger, make_ledger(c, paths))
    if incomplete:
        with pytest.raises(ValueError, match="INCOMPLETE"):
            aggregate(c, ledger)
    else:
        result = aggregate(c, ledger)
        assert result["wp5"] == (
            "PAUSE_LEARNER_FIX" if seconds > 1.6384 else "CEILING_NOT_FALSIFIED"
        )
        assert result["optimization"] == (
            "NUMERICAL_COMPARISON_PASS" if grad_pass else "REJECT_UNEXPLAINED_GRADIENT_CHANGE"
        )
