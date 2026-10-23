"""Exercise actual job skip logic without historical data or simulator load."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from research.a100_a2 import job
from research.a100_a2.protocol import NAMESPACE, allocation, allocation_hash, empty_ledger
from research.bl_ppo.exports import digest


def fixture_inputs(root, monkeypatch):
    root.mkdir()
    (root / "b8.pth").write_bytes(b"fake checkpoint for mocked fit")
    monkeypatch.setattr(job, "B8_SHA", digest(root / "b8.pth"))
    bindings = {
        k: digest(Path(job.__file__).with_name(n))
        for k, n in [
            ("preregistration", "PREREGISTRATION.md"),
            ("semantics_erratum", "SEMANTICS_ERRATUM.md"),
        ]
    }
    alloc = allocation_hash(bindings)
    ca = hashlib.sha256((alloc + "-canary").encode()).hexdigest()
    (root / "canary-ledger.json").write_text(json.dumps(empty_ledger(NAMESPACE + "-canary", ca)))
    control = dict(
        schema="a2-control/v1",
        bindings=bindings,
        allocation=alloc,
        input_hashes={"b8.pth": digest(root / "b8.pth")},
    )
    (root / "control.json").write_text(json.dumps(control))
    return control, ca


def test_entrypoint_skips_exactly_verified_units(tmp_path, monkeypatch):
    from research.a100_a2 import run

    inputs = tmp_path / "inputs"
    _, ca = fixture_inputs(inputs, monkeypatch)
    units = [
        u
        for u in allocation()
        if u["table"] == "t5" and u["dose"] == 100 and u["seed"] == 20261022 and u["fold"] == 0
    ]
    ledger = empty_ledger(NAMESPACE + "-canary", ca)
    i = units[0]["index"]
    ledger["units"] = [
        dict(
            index=i,
            unit_id=f"{NAMESPACE}-canary/{ca}/{i}",
            name=f"fit-{i:04d}.npz",
            size=1,
            sha256="a" * 64,
        )
    ]
    (inputs / "canary-ledger.json").write_text(json.dumps(ledger))
    calls = []

    def fit(*args, **kwargs):
        calls.append((kwargs["encoder"], kwargs["target"]))
        return {"p_d300": np.zeros((2, 6))}

    monkeypatch.setattr(run, "fit_unit", fit)
    receipt = job.run(inputs, tmp_path / "out", "cpu", "smoke")
    assert receipt["skipped"] == [i]
    assert len(calls) == 7
    assert (units[0]["encoder"], units[0]["target"]) not in calls
    assert len(receipt["completed"]) == 7
    ledger["namespace"] = "foreign"
    (inputs / "canary-ledger.json").write_text(json.dumps(ledger))
    calls.clear()
    with pytest.raises(ValueError, match="foreign"):
        job.run(inputs, tmp_path / "foreign", "cpu", "smoke")
    assert not calls


def test_main_materialization_uses_archive_portably(tmp_path, monkeypatch):
    from research.a100_a2 import data, run

    inputs = tmp_path / "inputs"
    control, _ = fixture_inputs(inputs, monkeypatch)
    (inputs / "verified-ledger.json").write_text(
        json.dumps(empty_ledger(NAMESPACE, control["allocation"]))
    )
    monkeypatch.setattr(job, "hardware", lambda *a, **kw: {"name": "mock-A100-40"})
    units = allocation()[:1]
    monkeypatch.setattr(job, "allocation", lambda: units)
    calls = []

    def materialize(bundle, output, checkpoint):
        calls.append((bundle.name, checkpoint.name))
        output.mkdir()
        (output / "materialization.json").write_text(json.dumps({"synthetic_fixture": True}))
        for name in ("t5", "a5"):
            np.savez(output / f"{name}.npz", **job.synthetic_table())
        return {n: output / f"{n}.npz" for n in ("t5", "a5")}

    monkeypatch.setattr(data, "materialize_bundle", materialize)
    monkeypatch.setattr(run, "fit_unit", lambda *a, **kw: {"p_d300": np.zeros((2, 6))})
    result = job.run(inputs, tmp_path / "output", "cuda", "main")
    assert calls == [("labels.tar.gz", "b8.pth")]
    assert len(result["completed"]) == 1
    assert result["materialization_sha256"] == digest(tmp_path / "output/materialization.json")
