"""Synthetic trace receipt tests; these do not execute parity worlds."""

import copy
import hashlib

import pytest

from research.q0q1_20261017 import q1_bank, q1_evaluate, q1_parity


def _h(value):
    return hashlib.sha256(q1_parity.canonical(value).encode()).hexdigest()


def _receipt():
    bank = q1_bank.load_world_bank()
    source = q1_evaluate.source_manifest()
    worlds = []
    for w in bank["parity_worlds"]:
        frames = ["f" * 64] * 5000
        worlds.append(
            {
                "world_id": str(w["world_seed"]),
                "world_seed": w["world_seed"],
                "mix": w["mix"],
                "roster_digest": w["roster_semantic_sha256"],
                "world_digest": _h(frames),
                "frame_digests": frames,
                "record": {"mass_integral": 100},
                "actions": [
                    {
                        "frame": 0,
                        "slot": 0,
                        "policy": "ego2s",
                        "action": 0,
                        "mask": [True] * 6,
                        "q": [1, 0, 0, 0, 0, 0],
                    }
                ],
            }
        )
    environment = {"system": "Darwin", "cpu_threads": 1, "gpu_used": False}
    artifact = {
        "schema": q1_parity.SCHEMA,
        "bank_namespace": q1_bank.NAMESPACE,
        "horizon": 5000,
        "bank_sha256": _h(bank),
        "b8_sha256": q1_evaluate.B8_SHA256,
        "profile_digest": bank["profile_semantic_sha256"],
        "source_sha256": _h(source),
        "worlds": worlds,
        "runtime": environment,
    }
    return {
        "schema": "q1-runtime-parity-receipt/v1",
        "pass": True,
        "bank_sha256": _h(bank),
        "b8_sha256": q1_evaluate.B8_SHA256,
        "source_manifest": source,
        "environment": {"sha256": _h(environment)},
        "reference": artifact,
        "runtime_artifact": copy.deepcopy(artifact),
        "comparison": q1_parity.compare_artifacts(artifact, artifact),
    }


def _validate(receipt):
    q1_parity.validate_runtime_receipt(
        receipt,
        bank_sha256=_h(q1_bank.load_world_bank()),
        b8_sha256=q1_evaluate.B8_SHA256,
        source_manifest=q1_evaluate.source_manifest(),
    )


def test_valid_synthetic_receipt_and_wrong_namespace():
    receipt = _receipt()
    _validate(receipt)
    receipt["runtime_artifact"]["bank_namespace"] = "wrong"
    with pytest.raises(ValueError):
        _validate(receipt)


@pytest.mark.parametrize(
    "kind", ["forged_pass", "environment", "source", "worlds", "bank", "mask", "nonmac"]
)
def test_runtime_receipt_cannot_self_assert_parity(kind):
    receipt = _receipt()
    runtime = receipt["runtime_artifact"]
    if kind == "forged_pass":
        runtime["worlds"][0]["actions"][0]["action"] = 2
    elif kind == "environment":
        receipt["environment"]["sha256"] = "0" * 64
    elif kind == "source":
        for artifact in (runtime, receipt["reference"]):
            artifact["source_sha256"] = "0" * 64
    elif kind == "worlds":
        for artifact in (runtime, receipt["reference"]):
            artifact["worlds"][0]["world_id"] = "unregistered"
    elif kind == "bank":
        for artifact in (runtime, receipt["reference"]):
            artifact["bank_sha256"] = "0" * 64
    elif kind == "mask":
        runtime["worlds"][0]["actions"][0]["mask"][1] = False
    elif kind == "nonmac":
        receipt["reference"]["runtime"]["system"] = "Linux"
    if kind != "forged_pass":
        receipt["comparison"] = q1_parity.compare_artifacts(receipt["reference"], runtime)
    with pytest.raises(ValueError):
        _validate(receipt)
