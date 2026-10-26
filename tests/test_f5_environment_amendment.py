"""Verify the complete prospective F5 environment without scientific execution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pip._internal.network.session import PipSession
from pip._internal.req.req_file import parse_requirements

ROOT = Path(__file__).parents[1]
PACKAGE = ROOT / "research/bl_residual/f5_adoption_environment_20261026_v6"


def record() -> dict:
    return json.loads((PACKAGE / "f5-common-environment-v6.json").read_bytes())


def test_every_environment_artifact_is_sealed_and_bound() -> None:
    value = record()
    for path in PACKAGE.iterdir():
        if path.name.endswith(".sha256"):
            continue
        expected, name = path.with_name(path.name + ".sha256").read_text().split()
        assert path.name == name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    for env in value["common_environments"].values():
        lock = env["package_lock"]
        assert hashlib.sha256((PACKAGE / lock["path"]).read_bytes()).hexdigest() == lock["sha256"]
    build = value["common_environments"]["mac_cpu_stages"]["build_identity"]
    assert hashlib.sha256((PACKAGE / build["path"]).read_bytes()).hexdigest() == build["sha256"]


def test_l4_lock_parses_as_complete_hashed_installable_requirements() -> None:
    l4 = record()["common_environments"]["l4_paired_fits"]
    lock = PACKAGE / l4["package_lock"]["path"]
    requirements = list(parse_requirements(str(lock), session=PipSession()))
    assert len(requirements) == 69
    assert all(item.options.get("hashes", {}).get("sha256") for item in requirements)
    torch = next(x for x in requirements if x.requirement.startswith("torch=="))
    assert torch.requirement.strip() == "torch==2.9.1+cu126"
    assert l4["torch_wheel_sha256"] in torch.options["hashes"]["sha256"]
    assert "--index-url https://download.pytorch.org/whl/cu126" in l4["package_lock"]["install"]
    assert "--require-hashes" in l4["package_lock"]["install"]


def test_stage_placement_and_determinism_preserve_r8() -> None:
    value = record()
    l4 = value["common_environments"]["l4_paired_fits"]
    mac = value["common_environments"]["mac_cpu_stages"]
    assert l4["python"] == mac["python"] == "CPython 3.12.3"
    assert l4["arms"] == ["C", "T"]
    assert l4["stage_workers"]["exact_continuation_if_COV"] == 6
    assert l4["stage_workers"]["fresh_archive_retention_labeling"] == 6
    assert mac["stage_workers"] == {
        "drift": 2,
        "gameplay": 2,
        "product": 2,
        "offline_reduction": 1,
        "nice": 15,
        "math_threads_per_worker": 1,
    }
    assert l4["determinism"]["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert l4["determinism"]["cudnn_benchmark"] is False
    assert all(x in ("1", 1) for x in value["thread_settings"].values())
    assert "CPU model/ISA/vCPU allocation" in value["hardware_receipt"]["equality"]
    assert "create-only receipt" in value["hardware_receipt"]["publication"]


def test_batch_source_and_lossless_continuation_state_are_frozen() -> None:
    value = record()
    batch = value["batch_plan"]
    assert batch["seed"] == 3403 and batch["updates"] == 2000
    assert batch["batch"] == {"total": 256, "doom": 128, "ordinary": 128}
    assert "512000" in batch["rule"] and "sorted life groups" in batch["rule"]
    assert value["source_bindings"]["semantic_commit"] == "124dd22cd560a74a210a61f0ba473461328e82f8"
    assert value["source_bindings"]["fit_py_sha256_at_semantic_commit"] == (
        "85c7e77d34e14a3b5036984cc1af92df4e3724f4c9fe453d813a56c1963822a7"
    )
    assert "torch.cuda.get_rng_state_all" in batch["initial_boundary"]
    assert "source hash" in value["continuation_boundary"]["record"]
    assert "FitState optimizer_state hash" in value["continuation_boundary"]["record"]
    assert "not hashes alone" in value["rng_state_recording"]["serialization"]
    assert "opponent carry" in value["cov"]["required"]
    assert value["cov"]["failure"].startswith("HOLD_COVERAGE_IMPLEMENTATION")
    assert value["historical_f4_identity_claimed"] is False
