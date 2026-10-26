"""Synthetic, payload-free tests for the F5 assignment identity schema."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from research.bl_residual import f5_environment as environment


def _hash(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _metadata(raw: bytes, root: str, name: str) -> dict[str, str]:
    return {
        "raw_base64": base64.b64encode(raw).decode("ascii"),
        "raw_sha256": _hash(raw),
        "canonical_sha256": _hash(environment._canonical_metadata_bytes(raw, root, name)),
    }


def _package(root: str) -> dict[str, Any]:
    metadata = {
        "METADATA": _metadata(b"Name: fixture\nVersion: 1\n", root, "METADATA"),
        "WHEEL": _metadata(b"Wheel-Version: 1.0\n", root, "WHEEL"),
        "RECORD": _metadata(f"{root}/fixture.py,,\n".encode(), root, "RECORD"),
        "direct_url.json": _metadata(b"<ABSENT>", root, "direct_url.json"),
    }
    contents = {
        "fixture/__init__.py": {
            "raw_sha256": _hash(b"fixed installed library bytes"),
            "canonical_sha256": _hash(b"fixed installed library bytes"),
        }
    }
    return {
        "version": "1.0",
        "installation_root": root,
        "installed_metadata": metadata,
        "installed_contents": contents,
        "installed_contents_sha256": _hash(
            environment.canonical_json(
                {key: item["canonical_sha256"] for key, item in contents.items()}
            ).encode()
        ),
    }


def _receipt(*, uuid: str = "A", node: str = "host-a", root: str = "/venv-a") -> dict[str, Any]:
    packages = {name: _package(root) for name in sorted(environment.NUMERIC_PACKAGES)}
    packages["torch"]["version"] = "2.9.1+cu126"
    packages["numpy"]["version"] = "2.1"
    wheels = {
        name: {
            "filename": f"{name}-1.0-py3-none-any.whl",
            "raw_url": f"file://{root}/wheelhouse/{name}-1.0-py3-none-any.whl",
            "canonical_url": f"file://<WHEELHOUSE>/{name}-1.0-py3-none-any.whl",
            "sha256": _hash(f"wheel-{name}".encode()),
        }
        for name in sorted(environment.NUMERIC_PACKAGES)
    }
    return {
        "schema": environment.SCHEMA,
        "capture_status": "SUCCESS",
        "schema_sha256": environment.schema_sha256(),
        "platform": "Linux-6.1-x86_64",
        "python": "3.12.3 fixture",
        "cpu": "fixture CPU",
        "cpu_count": 8,
        "torch": "2.9.1+cu126",
        "cuda": "12.6",
        "gpu": "NVIDIA L4",
        "gpu_uuid": uuid,
        "gpu_capability": [8, 9],
        "gpu_memory": 23034 * environment.MIB_BYTES,
        "memory_units": dict(environment.MEMORY_UNITS),
        "os_kernel": {
            "system": "Linux",
            "node": node,
            "release": "6.1",
            "version": "#1",
            "machine": "x86_64",
            "processor": "",
        },
        "python_build": ["main", "fixture"],
        "python_compiler": "GCC",
        "python_cache_tag": "cpython-312",
        "cpu_isa": {"model name": "fixture CPU", "flags": "avx2"},
        "allocated_cpu_affinity": [0, 1, 2, 3, 4, 5, 6, 7],
        "gpu_driver_query": {
            "name": "NVIDIA L4",
            "uuid": f"GPU-{uuid}",
            "driver_version": "550",
            "compute_cap": "8.9",
            "memory_total": "23034",
        },
        "gpu_driver_query_raw": f"NVIDIA L4, GPU-{uuid}, 550, 8.9, 23034",
        "cudnn": "90100",
        "torch_git_version": "fixture-git",
        "torch_build": "fixed build",
        "numpy": "2.1",
        "numpy_blas_lapack_build": "fixed blas",
        "installed_numeric_packages": packages,
        "installed_packages": copy.deepcopy(packages),
        "installed_versions_sha256": _hash(b"all package versions"),
        "environment_lock_sha256": _hash(b"lock"),
        "environment_descriptor_sha256": _hash(b"descriptor"),
        "locked_wheels": wheels,
        "torch_threads": {"intra": 1, "inter": 1},
        "numeric_controls": {
            "deterministic": True,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "matmul_tf32": False,
            "cudnn_tf32": False,
            "default_dtype": "torch.float32",
            "amp": False,
            "bf16": False,
        },
        "thread_env": {
            name: "1" for name in environment.THREAD_ENV_KEYS if name != "CUBLAS_WORKSPACE_CONFIG"
        }
        | {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
    }


def test_cross_assignment_allows_only_documented_provenance() -> None:
    first = _receipt()
    second = _receipt(uuid="B", node="host-b", root="/venv-b")
    report = environment.compare_environment(first, second, mode="cross-assignment")
    assert report["expected_raw_sha256"] != report["actual_raw_sha256"]
    assert report["expected_canonical_sha256"] == report["actual_canonical_sha256"]
    assert report["assignment_provenance"]["actual"]["gpu_uuid"] == "B"


def test_same_assignment_requires_complete_raw_identity() -> None:
    first = _receipt()
    changed = _receipt(uuid="B")
    with pytest.raises(environment.F5EnvironmentRefusal, match="same-assignment"):
        environment.compare_environment(first, changed, mode="same-assignment")


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("numeric_controls", "matmul_tf32"), True),
        (("gpu_driver_query", "driver_version"), "999"),
        (("installed_packages", "torch", "installed_contents", "fixture/__init__.py"), "0" * 64),
        (("locked_wheels", "torch", "sha256"), "0" * 64),
    ],
)
def test_cross_assignment_refuses_genuine_build_or_numerical_drift(
    path: tuple[str, ...], value: Any
) -> None:
    first = _receipt()
    changed = _receipt(uuid="B", node="host-b", root="/venv-b")
    target: Any = changed
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(environment.F5EnvironmentRefusal):
        environment.compare_environment(first, changed, mode="cross-assignment")


def test_unknown_nested_field_and_malformed_boolean_refuse() -> None:
    receipt = _receipt()
    receipt["installed_packages"]["torch"]["unexpected"] = "not classified"
    with pytest.raises(environment.F5EnvironmentRefusal, match="unclassified"):
        environment.validate_environment(receipt)
    receipt = _receipt()
    receipt["cpu_count"] = True
    with pytest.raises(environment.F5EnvironmentRefusal, match="cpu_count"):
        environment.validate_environment(receipt)
    receipt = _receipt()
    receipt["gpu_driver_query_raw"] = "NVIDIA L4, forged, 550, 8.9, 23034"
    with pytest.raises(environment.F5EnvironmentRefusal, match="raw GPU query"):
        environment.validate_environment(receipt)


@pytest.mark.parametrize("smi_uuid", ["gpu-A", "A", "GPU-GPU-A", "GPU-"])
def test_v4_requires_the_exact_single_smi_uuid_prefix(smi_uuid: str) -> None:
    receipt = _receipt()
    receipt["gpu_driver_query"]["uuid"] = smi_uuid
    receipt["gpu_driver_query_raw"] = f"NVIDIA L4, {smi_uuid}, 550, 8.9, 23034"
    with pytest.raises(environment.F5EnvironmentRefusal, match="SMI UUID"):
        environment.validate_environment(receipt)


def test_v4_retains_raw_torch_and_smi_identity_and_refuses_name_aliases() -> None:
    receipt = _receipt(uuid="aBc-123")
    assert receipt["gpu_uuid"] == "aBc-123"
    assert receipt["gpu_driver_query"]["uuid"] == "GPU-aBc-123"
    assert environment.normalize_smi_uuid("GPU-aBc-123") == "aBc-123"
    environment.validate_environment(receipt)

    receipt["gpu_driver_query"]["name"] = "NVIDIA  L4"
    receipt["gpu_driver_query_raw"] = "NVIDIA  L4, GPU-aBc-123, 550, 8.9, 23034"
    with pytest.raises(environment.F5EnvironmentRefusal, match="GPU name"):
        environment.validate_environment(receipt)


def test_v4_collects_all_real_runtime_source_agreement_failures() -> None:
    receipt = _receipt()
    receipt["gpu_uuid"] = "B"
    receipt["gpu"] = "Tesla L4"
    receipt["gpu_capability"] = [8, 6]
    receipt["gpu_memory"] = 23036 * environment.MIB_BYTES
    failures = environment.collect_environment_failures(receipt)
    assert set(failures) >= {
        "F5 environment GPU UUID provenance fields disagree",
        "F5 environment GPU name provenance fields disagree",
        "F5 environment parsed GPU capability differs",
        "F5 environment parsed GPU memory differs",
    }


def test_raw_metadata_is_auditable_and_cannot_claim_false_canonical_mapping() -> None:
    receipt = _receipt(root="/venv-a")
    receipt["installed_packages"]["torch"]["installed_metadata"]["RECORD"]["canonical_sha256"] = (
        "0" * 64
    )
    receipt["installed_numeric_packages"]["torch"] = receipt["installed_packages"]["torch"]
    with pytest.raises(environment.F5EnvironmentRefusal, match="canonical metadata hash"):
        environment.validate_environment(receipt)


def test_each_nested_leaf_has_exactly_one_frozen_classification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = _receipt()
    classified = environment.classify_receipt(receipt)
    assert classified[json.dumps(("gpu_driver_query", "driver_version"))] == "invariant"
    assert classified[json.dumps(("os_kernel", "node"))] == "assignment-provenance"
    artifact = json.loads(environment.schema_path().read_text())
    artifact["leaf_rules"].append(
        {
            "path": ["gpu_driver_query", "*"],
            "comparison": "assignment-provenance",
            "type": "string",
        }
    )
    path = tmp_path / "ambiguous-schema.json"
    path.write_text(json.dumps(artifact))
    monkeypatch.setattr(environment, "schema_path", lambda: path)
    with pytest.raises(environment.F5EnvironmentRefusal, match="multiply classified"):
        environment.classify_receipt(receipt)
    artifact["leaf_rules"] = [
        rule for rule in artifact["leaf_rules"] if rule["path"] != ["platform"]
    ]
    path.write_text(json.dumps(artifact))
    with pytest.raises(environment.F5EnvironmentRefusal, match="unclassified"):
        environment.classify_receipt(receipt)


def test_encoded_wheel_url_and_console_record_normalization_are_auditable() -> None:
    receipt = _receipt()
    wheel = receipt["locked_wheels"]["torch"]
    wheel["filename"] = "torch-2.9.1+cu126-py3-none-any.whl"
    wheel["raw_url"] = "file:///venv/wheelhouse/torch-2.9.1%2Bcu126-py3-none-any.whl"
    wheel["canonical_url"] = "file://<WHEELHOUSE>/torch-2.9.1+cu126-py3-none-any.whl"
    environment.validate_environment(receipt)
    raw_script = b"#!/venv-a/bin/python\nprint('fixed')\n"
    contents = {
        "__root_bound__/bin/fixture": {
            "raw_base64": base64.b64encode(raw_script).decode(),
            "raw_sha256": _hash(raw_script),
            "canonical_sha256": _hash(b"#!<INSTALLATION_ROOT>/bin/python\nprint('fixed')\n"),
        }
    }
    raw_record = b"../../../bin/fixture,sha256=old,1\nmodule.py,sha256=fixed,2\n"
    normalized = environment._canonical_metadata_bytes(raw_record, "/venv-a", "RECORD", contents)
    assert b"../../../bin/fixture,sha256=" in normalized
    assert b"old" not in normalized
    assert b"module.py,sha256=fixed,2" in normalized


def test_v3_rejects_unknown_or_legacy_root_bound_content_classes() -> None:
    receipt = _receipt()
    package = receipt["installed_packages"]["torch"]
    assert isinstance(package, dict)
    contents = package["installed_contents"]
    assert isinstance(contents, dict)
    contents["__root_bound__/etc/f5.conf"] = {
        "raw_sha256": _hash(b"fixed"),
        "canonical_sha256": _hash(b"fixed"),
    }
    with pytest.raises(environment.F5EnvironmentRefusal, match="root-bound"):
        environment.validate_environment(receipt)

    legacy = _receipt()
    legacy_package = legacy["installed_packages"]["torch"]
    assert isinstance(legacy_package, dict)
    legacy_contents = legacy_package["installed_contents"]
    assert isinstance(legacy_contents, dict)
    legacy_contents["__entrypoint__/old"] = {
        "raw_base64": base64.b64encode(b"#!/venv-a/bin/python\n").decode(),
        "raw_sha256": _hash(b"#!/venv-a/bin/python\n"),
        "canonical_sha256": _hash(b"#!<INSTALLATION_ROOT>/bin/python\n"),
    }
    with pytest.raises(environment.F5EnvironmentRefusal, match="legacy entrypoint"):
        environment.validate_environment(legacy)


def test_root_script_canonicalization_refuses_a_sibling_installation_prefix() -> None:
    """``/venv-a`` must never match a script from ``/venv-a-other``."""
    assert (
        environment._canonical_root_bound_script(b"#!/venv-a-other/bin/python\n", "/venv-a") is None
    )


def test_direct_url_uses_exact_verified_sibling_wheelhouse_url() -> None:
    raw = b'{"url":"file:///content/generation-a/f5-wheelhouse/torch-2.9.1%2Bcu126.whl"}'
    wheel = {
        "raw_url": "file:///content/generation-a/f5-wheelhouse/torch-2.9.1%2Bcu126.whl",
        "canonical_url": "file://<WHEELHOUSE>/torch-2.9.1+cu126.whl",
    }
    assert (
        environment._canonical_metadata_bytes(
            raw, "/content/generation-a/f5-cp3123", "direct_url.json", wheel=wheel
        )
        == b'{"url":"file://<WHEELHOUSE>/torch-2.9.1+cu126.whl"}'
    )
    with pytest.raises(environment.F5EnvironmentRefusal, match="verified locked wheel"):
        environment._canonical_metadata_bytes(
            b'{"url":"file:///other/wheel.whl"}',
            "/content/generation-a/f5-cp3123",
            "direct_url.json",
            wheel=wheel,
        )
