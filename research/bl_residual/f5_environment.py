"""F5 environment receipt capture and assignment-identity admission.

This module deliberately treats an environment receipt as typed evidence, not a
free-form diagnostic blob.  The frozen companion schema documents every field
and whether it is an invariant or assignment-local provenance.  A receipt with
an unknown, missing, malformed, or ambiguously classified field is refused.
"""

from __future__ import annotations

import base64
import contextlib
import csv
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import unquote, urlparse

import numpy as np
import torch

SCHEMA = "bl-residual-f5-environment-receipt/v2"
AMENDMENT_SHA256 = "7eda19efb72f89a7e2269d40863288f9295358f2566f3a02d3ca4d3c11933359"
ADOPTED_AMENDMENT_SHA256 = "a1dc92ffb8fa341fb51738a56fe0097d75e15b0aff7194940c12979adbb04d62"

# The implementation packet carries the digest for this JSON as a source pin.
# It is intentionally a fixed list: a new capture field requires a reviewed
# schema version rather than becoming silently non-comparable evidence.
REQUIRED_TOP_LEVEL = frozenset(
    {
        "schema",
        "schema_sha256",
        "platform",
        "python",
        "cpu",
        "cpu_count",
        "torch",
        "cuda",
        "gpu",
        "gpu_uuid",
        "gpu_capability",
        "gpu_memory",
        "memory_units",
        "os_kernel",
        "python_build",
        "python_compiler",
        "python_cache_tag",
        "cpu_isa",
        "allocated_cpu_affinity",
        "gpu_driver_query",
        "gpu_driver_query_raw",
        "cudnn",
        "torch_git_version",
        "torch_build",
        "numpy",
        "numpy_blas_lapack_build",
        "installed_numeric_packages",
        "installed_packages",
        "installed_versions_sha256",
        "environment_lock_sha256",
        "environment_descriptor_sha256",
        "locked_wheels",
        "torch_threads",
        "numeric_controls",
        "thread_env",
    }
)
NUMERIC_PACKAGES = frozenset({"torch", "numpy", "numba", "llvmlite", "scipy"})
REQUIRED_NUMERIC_PACKAGES = NUMERIC_PACKAGES - {"scipy"}
MEMORY_UNITS = {"gpu_memory": "bytes", "gpu_driver_query.memory_total": "MiB"}
MIB_BYTES = 1024 * 1024
PACKAGE_KEYS = frozenset(
    {
        "version",
        "installation_root",
        "installed_metadata",
        "installed_contents",
        "installed_contents_sha256",
    }
)
METADATA_KEYS = frozenset({"METADATA", "WHEEL", "RECORD", "direct_url.json"})
METADATA_RECEIPT_KEYS = frozenset({"raw_base64", "raw_sha256", "canonical_sha256"})
CONTENT_RECEIPT_KEYS = frozenset({"raw_sha256", "canonical_sha256"})
ENTRYPOINT_CONTENT_RECEIPT_KEYS = frozenset({"raw_base64", "raw_sha256", "canonical_sha256"})
LOCKED_WHEEL_KEYS = frozenset({"filename", "raw_url", "canonical_url", "sha256"})
OS_KERNEL_KEYS = frozenset({"system", "node", "release", "version", "machine", "processor"})
GPU_DRIVER_KEYS = frozenset({"name", "uuid", "driver_version", "compute_cap", "memory_total"})
THREAD_KEYS = frozenset({"intra", "inter"})
NUMERIC_CONTROL_KEYS = frozenset(
    {
        "deterministic",
        "cudnn_deterministic",
        "cudnn_benchmark",
        "matmul_tf32",
        "cudnn_tf32",
        "default_dtype",
        "amp",
        "bf16",
    }
)
THREAD_ENV_KEYS = frozenset(
    {
        "OMP_NUM_THREADS",
        "OMP_THREAD_LIMIT",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "CUBLAS_WORKSPACE_CONFIG",
    }
)


class F5EnvironmentRefusal(ValueError):
    """A receipt cannot establish the prospective F5 environment identity."""


def schema_path() -> Path:
    """Return the frozen, source-pinned classification schema."""
    return Path(__file__).with_name("f5_environment_schema_v2.json")


def schema_sha256() -> str:
    """Hash the reviewed schema bytes, never a generated approximation."""
    return hashlib.sha256(schema_path().read_bytes()).hexdigest()


def policy_binding() -> dict[str, str]:
    """Return the exact prospective control binding for this comparison policy."""
    return {
        "schema": SCHEMA,
        "schema_sha256": schema_sha256(),
        "draft_amendment_sha256": AMENDMENT_SHA256,
        "adopted_amendment_sha256": ADOPTED_AMENDMENT_SHA256,
    }


def validate_schema_artifact() -> None:
    """Refuse an edited schema that no longer declares exhaustive classifications."""
    try:
        artifact = json.loads(schema_path().read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise F5EnvironmentRefusal("F5 environment schema artifact is unreadable") from exc
    classification = artifact.get("field_classification") if isinstance(artifact, dict) else None
    required = {
        "invariant_text",
        "invariant_integer",
        "invariant_sha256",
        "invariant_exact_object",
        "raw_and_canonical_evidence",
        "assignment_provenance_only",
        "dynamic_maps",
        "forbidden",
    }
    if (
        artifact.get("schema") != SCHEMA
        or not isinstance(classification, dict)
        or set(classification) != required
    ):
        raise F5EnvironmentRefusal("F5 environment schema classification is incomplete")
    leaves = [
        item for key, value in classification.items() if isinstance(value, list) for item in value
    ]
    if len(leaves) != len(set(leaves)) or classification["forbidden"] != [
        "additional top-level keys",
        "additional nested keys",
        "duplicate classification",
        "unclassified leaves",
    ]:
        raise F5EnvironmentRefusal("F5 environment schema classifications are ambiguous")


def classify_receipt(receipt: Mapping[str, Any]) -> dict[str, str]:
    """Require exactly one frozen classification for every fully nested leaf."""
    artifact = json.loads(schema_path().read_text())
    rules = artifact.get("leaf_rules")
    if not isinstance(rules, list) or not rules:
        raise F5EnvironmentRefusal("F5 environment leaf classifications are missing")
    allowed = {"invariant", "assignment-provenance", "verified-raw-to-canonical"}
    for rule in rules:
        if (
            not isinstance(rule, dict)
            or set(rule) != {"path", "comparison", "type"}
            or not isinstance(rule["path"], list)
            or not rule["path"]
            or any(not isinstance(part, str) or not part for part in rule["path"])
            or rule["comparison"] not in allowed
            or not isinstance(rule["type"], str)
            or not rule["type"]
        ):
            raise F5EnvironmentRefusal("F5 environment leaf rule is malformed")
    classified: dict[str, str] = {}

    def visit(value: Any, path: tuple[str, ...]) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if not isinstance(key, str):
                    raise F5EnvironmentRefusal("F5 environment map key is malformed")
                visit(child, (*path, key))
            return
        matches = [
            rule
            for rule in rules
            if len(rule["path"]) == len(path)
            and all(pattern == "*" or pattern == part for pattern, part in zip(rule["path"], path))
        ]
        if len(matches) != 1:
            raise F5EnvironmentRefusal("F5 environment leaf is unclassified or multiply classified")
        classified[json.dumps(path)] = matches[0]["comparison"]

    visit(dict(receipt), ())
    return classified


def canonical_json(value: Any) -> str:
    """Return the only JSON serialization used for evidence hashes."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def receipt_sha256(value: Mapping[str, Any]) -> str:
    """Hash raw receipt bytes semantically after strict typed validation."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _exact_mapping(value: Any, keys: frozenset[str], path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise F5EnvironmentRefusal(f"F5 environment {path} has unknown or missing fields")
    return value


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise F5EnvironmentRefusal(f"F5 environment {path} must be a nonempty string")
    return value


def _sha(value: Any, path: str) -> str:
    value = _text(value, path)
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise F5EnvironmentRefusal(f"F5 environment {path} must be a lowercase SHA-256")
    return value


def _validate_package(name: str, value: Any, wheel: Mapping[str, str] | None = None) -> None:
    package = _exact_mapping(value, PACKAGE_KEYS, f"installed_numeric_packages.{name}")
    _text(package["version"], f"installed_numeric_packages.{name}.version")
    _text(package["installation_root"], f"installed_numeric_packages.{name}.installation_root")
    contents = package["installed_contents"]
    if not isinstance(contents, dict) or not contents:
        raise F5EnvironmentRefusal(
            f"F5 environment installed_numeric_packages.{name}.contents missing"
        )
    for relative, digest in contents.items():
        if (
            not isinstance(relative, str)
            or not relative
            or relative.startswith("/")
            or ".." in Path(relative).parts
        ):
            raise F5EnvironmentRefusal(f"F5 environment installed contents path is unsafe: {name}")
        keys = (
            ENTRYPOINT_CONTENT_RECEIPT_KEYS
            if relative.startswith("__entrypoint__/")
            else CONTENT_RECEIPT_KEYS
        )
        item = _exact_mapping(
            digest, keys, f"installed_numeric_packages.{name}.contents.{relative}"
        )
        _sha(item["raw_sha256"], f"installed_numeric_packages.{name}.contents.{relative}.raw")
        _sha(
            item["canonical_sha256"],
            f"installed_numeric_packages.{name}.contents.{relative}.canonical",
        )
        if relative.startswith("__entrypoint__/"):
            try:
                raw = base64.b64decode(item["raw_base64"], validate=True)
            except (ValueError, TypeError) as exc:
                raise F5EnvironmentRefusal(f"F5 entrypoint evidence is malformed: {name}") from exc
            if hashlib.sha256(raw).hexdigest() != item["raw_sha256"]:
                raise F5EnvironmentRefusal(f"F5 entrypoint raw hash differs: {name}")
            first, separator, rest = raw.partition(b"\n")
            prefix = b"#!" + package["installation_root"].encode("utf-8")
            if not first.startswith(prefix):
                raise F5EnvironmentRefusal(
                    f"F5 entrypoint lacks exact installation-root shebang: {name}"
                )
            canonical = b"#!<INSTALLATION_ROOT>" + first[len(prefix) :] + separator + rest
            if hashlib.sha256(canonical).hexdigest() != item["canonical_sha256"]:
                raise F5EnvironmentRefusal(f"F5 entrypoint canonical mapping differs: {name}")
        elif item["raw_sha256"] != item["canonical_sha256"]:
            raise F5EnvironmentRefusal(f"F5 installed library content was normalized: {name}")
    canonical_contents = {relative: item["canonical_sha256"] for relative, item in contents.items()}
    if (
        hashlib.sha256(canonical_json(canonical_contents).encode()).hexdigest()
        != package["installed_contents_sha256"]
    ):
        raise F5EnvironmentRefusal(f"F5 environment installed contents digest differs: {name}")
    metadata = _exact_mapping(
        package["installed_metadata"], METADATA_KEYS, f"installed_numeric_packages.{name}.metadata"
    )
    for metadata_name, item in metadata.items():
        hashes = _exact_mapping(
            item, METADATA_RECEIPT_KEYS, f"installed_numeric_packages.{name}.{metadata_name}"
        )
        _sha(hashes["raw_sha256"], f"installed_numeric_packages.{name}.{metadata_name}.raw")
        _sha(
            hashes["canonical_sha256"],
            f"installed_numeric_packages.{name}.{metadata_name}.canonical",
        )
        try:
            raw = base64.b64decode(hashes["raw_base64"], validate=True)
        except (ValueError, TypeError) as exc:
            raise F5EnvironmentRefusal(f"F5 environment raw metadata is malformed: {name}") from exc
        if hashlib.sha256(raw).hexdigest() != hashes["raw_sha256"]:
            raise F5EnvironmentRefusal(f"F5 environment raw metadata hash differs: {name}")
        canonical = _canonical_metadata_bytes(
            raw, package["installation_root"], metadata_name, contents, wheel
        )
        if hashlib.sha256(canonical).hexdigest() != hashes["canonical_sha256"]:
            raise F5EnvironmentRefusal(f"F5 environment canonical metadata hash differs: {name}")


def validate_environment(receipt: Mapping[str, Any]) -> None:
    """Refuse every receipt whose fully nested schema cannot be classified."""
    validate_schema_artifact()
    top = _exact_mapping(receipt, REQUIRED_TOP_LEVEL, "receipt")
    classify_receipt(top)
    if top["schema"] != SCHEMA:
        raise F5EnvironmentRefusal("F5 environment receipt schema differs")
    _sha(top["schema_sha256"], "schema_sha256")
    if top["schema_sha256"] != schema_sha256():
        raise F5EnvironmentRefusal("F5 environment schema source pin differs")
    if top["memory_units"] != MEMORY_UNITS:
        raise F5EnvironmentRefusal("F5 environment GPU memory units differ")
    for name in (
        "platform",
        "python",
        "cpu",
        "torch",
        "cuda",
        "gpu",
        "gpu_uuid",
        "python_compiler",
        "python_cache_tag",
        "cudnn",
        "torch_git_version",
        "torch_build",
        "numpy",
        "numpy_blas_lapack_build",
        "environment_lock_sha256",
        "environment_descriptor_sha256",
    ):
        _text(top[name], name)
    if (
        isinstance(top["cpu_count"], bool)
        or not isinstance(top["cpu_count"], int)
        or top["cpu_count"] < 1
    ):
        raise F5EnvironmentRefusal("F5 environment cpu_count is malformed")
    if (
        isinstance(top["gpu_memory"], bool)
        or not isinstance(top["gpu_memory"], int)
        or top["gpu_memory"] <= 0
    ):
        raise F5EnvironmentRefusal("F5 environment gpu_memory is malformed")
    if (
        not isinstance(top["gpu_capability"], list)
        or len(top["gpu_capability"]) != 2
        or not all(
            not isinstance(part, bool) and isinstance(part, int) and part >= 0
            for part in top["gpu_capability"]
        )
    ):
        raise F5EnvironmentRefusal("F5 environment gpu_capability is malformed")
    _exact_mapping(top["os_kernel"], OS_KERNEL_KEYS, "os_kernel")
    for key, value in top["os_kernel"].items():
        if not isinstance(value, str):
            raise F5EnvironmentRefusal(f"F5 environment os_kernel.{key} is malformed")
    if not isinstance(top["python_build"], list) or len(top["python_build"]) != 2:
        raise F5EnvironmentRefusal("F5 environment python_build is malformed")
    for index, value in enumerate(top["python_build"]):
        _text(value, f"python_build[{index}]")
    if not isinstance(top["cpu_isa"], dict) or set(top["cpu_isa"]) - {
        "model name",
        "flags",
        "Features",
        "Hardware",
    }:
        raise F5EnvironmentRefusal("F5 environment cpu_isa has unknown fields")
    if not top["cpu_isa"]:
        raise F5EnvironmentRefusal("F5 environment cpu_isa is missing")
    for key, value in top["cpu_isa"].items():
        _text(value, f"cpu_isa.{key}")
    if (
        not isinstance(top["allocated_cpu_affinity"], list)
        or not top["allocated_cpu_affinity"]
        or not all(
            not isinstance(value, bool) and isinstance(value, int) and value >= 0
            for value in top["allocated_cpu_affinity"]
        )
    ):
        raise F5EnvironmentRefusal("F5 environment allocated_cpu_affinity is malformed")
    if (
        len(top["allocated_cpu_affinity"]) != len(set(top["allocated_cpu_affinity"]))
        or len(top["allocated_cpu_affinity"]) < 6
    ):
        raise F5EnvironmentRefusal("F5 environment requires six unique allocated vCPUs")
    driver = _exact_mapping(top["gpu_driver_query"], GPU_DRIVER_KEYS, "gpu_driver_query")
    for key, value in driver.items():
        _text(value, f"gpu_driver_query.{key}")
    raw_driver = _text(top["gpu_driver_query_raw"], "gpu_driver_query_raw")
    raw_parts = [part.strip() for part in raw_driver.split(",")]
    if len(raw_parts) != 5 or raw_parts != [
        driver["name"],
        driver["uuid"],
        driver["driver_version"],
        driver["compute_cap"],
        driver["memory_total"],
    ]:
        raise F5EnvironmentRefusal("F5 environment raw GPU query disagrees with parsed fields")
    if driver["uuid"] != top["gpu_uuid"] or driver["name"] != top["gpu"]:
        raise F5EnvironmentRefusal("F5 environment GPU provenance fields disagree")
    try:
        capability = [int(part) for part in driver["compute_cap"].split(".")]
        memory = int(driver["memory_total"])
    except ValueError as exc:
        raise F5EnvironmentRefusal(
            "F5 environment parsed GPU capability/memory is malformed"
        ) from exc
    # SMI reports integer MiB; Torch reports bytes. Admit only a sub-MiB
    # quantization discrepancy, never a larger accounting difference. Each raw
    # observation remains an exact invariant in cross-assignment comparison.
    if (
        capability != top["gpu_capability"]
        or memory <= 0
        or abs(memory * MIB_BYTES - top["gpu_memory"]) >= MIB_BYTES
    ):
        raise F5EnvironmentRefusal("F5 environment parsed GPU capability/memory differs")
    packages = _exact_mapping(top["installed_numeric_packages"], NUMERIC_PACKAGES, "packages")
    all_packages = top["installed_packages"]
    if not isinstance(all_packages, dict) or not set(all_packages) >= REQUIRED_NUMERIC_PACKAGES:
        raise F5EnvironmentRefusal("F5 environment installed package closure is incomplete")
    wheels = top["locked_wheels"]
    if not isinstance(wheels, dict) or not set(wheels) >= REQUIRED_NUMERIC_PACKAGES:
        raise F5EnvironmentRefusal("F5 locked wheel closure is incomplete")
    if not set(all_packages) >= set(wheels):
        raise F5EnvironmentRefusal("F5 installed closure omits a locked wheel package")
    if ("scipy" in all_packages) != ("scipy" in wheels):
        raise F5EnvironmentRefusal("F5 optional scipy installed/locked presence differs")
    for name, wheel in wheels.items():
        wheel = _exact_mapping(wheel, LOCKED_WHEEL_KEYS, f"locked_wheels.{name}")
        _text(wheel["filename"], f"locked_wheels.{name}.filename")
        _text(wheel["raw_url"], f"locked_wheels.{name}.raw_url")
        canonical_url = _text(wheel["canonical_url"], f"locked_wheels.{name}.canonical_url")
        raw_url = urlparse(wheel["raw_url"])
        if raw_url.scheme != "file" or not unquote(raw_url.path).endswith("/" + wheel["filename"]):
            raise F5EnvironmentRefusal(f"F5 locked wheel URL is not a wheelhouse file: {name}")
        if canonical_url != "file://<WHEELHOUSE>/" + wheel["filename"]:
            raise F5EnvironmentRefusal(f"F5 locked wheel canonical URL differs: {name}")
        _sha(wheel["sha256"], f"locked_wheels.{name}.sha256")
    for name, value in all_packages.items():
        if not isinstance(name, str) or not name or name != name.lower() or "_" in name:
            raise F5EnvironmentRefusal("F5 environment installed package name is malformed")
        _validate_package(name, value, wheels.get(name))
    if {name: all_packages.get(name) for name in NUMERIC_PACKAGES} != packages:
        raise F5EnvironmentRefusal("F5 numeric package receipt disagrees with installed closure")
    for name in ("torch", "numpy"):
        if top[name] != all_packages[name]["version"]:
            raise F5EnvironmentRefusal(f"F5 {name} version disagrees with installed closure")
    _sha(top["installed_versions_sha256"], "installed_versions_sha256")
    _sha(top["environment_lock_sha256"], "environment_lock_sha256")
    _sha(top["environment_descriptor_sha256"], "environment_descriptor_sha256")
    threads = _exact_mapping(top["torch_threads"], THREAD_KEYS, "torch_threads")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in threads.values()
    ):
        raise F5EnvironmentRefusal("F5 environment torch thread value is malformed")
    controls = _exact_mapping(top["numeric_controls"], NUMERIC_CONTROL_KEYS, "numeric_controls")
    for key, value in controls.items():
        if key == "default_dtype":
            _text(value, "numeric_controls.default_dtype")
        elif not isinstance(value, bool):
            raise F5EnvironmentRefusal(f"F5 environment numeric_controls.{key} is malformed")
    thread_env = _exact_mapping(top["thread_env"], THREAD_ENV_KEYS, "thread_env")
    for key, value in thread_env.items():
        _text(value, f"thread_env.{key}")


def _canonical_package(package: Mapping[str, Any]) -> dict[str, Any]:
    """Retain raw hashes while excluding only the demonstrated install root."""
    return {
        "version": package["version"],
        "installed_contents_sha256": package["installed_contents_sha256"],
        "installed_metadata": {
            name: item["canonical_sha256"] for name, item in package["installed_metadata"].items()
        },
        "installed_contents": {
            relative: item["canonical_sha256"]
            for relative, item in package["installed_contents"].items()
        },
    }


def canonical_environment(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Build auditable cross-assignment identity; never erase build bytes."""
    validate_environment(receipt)
    canonical = dict(receipt)
    canonical["gpu_uuid"] = "ASSIGNMENT_LOCAL"
    canonical["allocated_cpu_affinity"] = "ASSIGNMENT_LOCAL"
    canonical["os_kernel"] = dict(canonical["os_kernel"])
    canonical["os_kernel"]["node"] = "ASSIGNMENT_LOCAL"
    canonical["gpu_driver_query"] = dict(canonical["gpu_driver_query"])
    canonical["gpu_driver_query"]["uuid"] = "ASSIGNMENT_LOCAL"
    raw_query = receipt["gpu_driver_query_raw"]
    uuid = receipt["gpu_driver_query"]["uuid"]
    if raw_query.count(uuid) != 1:
        raise F5EnvironmentRefusal("F5 raw driver query UUID is ambiguous")
    canonical["gpu_driver_query_raw"] = raw_query.replace(uuid, "ASSIGNMENT_LOCAL", 1)
    canonical["installed_numeric_packages"] = {
        name: None if package is None else _canonical_package(package)
        for name, package in receipt["installed_numeric_packages"].items()
    }
    canonical["installed_packages"] = {
        name: _canonical_package(package) for name, package in receipt["installed_packages"].items()
    }
    canonical["locked_wheels"] = {
        name: {
            "filename": wheel["filename"],
            "url": wheel["canonical_url"],
            "sha256": wheel["sha256"],
        }
        for name, wheel in receipt["locked_wheels"].items()
    }
    return canonical


def compare_environment(
    expected: Mapping[str, Any], actual: Mapping[str, Any], *, mode: str = "cross-assignment"
) -> dict[str, Any]:
    """Compare raw same-assignment identity or adopted cross-assignment invariants.

    ``same-assignment`` requires the complete raw receipt to match.  Cross
    assignment keeps UUID, host node, affinity numbers and package root only as
    provenance, while all build and numerical identity remains invariant.
    """
    if mode not in {"cross-assignment", "same-assignment"}:
        raise F5EnvironmentRefusal("F5 environment comparison mode is unknown")
    validate_environment(expected)
    validate_environment(actual)
    expected_raw_sha = receipt_sha256(expected)
    actual_raw_sha = receipt_sha256(actual)
    expected_canonical = canonical_environment(expected)
    actual_canonical = canonical_environment(actual)
    if mode == "same-assignment":
        if expected_raw_sha != actual_raw_sha:
            raise F5EnvironmentRefusal("F5 same-assignment environment raw receipt differs")
    elif expected_canonical != actual_canonical:
        raise F5EnvironmentRefusal("F5 cross-assignment invariant environment differs")
    return {
        **policy_binding(),
        "field_classification": classify_receipt(actual),
        "mode": mode,
        "expected_raw_sha256": expected_raw_sha,
        "actual_raw_sha256": actual_raw_sha,
        "expected_canonical_sha256": hashlib.sha256(
            canonical_json(expected_canonical).encode()
        ).hexdigest(),
        "actual_canonical_sha256": hashlib.sha256(
            canonical_json(actual_canonical).encode()
        ).hexdigest(),
        "assignment_provenance": {
            "expected": {
                "gpu_uuid": expected["gpu_uuid"],
                "node": expected["os_kernel"]["node"],
                "cpu_affinity": expected["allocated_cpu_affinity"],
                "installation_roots": {
                    name: package["installation_root"]
                    for name, package in expected["installed_numeric_packages"].items()
                    if package is not None
                },
            },
            "actual": {
                "gpu_uuid": actual["gpu_uuid"],
                "node": actual["os_kernel"]["node"],
                "cpu_affinity": actual["allocated_cpu_affinity"],
                "installation_roots": {
                    name: package["installation_root"]
                    for name, package in actual["installed_numeric_packages"].items()
                    if package is not None
                },
            },
        },
    }


def _file_sha(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise F5EnvironmentRefusal(f"F5 environment evidence is not a regular file: {path.name}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_metadata_bytes(
    raw: bytes,
    installation_root: str,
    metadata_name: str,
    contents: Mapping[str, Mapping[str, str]] | None = None,
    wheel: Mapping[str, str] | None = None,
) -> bytes:
    """Normalize roots only in the two path-bearing metadata formats.

    METADATA and WHEEL define build contents and are deliberately byte-exact.
    RECORD and direct_url.json may contain the captured install/wheelhouse root;
    their raw bytes remain in the receipt and only that exact root is replaced.
    """
    if metadata_name not in METADATA_KEYS:
        raise F5EnvironmentRefusal("F5 unknown metadata normalization target")
    if metadata_name in {"METADATA", "WHEEL"}:
        return raw
    root = installation_root.encode("utf-8")
    if not root:
        raise F5EnvironmentRefusal("F5 installation root is empty")
    if metadata_name == "direct_url.json" and wheel is not None:
        if raw == b"<ABSENT>":
            return raw
        raw_url = wheel["raw_url"].encode("utf-8")
        try:
            direct = json.loads(raw)
        except (ValueError, UnicodeDecodeError) as exc:
            raise F5EnvironmentRefusal("F5 direct_url is malformed JSON") from exc
        if (
            not isinstance(direct, dict)
            or direct.get("url") != wheel["raw_url"]
            or raw.count(raw_url) != 1
        ):
            raise F5EnvironmentRefusal("F5 direct_url does not bind the verified locked wheel URL")
        return raw.replace(raw_url, wheel["canonical_url"].encode("utf-8"), 1)
    normalized = raw.replace(root, b"<INSTALLATION_ROOT>")
    if metadata_name != "RECORD" or contents is None:
        return normalized
    # RECORD hashes generated console scripts, whose shebang contains the venv
    # root. Recompute only those rows from the retained raw script evidence.
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    try:
        rows = csv.reader(io.StringIO(normalized.decode("utf-8")))
        for row in rows:
            if len(row) == 3 and row[0].startswith("../../../bin/"):
                item = contents.get("__entrypoint__/" + Path(row[0]).name)
                if item is None or "raw_base64" not in item:
                    raise F5EnvironmentRefusal(
                        "F5 RECORD console-script row lacks raw entrypoint evidence"
                    )
                raw_script = base64.b64decode(item["raw_base64"], validate=True)
                first, separator, rest = raw_script.partition(b"\n")
                prefix = b"#!" + root
                if not first.startswith(prefix):
                    raise F5EnvironmentRefusal("F5 RECORD entrypoint shebang differs")
                canonical_script = (
                    b"#!<INSTALLATION_ROOT>" + first[len(prefix) :] + separator + rest
                )
                row[1] = "sha256=" + base64.urlsafe_b64encode(
                    hashlib.sha256(canonical_script).digest()
                ).decode("ascii").rstrip("=")
                row[2] = str(len(canonical_script))
            writer.writerow(row)
    except (UnicodeDecodeError, csv.Error) as exc:
        raise F5EnvironmentRefusal("F5 RECORD is not UTF-8 CSV") from exc
    return output.getvalue().encode("utf-8")


def _package_receipt(name: str, wheel: Mapping[str, str] | None = None) -> dict[str, Any]:
    try:
        distribution = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise F5EnvironmentRefusal(f"F5 required numeric package is absent: {name}") from exc
    dist_info = next(
        (item for item in distribution.files or () if ".dist-info/" in str(item)), None
    )
    if dist_info is None:
        raise F5EnvironmentRefusal(f"F5 package lacks dist-info evidence: {name}")
    site_root = Path(distribution.locate_file(dist_info)).parents[1].resolve()
    root = str(site_root.parents[2])
    metadata_raw: dict[str, bytes] = {}
    for metadata_name in sorted(METADATA_KEYS):
        item = next(
            (
                candidate
                for candidate in distribution.files or ()
                if candidate.name == metadata_name and ".dist-info/" in str(candidate)
            ),
            None,
        )
        raw = b"<ABSENT>" if item is None else Path(distribution.locate_file(item)).read_bytes()
        metadata_raw[metadata_name] = raw
    contents: dict[str, dict[str, str]] = {}
    for item in distribution.files or ():
        path = Path(distribution.locate_file(item))
        # dist-info paths are separately retained as raw metadata.  Do not
        # silently normalize them through this installed-library manifest.
        if path.is_symlink():
            raise F5EnvironmentRefusal(f"F5 package has an unproven symlink: {name}")
        if not path.is_file():
            raise F5EnvironmentRefusal(f"F5 package has a missing installed file: {name}")
        if ".dist-info/" in str(item) and item.name in METADATA_KEYS:
            # These four bytes are retained separately with raw evidence and
            # the narrowly permitted canonical mapping where applicable.
            continue
        if path.is_file():
            if path.suffix == ".pyc" or "__pycache__" in path.parts:
                raise F5EnvironmentRefusal(
                    "F5 locked environment must use pip --no-compile; pyc is root-dependent"
                )
            try:
                if ".dist-info/" in str(item):
                    key = "__dist_info__/" + str(item)
                else:
                    relative = path.resolve().relative_to(site_root)
                    key = str(relative)
                raw = path.read_bytes()
                canonical = raw
            except ValueError:
                # pip's RECORD legitimately lists console scripts as
                # ../../../bin/foo.  Their generated shebang is an
                # installation-root representation, not library code; retain
                # its raw digest and canonicalize only that exact prefix.
                key = "__entrypoint__/" + path.name
                raw = path.read_bytes()
                first, separator, rest = raw.partition(b"\n")
                prefix = b"#!" + root.encode("utf-8")
                if not first.startswith(prefix):
                    raise F5EnvironmentRefusal(
                        f"F5 external package file is not a root-bound console script: {name}"
                    )
                canonical = b"#!<INSTALLATION_ROOT>" + first[len(prefix) :] + separator + rest
            if key in contents:
                raise F5EnvironmentRefusal(f"F5 package records duplicate installed path: {name}")
            contents[key] = {
                "raw_sha256": hashlib.sha256(raw).hexdigest(),
                "canonical_sha256": hashlib.sha256(canonical).hexdigest(),
            }
            if key.startswith("__entrypoint__/"):
                contents[key]["raw_base64"] = base64.b64encode(raw).decode("ascii")
    if not contents:
        raise F5EnvironmentRefusal(f"F5 package has no installed-content evidence: {name}")
    metadata = {
        metadata_name: {
            "raw_base64": base64.b64encode(raw).decode("ascii"),
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "canonical_sha256": hashlib.sha256(
                _canonical_metadata_bytes(raw, root, metadata_name, contents, wheel)
            ).hexdigest(),
        }
        for metadata_name, raw in metadata_raw.items()
    }
    return {
        "version": distribution.version,
        "installation_root": root,
        "installed_metadata": metadata,
        "installed_contents": contents,
        "installed_contents_sha256": hashlib.sha256(
            canonical_json(
                {key: item["canonical_sha256"] for key, item in contents.items()}
            ).encode()
        ).hexdigest(),
    }


def _wheel_manifest() -> dict[str, dict[str, str]]:
    """Verify report-selected locked wheels against the required local wheelhouse."""
    report_path = Path(os.environ.get("F5_INSTALL_REPORT", ""))
    wheelhouse = Path(os.environ.get("F5_WHEELHOUSE", ""))
    if not report_path.is_file() or not wheelhouse.is_dir():
        raise F5EnvironmentRefusal("F5 requires F5_INSTALL_REPORT and F5_WHEELHOUSE evidence")
    try:
        report = json.loads(report_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise F5EnvironmentRefusal("F5 install report is unreadable") from exc
    installs = report.get("install") if isinstance(report, dict) else None
    if not isinstance(installs, list):
        raise F5EnvironmentRefusal("F5 install report has no install inventory")
    lock = (
        Path(__file__).with_name("f5_adoption_environment_20261026_v6")
        / "requirements-f5-l4-cu126.lock"
    )
    expected: dict[str, set[str]] = {}
    current: str | None = None
    for line in lock.read_text().splitlines():
        package = re.match(r"^([A-Za-z0-9][A-Za-z0-9_.-]*)==[^\\\s]+", line)
        if package:
            current = package.group(1).lower().replace("_", "-")
            expected[current] = set()
        for digest in re.findall(r"--hash=sha256:([0-9a-f]{64})", line):
            if current is None:
                raise F5EnvironmentRefusal("F5 lock hash precedes a package")
            expected[current].add(digest)
    if not expected or any(not hashes for hashes in expected.values()):
        raise F5EnvironmentRefusal("F5 lock lacks complete package hash evidence")
    found: dict[str, dict[str, str]] = {}
    for item in installs:
        if not isinstance(item, dict) or not isinstance(item.get("metadata"), dict):
            raise F5EnvironmentRefusal("F5 install report has malformed package item")
        name = str(item["metadata"].get("name", "")).lower().replace("_", "-")
        if not name:
            raise F5EnvironmentRefusal("F5 install report has unnamed package item")
        download = item.get("download_info")
        archive = download.get("archive_info") if isinstance(download, dict) else None
        hash_value = archive.get("hash") if isinstance(archive, dict) else None
        url = download.get("url") if isinstance(download, dict) else None
        if (
            not isinstance(hash_value, str)
            or not hash_value.startswith("sha256=")
            or not isinstance(url, str)
        ):
            raise F5EnvironmentRefusal(f"F5 install report lacks wheel hash for {name}")
        sha = hash_value.removeprefix("sha256=")
        _sha(sha, f"install report wheel {name}")
        filename = Path(unquote(urlparse(url).path)).name
        local = wheelhouse / filename
        if not local.is_file() or _file_sha(local) != sha:
            raise F5EnvironmentRefusal(f"F5 wheelhouse does not verify reported wheel {name}")
        if name in found:
            raise F5EnvironmentRefusal(f"F5 install report repeats numeric package {name}")
        root_url = wheelhouse.resolve().as_uri().rstrip("/") + "/"
        if not url.startswith(root_url):
            raise F5EnvironmentRefusal(f"F5 install report wheel is outside the wheelhouse: {name}")
        found[name] = {
            "filename": filename,
            "raw_url": url,
            "canonical_url": "file://<WHEELHOUSE>/" + filename,
            "sha256": sha,
        }
    if set(found) != set(expected):
        raise F5EnvironmentRefusal(
            "F5 install report does not cover the complete locked wheel closure"
        )
    for name, wheel in found.items():
        if wheel["sha256"] not in expected[name]:
            raise F5EnvironmentRefusal(f"F5 report wheel hash is absent from lock: {name}")
    return found


def _driver_query() -> tuple[dict[str, str], str]:
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,compute_cap,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise F5EnvironmentRefusal("F5 cannot capture parsed nvidia-smi identity") from exc
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 5 or not all(parts):
        raise F5EnvironmentRefusal("F5 nvidia-smi query is malformed")
    return (
        dict(
            zip(
                ("name", "uuid", "driver_version", "compute_cap", "memory_total"),
                parts,
                strict=True,
            )
        ),
        raw,
    )


def capture_environment() -> dict[str, Any]:
    """Capture full raw L4 provenance plus portable build identity evidence."""
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise F5EnvironmentRefusal("F5 capture requires exactly one CUDA device")
    device = torch.cuda.get_device_properties(0)
    cpu_fields: dict[str, str] = {}
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            key, separator, value = line.partition(":")
            if separator and key.strip() in {"model name", "flags", "Features", "Hardware"}:
                cpu_fields.setdefault(key.strip(), value.strip())
    if not cpu_fields:
        raise F5EnvironmentRefusal("F5 capture requires CPU model/ISA evidence")
    numpy_build = io.StringIO()
    with contextlib.redirect_stdout(numpy_build):
        np.__config__.show()
    installed = sorted(
        (str(dist.metadata.get("Name", "")).lower(), dist.version)
        for dist in importlib.metadata.distributions()
    )
    environment_dir = Path(__file__).with_name("f5_adoption_environment_20261026_v6")
    lock = environment_dir / "requirements-f5-l4-cu126.lock"
    descriptor = environment_dir / "f5-common-environment-v6.json"
    if _file_sha(lock) != "2577ed9dd9fe34468f5562262def7bccb352c9c0bdce06b293bb4b12a0ea5301":
        raise F5EnvironmentRefusal("F5 environment lock differs")
    if _file_sha(descriptor) != "bc05167c188680478663235f2b3dcfb70d7959e9d392071384409ef958b0f0f5":
        raise F5EnvironmentRefusal("F5 common environment descriptor differs")
    uname = platform.uname()._asdict()
    driver, driver_raw = _driver_query()
    wheels = _wheel_manifest()
    all_packages = {
        name: _package_receipt(name, wheels.get(name))
        for name in sorted(
            {
                str(dist.metadata.get("Name", "")).lower().replace("_", "-")
                for dist in importlib.metadata.distributions()
                if str(dist.metadata.get("Name", ""))
            }
        )
    }
    receipt = {
        "schema": SCHEMA,
        "schema_sha256": schema_sha256(),
        "platform": platform.platform(),
        "python": sys.version,
        "cpu": platform.processor() or cpu_fields.get("model name", "unknown"),
        "cpu_count": os.cpu_count() or 0,
        "torch": torch.__version__,
        "cuda": torch.version.cuda or "none",
        "gpu": device.name,
        "gpu_uuid": str(device.uuid),
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
        "gpu_memory": int(device.total_memory),
        "memory_units": dict(MEMORY_UNITS),
        "os_kernel": {
            "system": uname["system"],
            "node": uname["node"],
            "release": uname["release"],
            "version": uname["version"],
            "machine": uname["machine"],
            "processor": uname["processor"] or "unknown",
        },
        "python_build": list(platform.python_build()),
        "python_compiler": platform.python_compiler(),
        "python_cache_tag": sys.implementation.cache_tag,
        "cpu_isa": cpu_fields,
        "allocated_cpu_affinity": sorted(os.sched_getaffinity(0)),
        "gpu_driver_query": driver,
        "gpu_driver_query_raw": driver_raw,
        "cudnn": str(torch.backends.cudnn.version()),
        "torch_git_version": torch.version.git_version or "unknown",
        "torch_build": torch.__config__.show(),
        "numpy": np.__version__,
        "numpy_blas_lapack_build": numpy_build.getvalue(),
        "installed_numeric_packages": {
            name: all_packages.get(name) for name in sorted(NUMERIC_PACKAGES)
        },
        "installed_packages": all_packages,
        "installed_versions_sha256": hashlib.sha256(canonical_json(installed).encode()).hexdigest(),
        "environment_lock_sha256": _file_sha(lock),
        "environment_descriptor_sha256": _file_sha(descriptor),
        "locked_wheels": wheels,
        "torch_threads": {
            "intra": torch.get_num_threads(),
            "inter": torch.get_num_interop_threads(),
        },
        "numeric_controls": {
            "deterministic": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "default_dtype": str(torch.get_default_dtype()),
            "amp": torch.is_autocast_enabled("cuda"),
            "bf16": torch.is_autocast_enabled("cuda")
            and torch.get_autocast_dtype("cuda") == torch.bfloat16,
        },
        "thread_env": {name: os.environ.get(name, "") for name in sorted(THREAD_ENV_KEYS)},
    }
    validate_environment(receipt)
    return receipt
