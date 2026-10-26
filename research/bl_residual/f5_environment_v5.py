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
from pathlib import Path, PurePosixPath
from typing import Any, Mapping
from urllib.parse import unquote, urlparse

import numpy as np
import torch

SCHEMA = "bl-residual-f5-environment-receipt/v5"
AMENDMENT_SHA256 = "7eda19efb72f89a7e2269d40863288f9295358f2566f3a02d3ca4d3c11933359"
ADOPTED_AMENDMENT_SHA256 = "a1dc92ffb8fa341fb51738a56fe0097d75e15b0aff7194940c12979adbb04d62"

# The implementation packet carries the digest for this JSON as a source pin.
# It is intentionally a fixed list: a new capture field requires a reviewed
# schema version rather than becoming silently non-comparable evidence.
REQUIRED_TOP_LEVEL = frozenset(
    {
        "schema",
        "capture_status",
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
# A fixed physical-consistency guard, never a cross-assignment equality tolerance.
GPU_MEMORY_MIN_NUMERATOR = 95
GPU_MEMORY_MIN_DENOMINATOR = 100
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
ROOT_BOUND_SCRIPT_RECEIPT_KEYS = frozenset({"raw_base64", "raw_sha256", "canonical_sha256"})
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
# These are the reviewed F5 installation-prefix classes.  A new class is a
# receipt-schema change, never an implicit extension of the captured
# environment; the 69-wheel RECORD audit reports which reviewed classes occur.
ROOT_BOUND_FILE_CLASSES = frozenset({"bin", "include", "share"})


class F5EnvironmentRefusal(ValueError):
    """A receipt cannot establish the prospective F5 environment identity."""


class F5CrossAssignmentRefusal(F5EnvironmentRefusal):
    """Carry outcome-blind refusal diagnostics without creating admissible evidence."""

    def __init__(self, diagnostic: dict[str, Any]) -> None:
        super().__init__("F5 cross-assignment invariant environment differs")
        self.diagnostic = diagnostic


def schema_path() -> Path:
    """Return the frozen, source-pinned classification schema."""
    return Path(__file__).with_name("f5_environment_schema_v5.json")


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
    """Fingerprint raw receipt content; this hash alone never establishes admission."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _exact_mapping(value: Any, keys: frozenset[str], path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise F5EnvironmentRefusal(f"F5 environment {path} has unknown or missing fields")
    return value


def normalize_smi_uuid(smi_uuid: str) -> str:
    """Return the one permitted SMI-to-Torch UUID normalization.

    ``nvidia-smi`` prints a case-sensitive ``GPU-`` prefix that PyTorch device
    properties omit.  This deliberately strips exactly that single prefix from
    the SMI value only; lower-case, absent, doubled, and empty forms are
    evidence failures rather than aliases.
    """
    if not isinstance(smi_uuid, str) or not smi_uuid.startswith("GPU-"):
        raise F5EnvironmentRefusal("F5 environment SMI UUID lacks exact GPU- prefix")
    normalized = smi_uuid.removeprefix("GPU-")
    if not normalized or normalized.startswith("GPU-"):
        raise F5EnvironmentRefusal("F5 environment SMI UUID has invalid GPU- prefix count")
    return normalized


def _source_consistency_failures(receipt: Mapping[str, Any]) -> list[str]:
    """Collect independent real-runtime source agreement failures.

    The fields originate in APIs with intentionally different representations:
    Torch reports UUID without ``GPU-`` and memory in bytes, while SMI reports
    UUID with the prefix, framebuffer capacity in MiB, and a dotted capability string. Torch usable
    capacity and SMI framebuffer capacity are different quantities: require
    95% <= Torch/SMI <= 100%, using exact integer arithmetic. Neither raw
    observation is rounded or relaxed in cross-assignment comparisons.
    Keep every raw observation and report every independently checkable
    disagreement so an L4 diagnostic run is not consumed one bug at a time.
    """
    failures: list[str] = []
    driver = receipt.get("gpu_driver_query")
    raw_driver = receipt.get("gpu_driver_query_raw")
    if not isinstance(driver, Mapping):
        return ["F5 environment GPU driver query is unavailable for source comparison"]
    required = ("name", "uuid", "driver_version", "compute_cap", "memory_total")
    for key in required:
        if not isinstance(driver.get(key), str) or not driver[key]:
            failures.append(
                f"F5 environment GPU driver query {key} is malformed for source comparison"
            )
    if not isinstance(raw_driver, str):
        failures.append("F5 environment raw GPU query is malformed")
    else:
        raw_parts = [part.strip() for part in raw_driver.split(",")]
        expected_parts = [driver.get(key) for key in required]
        if len(raw_parts) != 5 or raw_parts != expected_parts:
            failures.append("F5 environment raw GPU query disagrees with parsed fields")

    torch_uuid = receipt.get("gpu_uuid")
    if not isinstance(torch_uuid, str) or not torch_uuid:
        failures.append("F5 environment Torch GPU UUID is malformed")
    else:
        try:
            smi_uuid = normalize_smi_uuid(driver.get("uuid"))
        except F5EnvironmentRefusal as exc:
            failures.append(str(exc))
        else:
            if smi_uuid != torch_uuid:
                failures.append("F5 environment GPU UUID provenance fields disagree")

    torch_name = receipt.get("gpu")
    if not isinstance(torch_name, str) or not torch_name:
        failures.append("F5 environment Torch GPU name is malformed")
    elif driver.get("name") != torch_name:
        failures.append("F5 environment GPU name provenance fields disagree")

    capability = receipt.get("gpu_capability")
    try:
        parsed_capability = [int(part) for part in driver.get("compute_cap", "").split(".")]
    except (ValueError, TypeError, AttributeError):
        failures.append("F5 environment parsed SMI GPU capability is malformed")
    else:
        if len(parsed_capability) != 2 or capability != parsed_capability:
            failures.append("F5 environment parsed GPU capability differs")

    memory_bytes = receipt.get("gpu_memory")
    try:
        memory_mib = int(driver.get("memory_total"))
    except (ValueError, TypeError):
        failures.append("F5 environment parsed SMI GPU memory is malformed")
    else:
        if (
            memory_mib <= 0
            or isinstance(memory_bytes, bool)
            or not isinstance(memory_bytes, int)
            or memory_bytes > memory_mib * MIB_BYTES
            or (
                memory_bytes * GPU_MEMORY_MIN_DENOMINATOR
                < memory_mib * MIB_BYTES * GPU_MEMORY_MIN_NUMERATOR
            )
        ):
            failures.append("F5 environment parsed GPU memory differs")
    return failures


def collect_environment_failures(receipt: Mapping[str, Any]) -> list[str]:
    """Return all independently observable receipt failures for diagnostics.

    The complete structural validator remains authoritative.  Source agreement
    checks are collected separately even when that validator already refused,
    allowing a non-admissible raw diagnostic receipt to expose every remaining
    Torch/SMI representation defect from the same L4 allocation.
    """
    failures: list[str] = []
    try:
        _validate_environment_first(receipt)
    except (F5EnvironmentRefusal, KeyError, TypeError, AttributeError, ValueError) as exc:
        failures.append(str(exc))
    # Probe independent leaf groups after the authoritative whole-receipt
    # validator stops.  Each probe is bounded and read-only; it never repairs
    # or normalizes evidence, so a diagnostic list cannot weaken admission.
    top = receipt if isinstance(receipt, Mapping) else {}
    if set(top) != REQUIRED_TOP_LEVEL:
        failures.append("F5 environment receipt has unknown or missing fields")
    if top.get("capture_status") != "SUCCESS":
        failures.append("F5 environment capture status is not SUCCESS")
    if top.get("schema") != SCHEMA:
        failures.append("F5 environment receipt schema differs")
    if top.get("schema_sha256") != schema_sha256():
        failures.append("F5 environment schema source pin differs")
    if top.get("memory_units") != MEMORY_UNITS:
        failures.append("F5 environment GPU memory units differ")
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
        if not isinstance(top.get(name), str) or not top.get(name):
            failures.append(f"F5 environment {name} must be a nonempty string")
    if (
        isinstance(top.get("cpu_count"), bool)
        or not isinstance(top.get("cpu_count"), int)
        or top.get("cpu_count", 0) < 1
    ):
        failures.append("F5 environment cpu_count is malformed")
    if (
        isinstance(top.get("gpu_memory"), bool)
        or not isinstance(top.get("gpu_memory"), int)
        or top.get("gpu_memory", 0) <= 0
    ):
        failures.append("F5 environment gpu_memory is malformed")
    capability = top.get("gpu_capability")
    if (
        not isinstance(capability, list)
        or len(capability) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in capability
        )
    ):
        failures.append("F5 environment gpu_capability is malformed")
    build = top.get("python_build")
    if (
        not isinstance(build, list)
        or len(build) != 2
        or any(not isinstance(value, str) or not value for value in build)
    ):
        failures.append("F5 environment python_build is malformed")
    affinity = top.get("allocated_cpu_affinity")
    if (
        not isinstance(affinity, list)
        or not affinity
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in affinity
        )
        or (
            isinstance(affinity, list)
            and (len(affinity) != len(set(affinity)) or len(affinity) < 6)
        )
    ):
        failures.append("F5 environment allocated_cpu_affinity is malformed")
    os_kernel = top.get("os_kernel")
    if (
        not isinstance(os_kernel, dict)
        or set(os_kernel) != OS_KERNEL_KEYS
        or any(not isinstance(value, str) for value in os_kernel.values())
    ):
        failures.append("F5 environment os_kernel is malformed")
    controls = top.get("numeric_controls")
    if (
        not isinstance(controls, dict)
        or set(controls) != NUMERIC_CONTROL_KEYS
        or any(
            (key != "default_dtype" and not isinstance(value, bool))
            or (key == "default_dtype" and (not isinstance(value, str) or not value))
            for key, value in controls.items()
        )
    ):
        failures.append("F5 environment numeric controls are malformed")
    threads = top.get("torch_threads")
    if (
        not isinstance(threads, dict)
        or set(threads) != THREAD_KEYS
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in threads.values()
        )
    ):
        failures.append("F5 environment torch thread value is malformed")
    thread_env = top.get("thread_env")
    if (
        not isinstance(thread_env, dict)
        or set(thread_env) != THREAD_ENV_KEYS
        or any(not isinstance(value, str) or not value for value in thread_env.values())
    ):
        failures.append("F5 environment thread environment is malformed")
    packages = top.get("installed_packages")
    wheels = top.get("locked_wheels")
    if isinstance(packages, dict):
        wheel_map = wheels if isinstance(wheels, dict) else {}
        for name, package in packages.items():
            failures.extend(collect_package_failures(str(name), package, wheel_map.get(str(name))))
    else:
        failures.append("F5 environment installed package closure is incomplete")

    def probe(operation: Any) -> None:
        try:
            operation()
        except (F5EnvironmentRefusal, KeyError, TypeError, ValueError, AttributeError) as exc:
            failures.append(str(exc))

    for key in (
        "schema_sha256",
        "installed_versions_sha256",
        "environment_lock_sha256",
        "environment_descriptor_sha256",
    ):
        probe(lambda: _sha(top.get(key), key))
    for group, keys in (
        ("os_kernel", OS_KERNEL_KEYS),
        ("torch_threads", THREAD_KEYS),
        ("numeric_controls", NUMERIC_CONTROL_KEYS),
        ("thread_env", THREAD_ENV_KEYS),
    ):
        value = top.get(group)
        if not isinstance(value, dict):
            continue
        for key in keys:
            item = value.get(key)
            if group == "torch_threads":
                valid = isinstance(item, int) and not isinstance(item, bool) and item >= 1
            elif group == "numeric_controls" and key != "default_dtype":
                valid = isinstance(item, bool)
            else:
                valid = isinstance(item, str) and (group == "os_kernel" or bool(item))
            if not valid:
                failures.append(f"F5 environment {group}.{key} is malformed")
    isa = top.get("cpu_isa")
    if (
        not isinstance(isa, dict)
        or not isa
        or set(isa) - {"model name", "flags", "Features", "Hardware"}
    ):
        failures.append("F5 environment cpu_isa has unknown fields or is missing")
    if isinstance(isa, dict):
        for key, value in isa.items():
            probe(lambda: _text(value, f"cpu_isa.{key}"))
    driver = top.get("gpu_driver_query")
    probe(lambda: _exact_mapping(driver, GPU_DRIVER_KEYS, "gpu_driver_query"))
    if isinstance(driver, dict):
        for key, value in driver.items():
            probe(lambda: _text(value, f"gpu_driver_query.{key}"))
    probe(lambda: _text(top.get("gpu_driver_query_raw"), "gpu_driver_query_raw"))
    aliases = top.get("installed_numeric_packages")
    probe(lambda: _exact_mapping(aliases, NUMERIC_PACKAGES, "packages"))
    if isinstance(packages, dict):
        if not set(packages) >= REQUIRED_NUMERIC_PACKAGES:
            failures.append("F5 environment installed package closure is incomplete")
        if {name: packages.get(name) for name in NUMERIC_PACKAGES} != aliases:
            failures.append("F5 numeric package receipt disagrees with installed closure")
        for key in ("torch", "numpy"):
            package = packages.get(key)
            if isinstance(package, dict) and top.get(key) != package.get("version"):
                failures.append(f"F5 {key} version disagrees with installed closure")
        for name in packages:
            if not isinstance(name, str) or not name or name != name.lower() or "_" in name:
                failures.append(f"F5 environment installed package name is malformed: {name}")
    if isinstance(wheels, dict):
        if not set(wheels) >= REQUIRED_NUMERIC_PACKAGES:
            failures.append("F5 locked wheel closure is incomplete")
        if isinstance(packages, dict):
            if not set(packages) >= set(wheels):
                failures.append("F5 installed closure omits a locked wheel package")
            if ("scipy" in packages) != ("scipy" in wheels):
                failures.append("F5 optional scipy installed/locked presence differs")
        for name, wheel in wheels.items():
            probe(lambda: _exact_mapping(wheel, LOCKED_WHEEL_KEYS, f"locked_wheels.{name}"))
            if not isinstance(wheel, dict):
                continue
            for key in ("filename", "raw_url", "canonical_url"):
                probe(lambda: _text(wheel.get(key), f"locked_wheels.{name}.{key}"))
            probe(lambda: _sha(wheel.get("sha256"), f"locked_wheels.{name}.sha256"))
            if isinstance(wheel.get("raw_url"), str) and isinstance(wheel.get("filename"), str):
                url = urlparse(wheel["raw_url"])
                if url.scheme != "file" or not unquote(url.path).endswith("/" + wheel["filename"]):
                    failures.append(f"F5 locked wheel URL is not a wheelhouse file: {name}")
                if wheel.get("canonical_url") != "file://<WHEELHOUSE>/" + wheel["filename"]:
                    failures.append(f"F5 locked wheel canonical URL differs: {name}")
    for failure in _source_consistency_failures(receipt):
        if failure not in failures:
            failures.append(failure)
    return list(dict.fromkeys(failures))


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise F5EnvironmentRefusal(f"F5 environment {path} must be a nonempty string")
    return value


def _sha(value: Any, path: str) -> str:
    value = _text(value, path)
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise F5EnvironmentRefusal(f"F5 environment {path} must be a lowercase SHA-256")
    return value


def _root_bound_content_kind(relative: str) -> str | None:
    """Return the admitted root-bound class encoded by a receipt content key."""
    prefix = "__root_bound__/"
    if (
        relative == "__root_bound__"
        or relative.startswith("__root_bound__")
        and not relative.startswith(prefix)
    ):
        raise F5EnvironmentRefusal("F5 root-bound installed content namespace is malformed")
    if not relative.startswith(prefix):
        return None
    path = PurePosixPath(relative.removeprefix(prefix))
    if (
        path.is_absolute()
        or len(path.parts) < 2
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.parts[0] not in ROOT_BOUND_FILE_CLASSES
    ):
        raise F5EnvironmentRefusal("F5 root-bound installed content class is unsafe or unknown")
    return path.parts[0]


def _canonical_root_bound_script(raw: bytes, installation_root: str) -> bytes | None:
    """Canonicalize only a generated script with an exact root/bin interpreter."""
    first, separator, rest = raw.partition(b"\n")
    prefix = b"#!" + installation_root.encode("utf-8") + b"/"
    if not first.startswith(prefix):
        return None
    interpreter = first[len(prefix) :].split(None, 1)[0]
    parts = PurePosixPath(interpreter.decode("utf-8", "surrogateescape")).parts
    if len(parts) < 2 or parts[0] != "bin" or any(part in {"", ".", ".."} for part in parts):
        raise F5EnvironmentRefusal("F5 root-bound script interpreter is unsafe")
    return b"#!<INSTALLATION_ROOT>/" + first[len(prefix) :] + separator + rest


def collect_package_failures(
    name: str, value: Any, wheel: Mapping[str, str] | None = None
) -> list[str]:
    """Inspect every package leaf in linear time without copying file inventories."""
    failures: list[str] = []

    def probe(operation: Any) -> Any:
        try:
            return operation()
        except (F5EnvironmentRefusal, KeyError, TypeError, ValueError, AttributeError) as exc:
            failures.append(str(exc))
            return None

    probe(lambda: _exact_mapping(value, PACKAGE_KEYS, f"installed_numeric_packages.{name}"))
    if not isinstance(value, dict):
        return failures
    for key in ("version", "installation_root"):
        probe(lambda: _text(value.get(key), f"installed_numeric_packages.{name}.{key}"))
    contents = value.get("installed_contents")
    if not isinstance(contents, dict) or not contents:
        failures.append(f"F5 environment installed_numeric_packages.{name}.contents missing")
        contents = {}
    for relative, item in contents.items():
        path = f"installed_numeric_packages.{name}.contents.{relative}"
        if (
            not isinstance(relative, str)
            or not relative
            or relative.startswith("/")
            or ".." in Path(relative).parts
        ):
            failures.append(f"F5 environment installed contents path is unsafe: {name}")
            continue
        kind = probe(lambda: _root_bound_content_kind(relative))
        if relative.startswith("__entrypoint__/"):
            failures.append("F5 v3 does not admit legacy entrypoint content keys")
        script = kind == "bin" and isinstance(item, dict) and "raw_base64" in item
        probe(
            lambda: _exact_mapping(
                item, ROOT_BOUND_SCRIPT_RECEIPT_KEYS if script else CONTENT_RECEIPT_KEYS, path
            )
        )
        if not isinstance(item, dict):
            continue
        for key in ("raw_sha256", "canonical_sha256"):
            probe(lambda: _sha(item.get(key), f"{path}.{key}"))
        if script:
            raw = probe(lambda: base64.b64decode(item.get("raw_base64"), validate=True))
            if raw is not None:
                if hashlib.sha256(raw).hexdigest() != item.get("raw_sha256"):
                    failures.append(f"F5 entrypoint raw hash differs: {name}: {relative}")
                canonical = probe(
                    lambda: _canonical_root_bound_script(raw, value.get("installation_root"))
                )
                if canonical is None:
                    failures.append(
                        f"F5 entrypoint lacks exact installation-root shebang: {name}: {relative}"
                    )
                elif hashlib.sha256(canonical).hexdigest() != item.get("canonical_sha256"):
                    failures.append(f"F5 entrypoint canonical mapping differs: {name}: {relative}")
        elif item.get("raw_sha256") != item.get("canonical_sha256"):
            failures.append(f"F5 installed library content was normalized: {name}: {relative}")
    canonical_contents = {
        relative: item.get("canonical_sha256") if isinstance(item, dict) else None
        for relative, item in contents.items()
    }
    aggregate = probe(
        lambda: hashlib.sha256(canonical_json(canonical_contents).encode()).hexdigest()
    )
    if aggregate != value.get("installed_contents_sha256"):
        failures.append(f"F5 environment installed contents digest differs: {name}")
    metadata = value.get("installed_metadata")
    probe(
        lambda: _exact_mapping(
            metadata, METADATA_KEYS, f"installed_numeric_packages.{name}.metadata"
        )
    )
    if not isinstance(metadata, dict):
        return list(dict.fromkeys(failures))
    for metadata_name, item in metadata.items():
        path = f"installed_numeric_packages.{name}.{metadata_name}"
        probe(lambda: _exact_mapping(item, METADATA_RECEIPT_KEYS, path))
        if not isinstance(item, dict):
            continue
        for key in ("raw_sha256", "canonical_sha256"):
            probe(lambda: _sha(item.get(key), f"{path}.{key}"))
        raw = probe(lambda: base64.b64decode(item.get("raw_base64"), validate=True))
        if raw is None:
            failures.append(f"F5 environment raw metadata is malformed: {name}: {metadata_name}")
            continue
        if hashlib.sha256(raw).hexdigest() != item.get("raw_sha256"):
            failures.append(f"F5 environment raw metadata hash differs: {name}: {metadata_name}")
        canonical = probe(
            lambda: _canonical_metadata_bytes(
                raw, value.get("installation_root"), metadata_name, contents, wheel
            )
        )
        if canonical is not None and hashlib.sha256(canonical).hexdigest() != item.get(
            "canonical_sha256"
        ):
            failures.append(
                f"F5 environment canonical metadata hash differs: {name}: {metadata_name}"
            )
    return list(dict.fromkeys(failures))


def _validate_package(name: str, value: Any, wheel: Mapping[str, str] | None = None) -> None:
    """Refuse package evidence only after inspecting every independent item."""
    failures = collect_package_failures(name, value, wheel)
    if failures:
        raise F5EnvironmentRefusal("; ".join(failures))


def _validate_environment_first(receipt: Mapping[str, Any]) -> None:
    """Refuse every receipt whose fully nested schema cannot be classified."""
    validate_schema_artifact()
    top = _exact_mapping(receipt, REQUIRED_TOP_LEVEL, "receipt")
    classify_receipt(top)
    if top["capture_status"] != "SUCCESS":
        raise F5EnvironmentRefusal("F5 environment capture status is not SUCCESS")
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
    _text(top["gpu_driver_query_raw"], "gpu_driver_query_raw")
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
    # Keep source comparisons after the independent typed/build validation so
    # diagnostic collection can return both a structural failure and every
    # Torch/SMI disagreement from the same receipt.
    source_failures = _source_consistency_failures(top)
    if source_failures:
        raise F5EnvironmentRefusal(source_failures[0])


def validate_environment(receipt: Mapping[str, Any]) -> None:
    """Refuse only after collecting every independently observable defect."""
    failures = collect_environment_failures(receipt)
    if failures:
        raise F5EnvironmentRefusal("; ".join(failures))


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


def _differing_canonical_keys(expected: Any, actual: Any, path: str = "") -> list[str]:
    """List differing canonical keys as sorted JSON pointers; lists are exact values."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        differences: list[str] = []
        for key in sorted(expected.keys() | actual.keys()):
            child = path + "/" + key.replace("~", "~0").replace("/", "~1")
            if key not in expected or key not in actual:
                differences.append(child)
            else:
                differences.extend(_differing_canonical_keys(expected[key], actual[key], child))
        return differences
    return [path] if expected != actual else []


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
        # Retain this exact observation rather than recapturing a changed runtime.
        # Mark the raw receipt refused so even direct reference consumers reject it.
        raise F5CrossAssignmentRefusal(
            {
                "schema": "bl-residual-f5-cross-assignment-refusal-diagnostic/v1",
                "kind": "OUTCOME_BLIND_NON_ADMISSIBLE_DIAGNOSTIC",
                "mode": mode,
                "admissible": False,
                "scientific_use_authorized": False,
                "fit_usable": False,
                "expected_raw_sha256": expected_raw_sha,
                "actual_raw_sha256": actual_raw_sha,
                "expected_canonical_sha256": receipt_sha256(expected_canonical),
                "actual_canonical_sha256": receipt_sha256(actual_canonical),
                "differing_canonical_keys": _differing_canonical_keys(
                    expected_canonical, actual_canonical
                ),
                "raw_receipt": refuse_environment_receipt(
                    actual, ["F5 cross-assignment invariant environment differs"]
                ),
            }
        )
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


def _refuse_symlink_components(path: Path, root: Path) -> Path:
    """Return an absolute root-contained path after rejecting every symlink component."""
    root_absolute = Path(os.path.abspath(root))
    absolute = Path(os.path.abspath(path))
    try:
        relative = absolute.relative_to(root_absolute)
    except ValueError as exc:
        raise F5EnvironmentRefusal("F5 installed path escapes the installation root") from exc
    if root_absolute.is_symlink():
        raise F5EnvironmentRefusal("F5 installation root is a symlink")
    cursor = root_absolute
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise F5EnvironmentRefusal("F5 installed path contains a symlink")
    return absolute


def _root_bound_content_key(
    item: object, path: Path, site_root: Path, root: Path
) -> tuple[str, str]:
    """Classify a RECORD path installed outside site-packages without escapes.

    Wheels sometimes install generated scripts under ``bin`` and ordinary
    package bytes under ``share`` or ``include``.  Those bytes are evidence
    just like Python modules, but their RECORD spelling climbs from
    site-packages to the virtualenv root.  Accept only the exact three-parent
    spelling and the reviewed root classes; resolve neither a symlink nor a
    foreign prefix into the receipt namespace.
    """
    record_path = PurePosixPath(str(item))
    if record_path.is_absolute() or record_path.parts[:3] != ("..", "..", ".."):
        raise F5EnvironmentRefusal("F5 external package RECORD path is not root-bound")
    tail = record_path.parts[3:]
    if (
        len(tail) < 2
        or any(part in {"", ".", ".."} for part in tail)
        or tail[0] not in ROOT_BOUND_FILE_CLASSES
    ):
        raise F5EnvironmentRefusal("F5 external package file has an unknown root-bound class")
    # The textual RECORD path and the installed object must name precisely the
    # same child of the venv root.  Check each component before reading bytes
    # so a symlink in an ancestor cannot carry the capture outside the root.
    root_absolute = Path(os.path.abspath(root))
    expected = root_absolute.joinpath(*tail)
    try:
        if _refuse_symlink_components(path, root_absolute) != expected or expected.relative_to(
            root_absolute
        ) != Path(*tail):
            raise ValueError
    except ValueError as exc:
        raise F5EnvironmentRefusal(
            "F5 external package file escapes the installation root"
        ) from exc
    # ``site_root`` is deliberately an argument: the exact three-parent RECORD
    # relation is part of the root-bound representation, rather than inferred
    # from an arbitrary absolute path.
    try:
        if Path(os.path.abspath(site_root.parents[2])) != root_absolute:
            raise ValueError
    except (IndexError, ValueError) as exc:
        raise F5EnvironmentRefusal(
            "F5 package site root has unexpected installation geometry"
        ) from exc
    return "__root_bound__/" + "/".join(tail), tail[0]


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
                script_path = PurePosixPath(row[0])
                if (
                    script_path.parts[:4] != ("..", "..", "..", "bin")
                    or len(script_path.parts) < 5
                    or any(part in {"", ".", ".."} for part in script_path.parts[3:])
                ):
                    raise F5EnvironmentRefusal("F5 RECORD root-bound bin path is unsafe")
                item = contents.get("__root_bound__/" + "/".join(script_path.parts[3:]))
                if item is None:
                    raise F5EnvironmentRefusal(
                        "F5 RECORD root-bound bin row lacks installed content evidence"
                    )
                if "raw_base64" not in item:
                    writer.writerow(row)
                    continue
                raw_script = base64.b64decode(item["raw_base64"], validate=True)
                canonical_script = _canonical_root_bound_script(raw_script, installation_root)
                if canonical_script is None:
                    raise F5EnvironmentRefusal("F5 RECORD entrypoint shebang differs")
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
    # Anchor geometry at the direct dist-info METADATA row.  A nested
    # ``licenses/`` entry can legally sort first in RECORD, but does not locate
    # site-packages.
    dist_info = next(
        (
            item
            for item in distribution.files or ()
            if item.name == "METADATA"
            and len(PurePosixPath(str(item)).parts) == 2
            and PurePosixPath(str(item)).parts[0].endswith(".dist-info")
        ),
        None,
    )
    if dist_info is None:
        raise F5EnvironmentRefusal(f"F5 package lacks dist-info evidence: {name}")
    own_dist_info = PurePosixPath(str(dist_info)).parts[0]
    dist_info_path = Path(distribution.locate_file(dist_info))
    site_root = dist_info_path.parents[1]
    root_path = site_root.parents[2]
    _refuse_symlink_components(dist_info_path, root_path)
    root = str(Path(os.path.abspath(root_path)))
    metadata_raw: dict[str, bytes] = {}
    for metadata_name in sorted(METADATA_KEYS):
        item = next(
            (
                candidate
                for candidate in distribution.files or ()
                if candidate.name == metadata_name
                and PurePosixPath(str(candidate)).parts == (own_dist_info, metadata_name)
            ),
            None,
        )
        if item is None:
            raw = b"<ABSENT>"
        else:
            metadata_path = Path(distribution.locate_file(item))
            _refuse_symlink_components(metadata_path, Path(root))
            raw = metadata_path.read_bytes()
        metadata_raw[metadata_name] = raw
    contents: dict[str, dict[str, str]] = {}
    for item in distribution.files or ():
        path = Path(distribution.locate_file(item))
        _refuse_symlink_components(path, Path(root))
        # dist-info paths are separately retained as raw metadata.  Do not
        # silently normalize them through this installed-library manifest.
        if not path.is_file():
            raise F5EnvironmentRefusal(f"F5 package has a missing installed file: {name}")
        if (
            PurePosixPath(str(item)).parts == (own_dist_info, item.name)
            and item.name in METADATA_KEYS
        ):
            # These four bytes are retained separately with raw evidence and
            # the narrowly permitted canonical mapping where applicable.
            continue
        if path.is_file():
            if path.suffix == ".pyc" or "__pycache__" in path.parts:
                raise F5EnvironmentRefusal(
                    "F5 locked environment must use pip --no-compile; pyc is root-dependent"
                )
            try:
                raw = path.read_bytes()
                canonical = raw
                record_path = PurePosixPath(str(item))
                if record_path.is_absolute():
                    raise F5EnvironmentRefusal("F5 package RECORD path is absolute")
                try:
                    relative = Path(os.path.abspath(path)).relative_to(
                        Path(os.path.abspath(site_root))
                    )
                except ValueError:
                    key, root_bound_kind = _root_bound_content_key(
                        item, path, site_root, Path(root)
                    )
                    if root_bound_kind == "bin":
                        canonical = _canonical_root_bound_script(raw, root) or raw
                else:
                    if record_path != PurePosixPath(relative.as_posix()):
                        raise F5EnvironmentRefusal(
                            "F5 package RECORD path is not site-root relative"
                        )
                    if ".dist-info/" in str(record_path):
                        key = "__dist_info__/" + str(record_path)
                    else:
                        key = str(relative)
            except ValueError:
                raise F5EnvironmentRefusal(
                    f"F5 package has unsafe installed path: {name}"
                ) from None
            if key in contents:
                raise F5EnvironmentRefusal(f"F5 package records duplicate installed path: {name}")
            contents[key] = {
                "raw_sha256": hashlib.sha256(raw).hexdigest(),
                "canonical_sha256": hashlib.sha256(canonical).hexdigest(),
            }
            if _root_bound_content_kind(key) == "bin" and raw != canonical:
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


def _driver_query_raw() -> str:
    """Capture raw SMI stdout before parsing it into the strict receipt form."""
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,compute_cap,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=15,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise F5EnvironmentRefusal("F5 cannot capture nvidia-smi identity") from exc


def _parse_driver_query(raw: str) -> dict[str, str]:
    """Parse one previously retained SMI stdout value without rereading it."""
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 5 or not all(parts):
        raise F5EnvironmentRefusal("F5 nvidia-smi query is malformed")
    return dict(
        zip(
            ("name", "uuid", "driver_version", "compute_cap", "memory_total"),
            parts,
            strict=True,
        )
    )


def _driver_query() -> tuple[dict[str, str], str]:
    """Capture then strictly parse SMI identity for admissible receipts."""
    raw = _driver_query_raw()
    return _parse_driver_query(raw), raw


def _capture_environment_raw() -> dict[str, Any]:
    """Observe each source once, retaining successful fields after any refusal.

    Acquisition errors stay in the raw receipt as well as its diagnostic envelope.
    Refused observations retain their bytes but cannot become reference evidence.
    No failed read suppresses independent reads.
    """
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "capture_status": "SUCCESS",
        "memory_units": dict(MEMORY_UNITS),
    }
    errors: list[str] = []
    acquisition_evidence: dict[str, Any] = {}

    def observe(name: str, operation: Any) -> Any:
        try:
            return operation()
        except Exception as exc:
            errors.append(f"F5 environment acquisition {name}: {type(exc).__name__}: {exc}")
            cause = exc.__cause__ or exc
            evidence = {"exception_type": type(cause).__name__, "message": str(cause)}
            for key in ("stdout", "stderr", "output"):
                value = getattr(cause, key, None)
                if isinstance(value, bytes):
                    evidence[key + "_base64"] = base64.b64encode(value).decode("ascii")
                elif isinstance(value, str):
                    evidence[key] = value
            acquisition_evidence[name] = evidence
            if name == "gpu_driver_query_raw":
                stdout = getattr(cause, "stdout", None) or getattr(cause, "output", None)
                return stdout if isinstance(stdout, str) else None
            return None

    def field(name: str, operation: Any) -> Any:
        receipt[name] = observe(name, operation)
        return receipt[name]

    field("schema_sha256", schema_sha256)
    field("platform", platform.platform)
    field("python", lambda: sys.version)
    field("cpu_count", lambda: os.cpu_count() or 0)
    field("torch", lambda: str(torch.__version__))
    field("cuda", lambda: torch.version.cuda or "none")
    field("python_build", lambda: list(platform.python_build()))
    field("python_compiler", platform.python_compiler)
    field("python_cache_tag", lambda: sys.implementation.cache_tag)
    field("allocated_cpu_affinity", lambda: sorted(os.sched_getaffinity(0)))

    def cpu_isa() -> dict[str, str]:
        fields: dict[str, str] = {}
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            key, separator, value = line.partition(":")
            if separator and key.strip() in {"model name", "flags", "Features", "Hardware"}:
                fields.setdefault(key.strip(), value.strip())
        if not fields:
            raise F5EnvironmentRefusal("F5 capture requires CPU model/ISA evidence")
        return fields

    cpu_fields = field("cpu_isa", cpu_isa)
    field("cpu", lambda: platform.processor() or (cpu_fields or {}).get("model name", "unknown"))

    def kernel() -> dict[str, str]:
        uname = platform.uname()._asdict()
        return {
            key: uname[key] or ("unknown" if key == "processor" else "") for key in OS_KERNEL_KEYS
        }

    field("os_kernel", kernel)
    available = observe("CUDA availability", torch.cuda.is_available)
    count = observe("CUDA device count", torch.cuda.device_count)
    if available is not True or count != 1:
        errors.append("F5 capture requires exactly one CUDA device")
    device = observe("CUDA properties", lambda: torch.cuda.get_device_properties(0))
    field("gpu", lambda: device.name)
    field("gpu_uuid", lambda: str(device.uuid))
    field("gpu_memory", lambda: int(device.total_memory))
    field("gpu_capability", lambda: list(torch.cuda.get_device_capability(0)))
    # Store stdout before parsing; malformed CSV still survives in diagnostics.
    raw_smi = field("gpu_driver_query_raw", _driver_query_raw)
    field("gpu_driver_query", lambda: _parse_driver_query(raw_smi))
    field("cudnn", lambda: str(torch.backends.cudnn.version()))
    field("torch_git_version", lambda: torch.version.git_version or "unknown")
    field("torch_build", torch.__config__.show)
    field("numpy", lambda: np.__version__)

    def numpy_build() -> str:
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            np.__config__.show()
        return stream.getvalue()

    field("numpy_blas_lapack_build", numpy_build)
    distributions = observe(
        "distribution inventory", lambda: list(importlib.metadata.distributions())
    )
    installed: list[tuple[str, str]] = []
    names: set[str] = set()
    for index, dist in enumerate(distributions or []):
        identity = observe(
            f"distribution[{index}]",
            lambda: (str(dist.metadata.get("Name", "")).lower(), dist.version),
        )
        if identity is not None:
            installed.append(identity)
            if identity[0]:
                names.add(identity[0].replace("_", "-"))
    field(
        "installed_versions_sha256",
        lambda: hashlib.sha256(canonical_json(sorted(installed)).encode()).hexdigest(),
    )
    environment_dir = Path(__file__).with_name("f5_adoption_environment_20261026_v6")
    for key, filename, expected in (
        (
            "environment_lock_sha256",
            "requirements-f5-l4-cu126.lock",
            "2577ed9dd9fe34468f5562262def7bccb352c9c0bdce06b293bb4b12a0ea5301",
        ),
        (
            "environment_descriptor_sha256",
            "f5-common-environment-v6.json",
            "bc05167c188680478663235f2b3dcfb70d7959e9d392071384409ef958b0f0f5",
        ),
    ):
        observed = field(key, lambda: _file_sha(environment_dir / filename))
        if observed != expected:
            errors.append(f"F5 environment acquisition {key}: pinned artifact differs")
    wheels = field("locked_wheels", _wheel_manifest)
    wheel_map = wheels if isinstance(wheels, dict) else {}
    packages: dict[str, Any] = {}
    for name in sorted(names):
        packages[name] = observe(
            f"package {name}", lambda: _package_receipt(name, wheel_map.get(name))
        )
    receipt["installed_packages"] = packages
    receipt["installed_numeric_packages"] = {
        name: packages.get(name) for name in sorted(NUMERIC_PACKAGES)
    }
    receipt["torch_threads"] = {
        "intra": observe("torch_threads.intra", torch.get_num_threads),
        "inter": observe("torch_threads.inter", torch.get_num_interop_threads),
    }
    controls = {
        "deterministic": torch.are_deterministic_algorithms_enabled,
        "cudnn_deterministic": lambda: torch.backends.cudnn.deterministic,
        "cudnn_benchmark": lambda: torch.backends.cudnn.benchmark,
        "matmul_tf32": lambda: torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": lambda: torch.backends.cudnn.allow_tf32,
        "default_dtype": lambda: str(torch.get_default_dtype()),
        "amp": lambda: torch.is_autocast_enabled("cuda"),
        "bf16": lambda: torch.is_autocast_enabled("cuda")
        and torch.get_autocast_dtype("cuda") == torch.bfloat16,
    }
    receipt["numeric_controls"] = {
        name: observe(f"numeric_controls.{name}", read) for name, read in controls.items()
    }
    receipt["thread_env"] = {name: os.environ.get(name, "") for name in sorted(THREAD_ENV_KEYS)}
    if errors:
        receipt["capture_status"] = "REFUSED"
        receipt["_acquisition_failures"] = errors
        receipt["_acquisition_evidence"] = acquisition_evidence
    return receipt


def refuse_environment_receipt(receipt: Mapping[str, Any], failures: list[str]) -> dict[str, Any]:
    """Retain refused observations with a status no evidence consumer can admit."""
    return {
        **receipt,
        "capture_status": "REFUSED",
        "_admission_failures": list(
            dict.fromkeys([*receipt.get("_admission_failures", []), *failures])
        ),
    }


def capture_environment_diagnostic() -> dict[str, Any]:
    """Return all available raw observations and failures without admitting them."""
    try:
        receipt = dict(_capture_environment_raw())
    except Exception as exc:
        # Defensive last resort for an unexpected collector-level programming
        # error; ordinary source failures are retained field by field above.
        return {
            "raw_receipt": refuse_environment_receipt({}, [str(exc)]),
            "acquisition_failures": [str(exc)],
            "validation_failures": [],
            "admissible": False,
        }
    acquisition = receipt.get("_acquisition_failures", [])
    details = receipt.get("_acquisition_evidence", {})
    failures = collect_environment_failures(receipt)
    if acquisition or failures:
        receipt = refuse_environment_receipt(receipt, [*acquisition, *failures])
    result = {
        "raw_receipt": receipt,
        "acquisition_failures": acquisition,
        "validation_failures": failures,
        "admissible": not acquisition and not failures,
    }
    if details:
        result["acquisition_evidence"] = details
    return result


def capture_environment() -> dict[str, Any]:
    """Capture only an admissible receipt; diagnostics retain refusal evidence."""
    diagnostic = capture_environment_diagnostic()
    if not diagnostic["admissible"]:
        failures = [
            *diagnostic["acquisition_failures"],
            *diagnostic["validation_failures"],
        ]
        raise F5EnvironmentRefusal("; ".join(failures))
    receipt = diagnostic["raw_receipt"]
    if not isinstance(receipt, dict):  # Defensive, for the typed public surface.
        raise F5EnvironmentRefusal("F5 environment diagnostic receipt is malformed")
    return receipt
