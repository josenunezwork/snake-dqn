"""Prospective kernel provenance rules; no CUDA execution is claimed here."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from research.bl_residual import f5_environment as env
from research.bl_residual import f5_environment_reference as reference
from research.bl_residual import f5_environment_v5 as frozen_v5
from research.bl_residual import f5_job
from tests.test_bl_residual_f5_environment import _receipt

TUPLES = (
    ("6.6.122+", "#1 SMP Thu Apr 30 18:17:14 UTC 2026"),
    ("6.6.159+", "#1 SMP Thu Oct 22 03:07:54 UTC 2026"),
)


def native(index: int = 0) -> dict[str, Any]:
    """Build independent synthetic native receipts with the real diagnostic tuples."""
    receipt = _receipt()
    release, version = TUPLES[index]
    receipt["schema"] = env.SCHEMA
    receipt["schema_sha256"] = env.schema_sha256()
    receipt["os_kernel"].update(release=release, version=version)
    receipt["libc"] = {"implementation": "glibc", "version": "2.39"}
    receipt["platform"] = f"Linux-{release}-x86_64-with-glibc2.39"
    return receipt


def test_exact_diagnostic_tuples_differ_v5_equal_only_cross_assignment_v6() -> None:
    first, second = native(0), native(1)
    originals = copy.deepcopy((first, second))
    report = env.compare_environment(first, second)
    assert report["expected_raw_sha256"] != report["actual_raw_sha256"]
    assert report["expected_canonical_sha256"] == report["actual_canonical_sha256"]
    assert (first, second) == originals
    old = []
    for receipt in originals:
        value = copy.deepcopy(receipt)
        value.pop("libc")
        value["schema"] = frozen_v5.SCHEMA
        value["schema_sha256"] = frozen_v5.schema_sha256()
        old.append(value)
    with pytest.raises(frozen_v5.F5CrossAssignmentRefusal) as raised:
        frozen_v5.compare_environment(*old)
    assert raised.value.diagnostic["differing_canonical_keys"] == [
        "/os_kernel/release",
        "/os_kernel/version",
        "/platform",
    ]
    with pytest.raises(env.F5EnvironmentRefusal, match="same-assignment"):
        env.compare_environment(first, second, mode="same-assignment")


@pytest.mark.parametrize("component", ["release", "version", "platform"])
def test_each_individual_raw_kernel_component_refuses_within_assignment(component: str) -> None:
    first, changed = native(), native()
    if component == "platform":
        changed[component] = native(1)[component]
    else:
        changed["os_kernel"][component] = native(1)["os_kernel"][component]
    with pytest.raises(env.F5EnvironmentRefusal):
        env.compare_environment(first, changed, mode="same-assignment")


@pytest.mark.parametrize(
    "platform",
    [
        "",
        "Linux-6.6.122+-x86_64",
        "Linux-6.6.122+-x86_64-with-glibc2.39-extra",
        "Linux-6.6.122+-6.6.122+-x86_64-with-glibc2.39",
        "Linux-6.6.159+-x86_64-with-glibc2.39",
        "Linux-6.6.122+-aarch64-with-glibc2.39",
        "Linux-6.6.122+-x86_64-with-glibc2.40",
        "Darwin-6.6.122+-x86_64-with-glibc2.39",
        "Linux--x86_64-with-glibc2.39",
        " Linux-6.6.122+-x86_64-with-glibc2.39",
    ],
)
def test_full_platform_structure_refuses_inconsistency_and_ambiguity(platform: str) -> None:
    changed = native()
    changed["platform"] = platform
    with pytest.raises(env.F5EnvironmentRefusal):
        env.compare_environment(native(), changed)


def test_consistent_but_unreviewed_kernel_refuses() -> None:
    changed = native()
    changed["os_kernel"].update(release="6.6.160+", version="#1 unreviewed")
    changed["platform"] = "Linux-6.6.160+-x86_64-with-glibc2.39"
    with pytest.raises(env.F5EnvironmentRefusal) as raised:
        env.compare_environment(native(), changed)
    assert "6.6.160+" in str(raised.value)
    assert "#1 unreviewed" in str(raised.value)
    assert changed["platform"] in str(raised.value)


@pytest.mark.parametrize("version", [b"2.40", b"", b"2.39-extra"])
def test_libc_acquisition_requires_independent_sources_to_agree(
    monkeypatch: pytest.MonkeyPatch, version: bytes
) -> None:
    def getter() -> bytes:
        return version

    monkeypatch.setattr(env.os, "confstr", lambda key: "glibc 2.39")
    monkeypatch.setattr(
        env.ctypes, "CDLL", lambda name: SimpleNamespace(gnu_get_libc_version=getter)
    )
    with pytest.raises(env.F5EnvironmentRefusal):
        env._running_libc_identity()


def test_every_classified_invariant_leaf_survives_kernel_projection() -> None:
    first = native()
    checked = []
    for encoded, classification in env.classify_receipt(first).items():
        if classification != "invariant":
            continue
        path = json.loads(encoded)
        changed = native(1)
        parent: Any = changed
        for key in path[:-1]:
            parent = parent[key]
        value = parent[path[-1]]
        if isinstance(value, bool):
            parent[path[-1]] = not value
        elif isinstance(value, int):
            parent[path[-1]] = value + 1
        elif isinstance(value, list):
            parent[path[-1]] = [*value, "unexpected"]
        elif isinstance(value, str):
            parent[path[-1]] = value + "-drift"
        else:
            parent[path[-1]] = "unexpected"
        with pytest.raises(env.F5EnvironmentRefusal):
            env.compare_environment(first, changed)
        checked.append(path)
    assert len(checked) >= 40
    assert ["gpu_driver_query", "driver_version"] in checked
    assert ["cpu_isa", "model name"] in checked
    assert ["libc", "version"] in checked
    assert ["numeric_controls", "matmul_tf32"] in checked


@pytest.mark.parametrize("mutation", ["extra", "missing", "extra-nested"])
def test_unknown_missing_and_extra_fields_refuse(mutation: str) -> None:
    changed = native(1)
    if mutation == "extra":
        changed["unexpected"] = "value"
    elif mutation == "missing":
        del changed["os_kernel"]["version"]
    else:
        changed["libc"]["unexpected"] = "value"
    with pytest.raises(env.F5EnvironmentRefusal):
        env.compare_environment(native(), changed)


@pytest.fixture(scope="module")
def capture05() -> dict[str, Any]:
    if not reference.CAPTURE05_PATH.is_file():
        pytest.skip("immutable local capture-05 evidence is not installed")
    return reference.capture05_reference()


def test_pinned_comparison_wrapper_is_self_contained_and_never_native(
    capture05: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_file_read(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("runtime adapter cannot reopen a Mac-only reference path")

    monkeypatch.setattr(reference, "_read_capture05", no_file_read)
    reference.validate_capture05_reference(capture05)
    with pytest.raises(env.F5EnvironmentRefusal):
        env.validate_environment(capture05)
    projected = reference._v6_projection(capture05)
    projected["os_kernel"].update(release=TUPLES[1][0], version=TUPLES[1][1])
    projected["platform"] = "Linux-6.6.159+-x86_64-with-glibc2.39"
    result = reference.compare_capture05_reference(capture05, projected)
    assert result["reference_kind"] == "PINNED_V5_SUCCESS_COMPARISON_ONLY"
    with pytest.raises(env.F5EnvironmentRefusal):
        env.compare_environment(capture05, projected, mode="same-assignment")


@pytest.mark.parametrize("component", ["path", "sha256", "size", "raw", "kind"])
def test_pinned_reference_tampering_refuses(capture05: dict[str, Any], component: str) -> None:
    changed = copy.deepcopy(capture05)
    if component == "raw":
        changed["raw_v5_base64"] = changed["raw_v5_base64"][:-4] + "AAAA"
    elif component == "kind":
        changed["kind"] = "NATIVE_SUCCESS"
    elif component == "size":
        changed["source"][component] += 1
    else:
        changed["source"][component] += "-tampered"
    with pytest.raises(ValueError):
        reference.validate_capture05_reference(changed)


def test_arbitrary_v5_and_nested_refused_objects_cannot_be_reference() -> None:
    old = native()
    old.pop("libc")
    old.update(schema=frozen_v5.SCHEMA, schema_sha256=frozen_v5.schema_sha256())
    for value in (old, {"raw_receipt": old}, {**old, "capture_status": "REFUSED"}):
        with pytest.raises(ValueError):
            reference.validate_capture05_reference(value)
        with pytest.raises(env.F5EnvironmentRefusal):
            env.validate_environment(value)


def test_actual_canary02_diagnostic_stays_non_admissible(capture05: dict[str, Any]) -> None:
    path = Path(
        "/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/"
        "f5-timing-recovery-offline-20261026/canary-02/output-02/exports/"
        "environment/cross-assignment-refusal.json"
    )
    if not path.is_file():
        pytest.skip("immutable canary-02 diagnostic is not installed")
    raw_bytes = path.read_bytes()
    assert hashlib.sha256(raw_bytes).hexdigest() == (
        "ae157f89cce650d73bab192acb324987f97a50cc29fed44ee83bc8a09faf48b9"
    )
    diagnostic = json.loads(raw_bytes)
    assert diagnostic["differing_canonical_keys"] == [
        "/os_kernel/release",
        "/os_kernel/version",
        "/platform",
    ]
    actual = diagnostic["raw_receipt"]
    assert (actual["os_kernel"]["release"], actual["os_kernel"]["version"]) == TUPLES[1]
    assert actual["platform"] == "Linux-6.6.159+-x86_64-with-glibc2.39"
    assert actual["capture_status"] == "REFUSED"
    for key in ("admissible", "fit_usable", "scientific_use_authorized"):
        assert diagnostic[key] is False
    for refused in (diagnostic, actual):
        with pytest.raises(ValueError):
            reference.validate_capture05_reference(refused)
        with pytest.raises(ValueError):
            f5_job._environment_evidence(
                {"environment_reference": capture05, "environment_policy": env.policy_binding()},
                refused,
            )
