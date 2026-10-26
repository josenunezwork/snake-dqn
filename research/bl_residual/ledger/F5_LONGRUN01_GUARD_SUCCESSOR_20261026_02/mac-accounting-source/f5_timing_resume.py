"""Path-bound immutable completed-unit recovery for F5 timing."""

from __future__ import annotations

import json
import math
import platform
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from research.bl_ppo.exports import export_json, verify_export

SCHEMA = "bl-f5-timing-resume-ledger/v2"
UNIT_KIND = "f5-timing-unit"
CALIBRATION_KEY = "coupled-cw-calibration"


class ResumeRefusal(ValueError):
    pass


def _bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()


def identity(
    *,
    namespace: str,
    allocation: str,
    source_sha256: str,
    environment_sha256: str,
    input_sha256: str,
) -> dict[str, str]:
    answer = {
        "namespace": namespace,
        "allocation": allocation,
        "source_sha256": source_sha256,
        "environment_sha256": environment_sha256,
        "input_sha256": input_sha256,
    }
    if not namespace or any(
        not isinstance(v, str) or re.fullmatch(r"[0-9a-f]{64}", v) is None
        for k, v in answer.items()
        if k != "namespace"
    ):
        raise ResumeRefusal("malformed timing unit identity")
    return answer


def unit_id(binding: Mapping[str, str], key: int | str) -> str:
    return f"{binding['namespace']}/{binding['allocation']}/{UNIT_KIND}/{key}"


def _keys() -> set[int | str]:
    return {*range(672, 684), CALIBRATION_KEY}


def _safe_relative(value: object) -> Path:
    if not isinstance(value, str):
        raise ResumeRefusal("completed unit path is malformed")
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ResumeRefusal("completed unit path escapes ledger root")
    return path


def _safe_file(path: Path) -> Path:
    """Refuse symlinks anywhere from ledger root through the named export."""
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise ResumeRefusal("timing ledger path has symlink ancestry")
        if parent.parent == parent:
            break
    return path


def empty_ledger(binding: Mapping[str, str]) -> dict[str, Any]:
    return {"schema": SCHEMA, "unit_kind": UNIT_KIND, "identity": dict(binding), "units": []}


def _validate_payload(
    key: int | str, payload: Mapping[str, Any], *, canary_only: bool = False
) -> None:
    elapsed = payload.get("elapsed_seconds")
    if (
        not isinstance(elapsed, (int, float))
        or isinstance(elapsed, bool)
        or not math.isfinite(elapsed)
        or elapsed <= 0
    ):
        raise ResumeRefusal("completed unit lacks a full uninterrupted duration")
    if canary_only:
        if payload.get("canary_only") is not True or payload.get("allocation_index") != key:
            raise ResumeRefusal("CANARY_ONLY unit classification differs")
        return
    if key != CALIBRATION_KEY:
        required = {
            "allocation_index",
            "elapsed_seconds",
            "parent_id",
            "group",
            "group_index",
            "timing_only",
            "parity",
            "q_shape",
            "full_h300",
            "long_body",
            "full_h5000_natural",
            "watch_fixture_frames",
        }
        if (
            not required <= set(payload)
            or payload.get("allocation_index") != key
            or payload.get("parity") != "PASS"
            or payload.get("q_shape") != [1, 6]
            or payload.get("timing_only") is not True
            or payload.get("canary_only") is True
            or any(
                type(payload.get(flag)) is not bool
                for flag in ("full_h300", "long_body", "full_h5000_natural")
            )
        ):
            raise ResumeRefusal("fixture completed-unit payload differs")
        # The completed-unit scanner is the skip authority.  It must reject a
        # record whose outer unit key is intact but whose source allocation
        # metadata was altered before a later reducer gets a chance to inspect
        # it.  Keep this tied to the frozen source allocation, rather than a
        # caller-provided fixture list.
        from research.bl_residual.f5_r10a_annex import allocate

        canonical = next(
            (row for row in allocate(0) if row.get("allocation_index") == key),
            None,
        )
        if canonical is None or any(
            payload.get(name) != canonical.get(name)
            for name in (
                "allocation_index",
                "parent_id",
                "group",
                "group_index",
                "seed",
                "timing_only",
            )
        ):
            raise ResumeRefusal("fixture allocation identity differs from frozen source allocation")
        # A completed measurement may expose failed coverage. Preserve that
        # result; the Mac reducer applies the registered per-stream rule and
        # never authorizes a completion retry for an unfavorable observation.
        return
    arms = payload.get("arms")
    if not isinstance(arms, Mapping) or not all(
        isinstance(arms.get(arm), Mapping)
        and arms[arm].get("serialized_continuation_exact") is True
        and isinstance(arms[arm].get("full_seconds"), (int, float))
        and isinstance(arms[arm].get("serialized_continuation_seconds"), (int, float))
        and all(
            not isinstance(arms[arm][field], bool)
            and math.isfinite(arms[arm][field])
            and arms[arm][field] > 0
            for field in ("full_seconds", "serialized_continuation_seconds")
        )
        for arm in ("C", "W")
    ):
        raise ResumeRefusal("coupled calibration record lacks both exact continuation checks")


def _verify_unit_environment(
    output: Path, key: int | str, payload: Mapping[str, Any]
) -> dict[str, Any]:
    """Reopen raw native attestations before publishing or admitting a skip."""
    from research.bl_residual import f5_environment

    captures = []
    for name in ("environment_before", "environment_after"):
        ref = payload.get(name)
        if (
            not isinstance(ref, Mapping)
            or set(ref) != {"path", "size", "sha256"}
            or type(ref.get("size")) is not int
            or ref["size"] <= 0
            or not isinstance(ref.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", ref["sha256"]) is None
        ):
            raise ResumeRefusal("timing unit lacks a typed raw environment link")
        relative = _safe_relative(ref["path"])
        after = "after-calibration.json" if key == CALIBRATION_KEY else f"after-fixture-{key}.json"
        expected_path = "environment/" + ("before.json" if name == "environment_before" else after)
        if relative.as_posix() != expected_path:
            raise ResumeRefusal("timing unit raw environment link has foreign scope")
        path = _safe_file(output / relative)
        try:
            if verify_export(path) != ref["sha256"] or path.stat().st_size != ref["size"]:
                raise ResumeRefusal("timing unit raw environment link hash or size differs")
            capture = json.loads(path.read_bytes())
            f5_environment.validate_environment(capture)
        except (OSError, ValueError) as exc:
            raise ResumeRefusal(
                "timing unit raw environment attestation is not verified native v6"
            ) from exc
        captures.append(capture)
    before, after = captures
    try:
        f5_environment.compare_environment(before, after, mode="same-assignment")
    except ValueError as exc:
        raise ResumeRefusal("timing unit has within-assignment raw environment drift") from exc
    expected = {
        "assignment_raw_sha256": f5_environment.receipt_sha256(before),
        "os_kernel": dict(before["os_kernel"]),
        "platform": before["platform"],
        "gpu_uuid": before["gpu_uuid"],
        "before": dict(payload["environment_before"]),
        "after": dict(payload["environment_after"]),
    }
    if payload.get("kernel_provenance") != expected:
        raise ResumeRefusal("timing unit kernel UUID or raw hash provenance differs")
    return expected


def _read_unit(
    path: Path, binding: Mapping[str, str], expected: Mapping[str, Any]
) -> tuple[int | str, dict[str, Any]]:
    try:
        _safe_file(path)
        observed_sha = verify_export(path)
        raw = json.loads(path.read_text())
    except (ValueError, json.JSONDecodeError) as exc:
        raise ResumeRefusal("completed timing unit is not a verified immutable export") from exc
    if (
        not isinstance(raw, dict)
        or raw.get("schema") != "bl-f5-timing-unit-record/v2"
        or set(raw) != {"schema", "unit_id", "key", "identity", "payload"}
    ):
        raise ResumeRefusal("malformed completed timing unit")
    key = raw.get("key")
    if (
        key not in _keys()
        or raw.get("identity") != dict(binding)
        or raw.get("unit_id") != unit_id(binding, key)
    ):
        raise ResumeRefusal("foreign timing completed unit")
    if expected.get("sha256") != observed_sha or expected.get("size") != path.stat().st_size:
        raise ResumeRefusal("completed timing unit path hash or size drift")
    payload = raw.get("payload")
    if not isinstance(payload, dict):
        raise ResumeRefusal("completed timing unit payload malformed")
    _validate_payload(key, payload, canary_only=bool(payload.get("canary_only")))
    if path.parent.name not in {"timing-units", "timing-canary"}:
        raise ResumeRefusal("completed timing unit has foreign output scope")
    _verify_unit_environment(path.parent.parent, key, payload)
    return key, payload


def load_ledger(
    path: Path, binding: Mapping[str, str], *, canary_only: bool = False
) -> dict[int | str, dict[str, Any]]:
    """Mac and runtime scanner verifies ledger plus every path/sidecar/marker."""
    try:
        _safe_file(path)
        verify_export(path)
        ledger = json.loads(path.read_text())
    except (ValueError, json.JSONDecodeError) as exc:
        raise ResumeRefusal("timing ledger is not a verified immutable export") from exc
    if (
        not isinstance(ledger, dict)
        or ledger.get("schema") != SCHEMA
        or ledger.get("unit_kind") != UNIT_KIND
        or ledger.get("identity") != dict(binding)
        or not isinstance(ledger.get("units"), list)
    ):
        raise ResumeRefusal("foreign timing ledger")
    result: dict[int | str, dict[str, Any]] = {}
    for item in ledger["units"]:
        if not isinstance(item, dict) or set(item) != {"key", "path", "sha256", "size"}:
            raise ResumeRefusal("malformed timing ledger path entry")
        key, payload = _read_unit(path.parent / _safe_relative(item["path"]), binding, item)
        if bool(payload.get("canary_only")) != canary_only:
            raise ResumeRefusal("ledger mixes CANARY_ONLY and full timing units")
        if key != item["key"] or key in result:
            raise ResumeRefusal("duplicate timing ledger unit")
        result[key] = payload
    return result


def publish_unit(
    output: Path,
    binding: Mapping[str, str],
    key: int | str,
    payload: Mapping[str, Any],
    *,
    canary_only: bool = False,
) -> dict[str, Any]:
    """Publish a complete unit atomically; export_json writes marker last."""
    if key not in _keys():
        raise ResumeRefusal("unknown timing unit")
    copy = dict(payload)
    if canary_only:
        copy["canary_only"] = True
    elif copy.get("canary_only") is True:
        raise ResumeRefusal("CANARY_ONLY record cannot be a timing unit")
    _validate_payload(key, copy, canary_only=canary_only)
    _verify_unit_environment(output, key, copy)
    name = "calibration" if key == CALIBRATION_KEY else f"fixture-{key}"
    folder = "timing-canary" if canary_only else "timing-units"
    path = output / folder / f"{name}.json"
    exported = export_json(
        path,
        {
            "schema": "bl-f5-timing-unit-record/v2",
            "unit_id": unit_id(binding, key),
            "key": key,
            "identity": dict(binding),
            "payload": copy,
        },
    )
    verify_export(path)
    return {
        "key": key,
        "path": path.relative_to(output).as_posix(),
        "sha256": exported["sha256"],
        "size": exported["size"],
    }


def build_ledger(
    output: Path,
    binding: Mapping[str, str],
    entries: Sequence[Mapping[str, Any]],
    *,
    name: str = "verified-ledger.json",
    canary_only: bool = False,
) -> dict[str, Any]:
    ledger = empty_ledger(binding)
    seen: set[int | str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ResumeRefusal("malformed completed-unit entry")
        source = output / _safe_relative(entry.get("path"))
        key, payload = _read_unit(source, binding, entry)
        if bool(payload.get("canary_only")) != canary_only:
            raise ResumeRefusal("ledger mixes CANARY_ONLY and full timing units")
        if key in seen:
            raise ResumeRefusal("duplicate completed-unit entry")
        seen.add(key)
        # Unit locations in a ledger are relative to the ledger's own parent,
        # which may be a nested resume/canary directory.
        ledger_parent = (output / name).parent
        ledger["units"].append(
            {**dict(entry), "path": source.relative_to(ledger_parent).as_posix()}
        )
    export_json(output / name, ledger)
    verify_export(output / name)
    return ledger


ACCOUNTING_SCHEMA = "bl-f5-timing-accounting-history/v1"
PREALLOCATION_REFUSAL_ALIAS_SCHEMA = "bl-f5-preallocation-refusal-alias/v1"
PREALLOCATION_NORMALIZATION_LEAF = "/lease_seconds"
PREALLOCATION_ALLOWED_LEAVES = [
    "/budget/unit_cap",
    "/job_id",
    "/operational_admission/live_account_preflight/path",
    "/operational_admission/live_account_preflight/sha256",
    "/operational_admission/live_account_preflight/size",
    "/output_dir",
    "/supervisor_state_dir",
]


def metadata_reference(path: Path) -> dict[str, Any]:
    """Freeze local operational evidence; never inspect training/science payloads."""
    from research.bl_ppo.exports import digest

    path = _safe_file(Path(path).absolute())
    if not path.is_file():
        raise ResumeRefusal("accounting metadata reference is absent")
    return {"path": str(path), "size": path.stat().st_size, "sha256": digest(path)}


def _bound_metadata(reference: Mapping[str, Any], *, immutable: bool = False) -> dict[str, Any]:
    from research.bl_ppo.exports import digest

    if not isinstance(reference, Mapping) or set(reference) != {"path", "size", "sha256"}:
        raise ResumeRefusal("malformed accounting metadata reference")
    if (
        not isinstance(reference["path"], str)
        or not Path(reference["path"]).is_absolute()
        or type(reference["size"]) is not int
        or reference["size"] <= 0
        or not isinstance(reference["sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", reference["sha256"]) is None
    ):
        raise ResumeRefusal("malformed accounting metadata path size or hash")
    path = _safe_file(Path(reference["path"]))
    try:
        if path.stat().st_size != reference["size"] or digest(path) != reference["sha256"]:
            raise ResumeRefusal("accounting metadata path size or hash drift")
        if immutable:
            verify_export(path)
        result = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ResumeRefusal("accounting metadata is not verified") from exc
    if not isinstance(result, dict):
        raise ResumeRefusal("accounting metadata must be an object")
    return result


_MISSING = object()


def _typed_differences(before: Any, after: Any, prefix: str = "") -> list[str]:
    """Return JSON-pointer leaves whose values differ, preserving type identity."""
    if before is _MISSING:
        if isinstance(after, dict):
            children = [
                leaf
                for key in sorted(after)
                for leaf in _typed_differences(
                    _MISSING, after[key], f"{prefix}/{key.replace('~', '~0').replace('/', '~1')}"
                )
            ]
            return children or [prefix]
        return [prefix]
    if after is _MISSING:
        if isinstance(before, dict):
            children = [
                leaf
                for key in sorted(before)
                for leaf in _typed_differences(
                    before[key], _MISSING, f"{prefix}/{key.replace('~', '~0').replace('/', '~1')}"
                )
            ]
            return children or [prefix]
        return [prefix]
    if type(before) is not type(after):
        return [prefix]
    if isinstance(before, dict):
        return [
            leaf
            for key in sorted(before.keys() | after.keys())
            for leaf in _typed_differences(
                before.get(key, _MISSING),
                after.get(key, _MISSING),
                f"{prefix}/{key.replace('~', '~0').replace('/', '~1')}",
            )
        ]
    if isinstance(before, list):
        if len(before) != len(after):
            return [prefix]
        return [
            leaf
            for index, (old, new) in enumerate(zip(before, after))
            for leaf in _typed_differences(old, new, f"{prefix}/{index}")
        ]
    return [] if before == after else [prefix]


def _control_input_sha256(manifest: Mapping[str, Any]) -> str:
    rows = manifest.get("inputs")
    matches = (
        [row for row in rows if isinstance(row, dict) and row.get("dest") == "control.json"]
        if isinstance(rows, list)
        else []
    )
    if len(matches) != 1 or not isinstance(matches[0].get("sha256"), str):
        raise ResumeRefusal("preallocation alias manifest lacks one bound control input")
    return matches[0]["sha256"]


def _preallocation_refusal_alias(
    reference: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    control: Mapping[str, Any],
    receipt: Mapping[str, Any] | None = None,
    observed_normalized_manifest: Path | None = None,
    attempt_index: int,
) -> dict[str, Any]:
    """Validate the sole Mac accounting alias for a pre-allocation refusal.

    The proof permits a successor broker identity to account for units tagged by
    an unchanged frozen control only after the original attempt demonstrably
    failed before allocation. It is never a runtime, retry, or science grant.
    """
    alias = _bound_metadata(reference, immutable=True)
    required = {
        "schema",
        "attempt_index",
        "logical_attempt_id",
        "broker_attempt_id",
        "predecessor",
        "successor",
        "allowed_leaf_diff",
    }
    if set(alias) != required or alias.get("schema") != PREALLOCATION_REFUSAL_ALIAS_SCHEMA:
        raise ResumeRefusal("preallocation refusal alias schema differs")
    if (
        type(alias.get("attempt_index")) is not int
        or alias.get("attempt_index") != 0
        or type(attempt_index) is not int
        or attempt_index != 0
    ):
        raise ResumeRefusal("preallocation refusal alias is limited to original attempt zero")
    logical, broker = alias.get("logical_attempt_id"), alias.get("broker_attempt_id")
    if (
        not isinstance(logical, str)
        or not logical
        or not isinstance(broker, str)
        or not broker
        or logical == broker
    ):
        raise ResumeRefusal("preallocation refusal alias attempt identities are malformed")
    predecessor, successor = alias.get("predecessor"), alias.get("successor")
    if not isinstance(predecessor, dict) or set(predecessor) != {
        "raw_manifest",
        "refusal_receipt",
        "normalized_manifest",
    }:
        raise ResumeRefusal("preallocation refusal alias predecessor binding differs")
    if not isinstance(successor, dict) or set(successor) != {"raw_manifest", "control"}:
        raise ResumeRefusal("preallocation refusal alias successor binding differs")
    allowed = alias.get("allowed_leaf_diff")
    if allowed != PREALLOCATION_ALLOWED_LEAVES:
        raise ResumeRefusal("preallocation refusal alias allowed leaf diff differs")
    old_raw = _bound_metadata(predecessor["raw_manifest"])
    old_normalized = _bound_metadata(predecessor["normalized_manifest"])
    old_receipt = _bound_metadata(predecessor["refusal_receipt"])
    new_raw = _bound_metadata(successor["raw_manifest"])
    bound_control = _bound_metadata(successor["control"])
    if _typed_differences(manifest, new_raw) or _typed_differences(control, bound_control):
        raise ResumeRefusal("preallocation refusal alias does not bind supplied successor inputs")
    if _typed_differences(old_raw, new_raw) != PREALLOCATION_ALLOWED_LEAVES:
        raise ResumeRefusal("preallocation refusal alias successor differs beyond seven leaves")
    expected_old = deepcopy(old_raw)
    expected_old["lease_seconds"] = 180.0
    if _typed_differences(expected_old, old_normalized) != []:
        raise ResumeRefusal("preallocation refusal alias predecessor normalization differs")
    if (
        old_raw.get("job_id") != logical
        or new_raw.get("job_id") != broker
        or old_receipt.get("job_id") != logical
        or old_receipt.get("manifest_sha256") != predecessor["normalized_manifest"].get("sha256")
        or "endpoint" not in old_receipt
        or "terminal" not in old_receipt
        or old_receipt["endpoint"] is not None
        or old_receipt["terminal"] is not None
        or old_receipt.get("exports") != []
        or old_receipt.get("error") != "Refused('insufficient balance for cap plus study reserve')"
        or "control_operations" not in old_receipt
        or "transfers" not in old_receipt
        or old_receipt["control_operations"] != []
        or old_receipt["transfers"] != []
    ):
        raise ResumeRefusal("preallocation refusal alias does not prove refusal before allocation")
    calls = old_receipt.get("cli_calls")
    if not isinstance(calls, list) or [
        row.get("command") for row in calls if isinstance(row, dict)
    ] != [
        "version",
        "sessions",
        "usage",
    ]:
        raise ResumeRefusal("preallocation refusal alias predecessor made provider operations")
    if (
        _control_input_sha256(old_raw) != successor["control"].get("sha256")
        or _control_input_sha256(new_raw) != successor["control"].get("sha256")
        or old_raw.get("commit") != new_raw.get("commit")
        or old_receipt.get("commit") != old_raw.get("commit")
        or bound_control.get("timing_attempt_id") != logical
        or old_raw.get("budget", {}).get("unit_cap") != 100000.0
        or new_raw.get("budget", {}).get("unit_cap") != 100.0
        or old_raw.get("budget", {}).get("reserve_units") != 0.0
        or new_raw.get("budget", {}).get("reserve_units") != 0.0
    ):
        raise ResumeRefusal("preallocation refusal alias source control or input differs")
    if receipt is not None:
        if receipt.get("job_id") != broker:
            raise ResumeRefusal("preallocation refusal alias broker receipt identity differs")
        if observed_normalized_manifest is None:
            raise ResumeRefusal("preallocation refusal alias needs observed normalized manifest")
        observed = _bound_metadata(metadata_reference(observed_normalized_manifest))
        expected_new = deepcopy(new_raw)
        expected_new["lease_seconds"] = 180.0
        if _typed_differences(expected_new, observed) != [] or (
            receipt.get("manifest_sha256")
            != metadata_reference(observed_normalized_manifest)["sha256"]
        ):
            raise ResumeRefusal("preallocation refusal alias successor normalization differs")
    return {
        "logical_attempt_id": logical,
        "broker_attempt_id": broker,
        "reference": dict(reference),
    }


def append_accounting_history(
    path: Path,
    binding: Mapping[str, str],
    attempt: Mapping[str, Any],
    *,
    environment_reference: Mapping[str, Any],
    previous_history: Mapping[str, Any] | None = None,
    prior_exposure: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create an immutable prefix extension, never rewrite a saved attempt/history.

    The returned reference must be frozen outside this history before reduction;
    a self-authored latest-history claim cannot prove that no attempt was omitted.
    ``prior_exposure`` binds the historical closure as accounting context only.
    """
    previous = None
    attempts: list[dict[str, Any]] = []
    if previous_history is not None:
        previous = verify_accounting_history(previous_history, binding)
        attempts = previous["attempts"]
        if previous["environment_reference"] != dict(environment_reference):
            raise ResumeRefusal("accounting history environment reference changed")
        if prior_exposure is not None and previous["prior_exposure"] != dict(prior_exposure):
            raise ResumeRefusal("accounting historical exposure changed")
        prior_exposure = previous["prior_exposure"]
    _bound_metadata(environment_reference)
    if not isinstance(attempt, Mapping) or attempt.get("attempt_index") != len(attempts):
        raise ResumeRefusal("accounting attempts must be contiguous from original index zero")
    if len(attempts) >= 3:
        raise ResumeRefusal("accounting allows original plus at most two completion attempts")
    value = {
        "schema": ACCOUNTING_SCHEMA,
        "identity": dict(binding),
        "environment_reference": dict(environment_reference),
        "prior_exposure": None if prior_exposure is None else dict(prior_exposure),
        "previous_history": None if previous_history is None else dict(previous_history),
        "attempts": [*attempts, dict(attempt)],
    }
    export_json(path, value)
    result = {**metadata_reference(path), "attempt_count": len(value["attempts"])}
    verify_accounting_history(result, binding)
    return result


def verify_accounting_history(
    frozen: Mapping[str, Any], binding: Mapping[str, str]
) -> dict[str, Any]:
    """Verify the externally frozen latest prefix and every immutable predecessor."""
    if not isinstance(frozen, Mapping) or set(frozen) != {
        "path",
        "size",
        "sha256",
        "attempt_count",
    }:
        raise ResumeRefusal("accounting needs an externally frozen history reference and count")
    count = frozen["attempt_count"]
    if type(count) is not int or not 1 <= count <= 3:
        raise ResumeRefusal("accounting history count exceeds completion retry maximum")
    raw = _bound_metadata({k: v for k, v in frozen.items() if k != "attempt_count"}, immutable=True)
    if (
        set(raw)
        != {
            "schema",
            "identity",
            "environment_reference",
            "prior_exposure",
            "previous_history",
            "attempts",
        }
        or raw["schema"] != ACCOUNTING_SCHEMA
        or raw["identity"] != dict(binding)
        or not isinstance(raw["attempts"], list)
        or len(raw["attempts"]) != count
    ):
        raise ResumeRefusal("accounting history identity or attempt count differs")
    names: set[str] = set()
    for index, attempt in enumerate(raw["attempts"]):
        if (
            not isinstance(attempt, dict)
            or attempt.get("attempt_index") != index
            or type(attempt.get("attempt_index")) is not int
            or not isinstance(attempt.get("attempt_id"), str)
            or not attempt["attempt_id"]
            or attempt["attempt_id"] in names
        ):
            raise ResumeRefusal(
                "accounting history has omitted duplicate or noncontiguous attempts"
            )
        names.add(attempt["attempt_id"])
    if count == 1:
        if raw["previous_history"] is not None:
            raise ResumeRefusal("original accounting history cannot have a predecessor")
    else:
        previous = raw["previous_history"]
        if not isinstance(previous, dict) or previous.get("attempt_count") != count - 1:
            raise ResumeRefusal("accounting history omitted its previous immutable prefix")
        prior = verify_accounting_history(previous, binding)
        if (
            any(
                raw[key] != prior[key]
                for key in ("identity", "environment_reference", "prior_exposure")
            )
            or raw["attempts"][:-1] != prior["attempts"]
        ):
            raise ResumeRefusal("accounting history rewrites previous attempts")
    if raw["prior_exposure"] is not None:
        exposure = raw["prior_exposure"]
        if not isinstance(exposure, dict) or set(exposure) != {"seconds", "reference"}:
            raise ResumeRefusal("historical accounting exposure is malformed")
        if exposure["seconds"] != 6811.840706833173:
            raise ResumeRefusal("historical accounting exposure differs from the ruling")
        _bound_metadata(exposure["reference"])
    return raw


def _attempt_accounting(
    frozen: Mapping[str, Any],
    binding: Mapping[str, str],
    records: Mapping[Any, Any],
    *,
    ledger_path: Path,
) -> tuple[float | None, dict[str, Any]]:
    from research.bl_residual import f5_environment

    history = verify_accounting_history(frozen, binding)
    # Do not accept an arbitrary mapping supplied by a reducer caller.  Reopen
    # the immutable, hash-bound ledger and retain its exact paths/digests for
    # broker-receipt provenance matching.
    scanned = load_ledger(ledger_path, binding)
    if dict(records) != scanned:
        raise ResumeRefusal("attempt accounting records differ from verified ledger")
    ledger = _bound_metadata(
        {
            "path": str(ledger_path.absolute()),
            "size": ledger_path.stat().st_size,
            "sha256": verify_export(ledger_path),
        },
        immutable=True,
    )
    ledger_entries: dict[int | str, dict[str, Any]] = {}
    for item in ledger["units"]:
        if not isinstance(item, dict) or item.get("key") in ledger_entries:
            raise ResumeRefusal("verified ledger provenance entries are malformed")
        ledger_entries[item["key"]] = item
    reference = _bound_metadata(history["environment_reference"])
    from research.bl_residual import f5_environment_reference

    legacy_reference = reference.get("schema") == f5_environment_reference.SCHEMA
    if legacy_reference:
        f5_environment_reference.validate_capture05_reference(reference)
    else:
        f5_environment.validate_environment(reference)
    overhead = 0.0
    measured: list[dict[str, Any]] = []
    missing: list[str] = []
    seen_completed_keys: set[int | str] = set()
    ids = {row["attempt_id"] for row in history["attempts"]}
    logical_ids = set(ids)
    alias_logical_ids: set[str] = set()
    for attempt in history["attempts"]:
        alias_reference = attempt.get("preallocation_refusal_alias")
        if alias_reference is None:
            continue
        alias = _preallocation_refusal_alias(
            alias_reference,
            manifest=_bound_metadata(attempt.get("manifest")),
            control=_bound_metadata(attempt.get("control")),
            attempt_index=attempt["attempt_index"],
        )
        if attempt["attempt_id"] != alias["broker_attempt_id"]:
            raise ResumeRefusal("preallocation refusal alias history broker identity differs")
        if (
            alias["logical_attempt_id"] in logical_ids
            or alias["logical_attempt_id"] in alias_logical_ids
        ):
            raise ResumeRefusal(
                "preallocation refusal alias logical identity collides with an attempt"
            )
        alias_logical_ids.add(alias["logical_attempt_id"])
        logical_ids.add(alias["logical_attempt_id"])
    if any(row.get("timing_attempt_id") not in logical_ids for row in records.values()):
        raise ResumeRefusal("completed unit has missing or foreign contributing attempt provenance")
    for attempt in history["attempts"]:
        name = attempt["attempt_id"]
        manifest = _bound_metadata(attempt.get("manifest"))
        receipt = _bound_metadata(attempt.get("broker_receipt"))
        upload = _bound_metadata(attempt.get("upload_measurement"))
        control = _bound_metadata(attempt.get("control"))
        alias_reference = attempt.get("preallocation_refusal_alias")
        logical_name = name
        if alias_reference is not None:
            alias = _preallocation_refusal_alias(
                alias_reference,
                manifest=manifest,
                control=control,
                receipt=receipt,
                observed_normalized_manifest=Path(attempt["broker_receipt"]["path"]).parent
                / "manifest.json",
                attempt_index=attempt["attempt_index"],
            )
            logical_name = alias["logical_attempt_id"]
        sealed = control.get("identity", {})
        if (
            manifest.get("job_id") != name
            or receipt.get("job_id") != name
            or (
                alias_reference is None
                and receipt.get("manifest_sha256") != attempt["manifest"]["sha256"]
            )
            or receipt.get("commit") != manifest.get("commit")
            or receipt.get("stop_verified") is not True
            or receipt.get("cleanup_error") is not None
            or control.get("timing_attempt_id") != logical_name
            or sealed.get("source_sha256") != binding["source_sha256"]
            or sealed.get("environment_sha256") != binding["environment_sha256"]
            or sealed.get("allocation_sha256") != binding["allocation"]
            or control.get("timing_input_sha256") != binding["input_sha256"]
            or not any(
                row.get("dest") == "control.json"
                and row.get("sha256") == attempt["control"]["sha256"]
                for row in manifest.get("inputs", [])
            )
        ):
            raise ResumeRefusal("attempt broker stop manifest control or identity binding differs")
        status = receipt.get("status")
        if status not in {"PASS", "INCOMPLETE"}:
            raise ResumeRefusal("attempt receipt status is not admissible")
        if status == "PASS" and (
            not isinstance(receipt.get("terminal"), dict)
            or receipt["terminal"].get("returncode") != 0
        ):
            raise ResumeRefusal("successful attempt lacks successful terminal evidence")
        exports = receipt.get("exports")
        if not isinstance(exports, list):
            raise ResumeRefusal("attempt receipt lacks broker export provenance")
        exported_by_path: dict[str, dict[str, Any]] = {}
        for exported in exports:
            if (
                not isinstance(exported, dict)
                or set(exported) != {"path", "size", "sha256"}
                or not isinstance(exported.get("path"), str)
                or not exported["path"]
                or type(exported.get("size")) is not int
                or exported["size"] < 0
                or not isinstance(exported.get("sha256"), str)
                or re.fullmatch(r"[0-9a-f]{64}", exported["sha256"]) is None
                or exported["path"] in exported_by_path
            ):
                raise ResumeRefusal("attempt broker exports are malformed or duplicate")
            exported_by_path[exported["path"]] = exported

        def require_export(reference: Any, *, label: str) -> dict[str, Any]:
            if (
                not isinstance(reference, Mapping)
                or set(reference) != {"path", "size", "sha256"}
                or not isinstance(reference.get("path"), str)
                or not reference["path"]
                or type(reference.get("size")) is not int
                or reference["size"] < 0
                or not isinstance(reference.get("sha256"), str)
                or re.fullmatch(r"[0-9a-f]{64}", reference["sha256"]) is None
                or exported_by_path.get(reference["path"]) != dict(reference)
            ):
                raise ResumeRefusal(f"attempt {label} is absent from broker verified exports")
            return dict(reference)

        entries_by_path = {entry["path"]: (key, entry) for key, entry in ledger_entries.items()}
        contributed: list[tuple[int | str, dict[str, Any]]] = []
        # Every receipt-listed timing-unit export is immutable evidence.  Do
        # not let a later ledger select a replacement record and hide a prior
        # completed unit in another attempt's receipt.
        for exported in exports:
            exported_path = exported["path"]
            if not exported_path.startswith("timing-units/"):
                continue
            key_entry = entries_by_path.get(exported_path)
            if key_entry is None:
                raise ResumeRefusal("attempt exported an unknown timing-unit path")
            key, entry = key_entry
            if key in seen_completed_keys:
                raise ResumeRefusal("completed timing-unit key appears in multiple attempts")
            if exported != {field: entry[field] for field in ("path", "size", "sha256")}:
                raise ResumeRefusal("attempt timing-unit export differs from verified final ledger")
            row = records[key]
            if row.get("timing_attempt_id") != logical_name:
                raise ResumeRefusal("attempt timing-unit payload provenance differs from receipt")
            seen_completed_keys.add(key)
            contributed.append((key, row))
            require_export(
                {field: entry[field] for field in ("path", "size", "sha256")},
                label="completed-unit payload",
            )
            # Each unit binds the raw runtime attestations it observed.  Their
            # availability is proved by the same broker receipt; absent
            # attestations hold/refuse rather than being inferred from a later
            # final export.
            require_export(row.get("environment_before"), label="environment-before attestation")
            require_export(row.get("environment_after"), label="environment-after attestation")
        before = _bound_metadata(attempt.get("environment_before"))
        after = _bound_metadata(attempt.get("environment_after"))
        # History references must name their broker export exactly, rather than
        # matching only a digest.  A before and after snapshot may legitimately
        # have identical bytes, so hash-only matching would be ambiguous.
        require_export(attempt.get("environment_before_export"), label="environment-before")
        require_export(attempt.get("environment_after_export"), label="environment-after")
        if (
            attempt["environment_before_export"].get("size")
            != attempt["environment_before"].get("size")
            or attempt["environment_before_export"].get("sha256")
            != attempt["environment_before"].get("sha256")
            or attempt["environment_after_export"].get("size")
            != attempt["environment_after"].get("size")
            or attempt["environment_after_export"].get("sha256")
            != attempt["environment_after"].get("sha256")
        ):
            raise ResumeRefusal(
                "attempt runtime attestation export differs from local verified reference"
            )
        if legacy_reference:
            f5_environment_reference.compare_capture05_reference(reference, before)
        else:
            f5_environment.compare_environment(reference, before, mode="cross-assignment")
        f5_environment.compare_environment(before, after, mode="same-assignment")
        assignment_raw_sha256 = f5_environment.receipt_sha256(before)
        for _, row in contributed:
            if row["kernel_provenance"]["assignment_raw_sha256"] != assignment_raw_sha256:
                raise ResumeRefusal("timing unit raw assignment differs from its broker attempt")
        # Absence is HOLD, never a fabricated zero or inferred decomposition.
        if attempt.get("accounting_measurement") is None:
            missing.append(name)
            continue
        measurement = _bound_metadata(attempt["accounting_measurement"], immutable=True)
        if measurement.get("identity") != dict(binding) or measurement.get("attempt_id") != name:
            raise ResumeRefusal("attempt overhead measurement identity differs")
        values = {}
        for key in ("setup_seconds", "io_seconds", "imbalance_seconds"):
            value = measurement.get(key)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value < 0
            ):
                raise ResumeRefusal("attempt overhead measurement is missing or nonfinite")
            values[key] = float(value)
        for key in ("setup_seconds", "transfer_seconds"):
            value = upload.get(key)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value < 0
            ):
                raise ResumeRefusal("attempt upload measurement is missing or nonfinite")
        if (
            values["setup_seconds"] < upload["setup_seconds"]
            or values["io_seconds"] < upload["transfer_seconds"]
        ):
            raise ResumeRefusal("attempt overhead omits measured setup or transfer I/O")
        elapsed = receipt.get("elapsed_seconds")
        useful = sum(row["elapsed_seconds"] for _, row in contributed)
        total = sum(values.values())
        if (
            not isinstance(elapsed, (int, float))
            or isinstance(elapsed, bool)
            or not math.isfinite(elapsed)
            or elapsed <= 0
            or not math.isclose(total + useful, elapsed, abs_tol=1e-6, rel_tol=1e-9)
        ):
            raise ResumeRefusal(
                "attempt accounting omits or double counts measured elapsed exposure"
            )
        overhead += total
        measured.append(
            {"attempt_id": name, **values, "elapsed_seconds": elapsed, "status": status}
        )
    if seen_completed_keys != set(records):
        raise ResumeRefusal("verified final ledger omits a receipt-bound completed timing unit")
    return (None if missing else overhead), {
        "history": dict(frozen),
        "measured_attempts": measured,
        "missing_measurements": missing,
        "prior_exposure": history["prior_exposure"],
    }


def seal_attempt_accounting(
    output: Path,
    binding: Mapping[str, str],
    *,
    manifest: Path,
    broker_receipt: Path,
    upload_measurement: Path,
    control: Path,
    exports_root: Path,
    attempt_index: int,
    environment_after_path: str | None = None,
    preallocation_refusal_alias: Path | None = None,
) -> dict[str, Any]:
    """Seal one real broker attempt's accounting inputs without inventing time.

    ``exports_root`` is the supervisor's verified ``output/exports`` directory.
    The broker inventory remains authoritative for every completed unit and for
    the initial/final (or last-unit) runtime attestation.  This helper only
    derives the residual imbalance from the broker's observed elapsed time.
    """
    if type(attempt_index) is not int or attempt_index < 0:
        raise ResumeRefusal("attempt accounting index is malformed")
    receipt = _bound_metadata(metadata_reference(broker_receipt))
    name = receipt.get("job_id")
    exports = receipt.get("exports")
    if not isinstance(name, str) or not name or not isinstance(exports, list):
        raise ResumeRefusal("attempt receipt lacks job or broker exports")
    manifest_value = _bound_metadata(metadata_reference(manifest))
    control_value = _bound_metadata(metadata_reference(control))
    alias_reference = None
    logical_name = name
    if preallocation_refusal_alias is not None:
        alias_reference = metadata_reference(preallocation_refusal_alias)
        alias = _preallocation_refusal_alias(
            alias_reference,
            manifest=manifest_value,
            control=control_value,
            receipt=receipt,
            observed_normalized_manifest=broker_receipt.parent / "manifest.json",
            attempt_index=attempt_index,
        )
        logical_name = alias["logical_attempt_id"]
    by_path: dict[str, dict[str, Any]] = {}
    for item in exports:
        if (
            not isinstance(item, dict)
            or set(item) != {"path", "size", "sha256"}
            or not isinstance(item.get("path"), str)
            or not item["path"]
            or type(item.get("size")) is not int
            or item["size"] < 0
            or not isinstance(item.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is None
            or item["path"] in by_path
        ):
            raise ResumeRefusal("attempt receipt exports are malformed or duplicate")
        local = _safe_file(exports_root / _safe_relative(item["path"]))
        try:
            if (
                not local.is_file()
                or verify_export(local) != item["sha256"]
                or local.stat().st_size != item["size"]
            ):
                raise ResumeRefusal("broker-exported attempt payload is unavailable or drifted")
        except OSError as exc:
            raise ResumeRefusal("broker-exported attempt payload is unavailable") from exc
        by_path[item["path"]] = dict(item)

    before_export = by_path.get("environment/before.json")
    if before_export is None:
        raise ResumeRefusal("attempt lacks broker-exported initial environment attestation")
    if environment_after_path is None:
        if receipt.get("status") == "PASS":
            environment_after_path = "environment/after.json"
        else:
            raise ResumeRefusal(
                "incomplete attempt must name its last completed environment attestation"
            )
    after_export = by_path.get(environment_after_path)
    if after_export is None:
        raise ResumeRefusal("attempt lacks broker-exported closing environment attestation")

    useful = 0.0
    completed_keys: set[int | str] = set()
    for exported in by_path.values():
        if not exported["path"].startswith("timing-units/") or not exported["path"].endswith(
            ".json"
        ):
            continue
        key, payload = _read_unit(
            exports_root / _safe_relative(exported["path"]), binding, exported
        )
        if key in completed_keys or payload.get("timing_attempt_id") != logical_name:
            raise ResumeRefusal("attempt completed-unit provenance is foreign or duplicate")
        completed_keys.add(key)
        useful += float(payload["elapsed_seconds"])
    upload = _bound_metadata(metadata_reference(upload_measurement))
    setup = upload.get("setup_seconds")
    transfer = upload.get("transfer_seconds")
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
        for value in (setup, transfer)
    ):
        raise ResumeRefusal("attempt upload measurement is missing or nonfinite")
    download = 0.0
    for row in receipt.get("transfers", []):
        if not isinstance(row, Mapping) or row.get("direction") != "download":
            continue
        value = row.get("elapsed_seconds")
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value < 0
        ):
            raise ResumeRefusal("attempt download timing is malformed")
        download += float(value)
    elapsed = receipt.get("elapsed_seconds")
    if (
        not isinstance(elapsed, (int, float))
        or isinstance(elapsed, bool)
        or not math.isfinite(elapsed)
        or elapsed <= 0
    ):
        raise ResumeRefusal("attempt receipt elapsed time is malformed")
    # Incremental export downloads may overlap remote fixture/calibration work.
    # They are observed operational I/O, but are not additive elapsed exposure;
    # adding them would double-count a valid concurrent interval and can make a
    # complete attempt look impossible.  The residual remains the disjoint
    # whole-session time after useful work and the staged setup/transfer terms.
    imbalance = float(elapsed) - useful - float(setup) - float(transfer)
    if not math.isfinite(imbalance) or imbalance < 0:
        raise ResumeRefusal("attempt elapsed is smaller than measured completed work and I/O")
    measurement_path = output / f"attempt-{attempt_index}-accounting.json"
    export_json(
        measurement_path,
        {
            "identity": dict(binding),
            "attempt_id": name,
            "setup_seconds": float(setup),
            "io_seconds": float(transfer),
            "imbalance_seconds": imbalance,
            "observed_incremental_export_download_seconds": download,
            "incremental_export_downloads_may_overlap_useful_runtime": True,
            "classification": "RESIDUAL_BOOTSTRAP_FAILED_PARTIAL_OR_OTHER_MEASURED_TIME",
        },
    )
    result = {
        "attempt_index": attempt_index,
        "attempt_id": name,
        "manifest": metadata_reference(manifest),
        "broker_receipt": metadata_reference(broker_receipt),
        "upload_measurement": metadata_reference(upload_measurement),
        "control": metadata_reference(control),
        "accounting_measurement": metadata_reference(measurement_path),
        "environment_before": metadata_reference(exports_root / before_export["path"]),
        "environment_after": metadata_reference(exports_root / after_export["path"]),
        "environment_before_export": before_export,
        "environment_after_export": after_export,
    }
    if alias_reference is not None:
        result["preallocation_refusal_alias"] = alias_reference
        result["logical_attempt_id"] = logical_name
    return result


def reduce_ledger(
    path: Path,
    binding: Mapping[str, str],
    *,
    attempts: Sequence[Mapping[str, Any]] = (),
    accounting_history: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Mac-only reducer requires the exact 13 units and all-attempt overhead."""
    if platform.system() != "Darwin":
        raise ResumeRefusal("timing reducer is Mac-only")
    records = load_ledger(path, binding)
    if set(records) != _keys():
        raise ResumeRefusal("Mac reducer requires all thirteen completed units")
    from research.bl_residual.f5_timing import _canonical_timing_parents

    fixtures = [records[index] for index in range(672, 684)]
    expected = _canonical_timing_parents()
    for actual, canonical in zip(fixtures, expected):
        if any(
            actual.get(key) != canonical[key]
            for key in ("allocation_index", "parent_id", "group", "group_index", "timing_only")
        ):
            raise ResumeRefusal("Mac reducer fixture allocation identity differs")
    maxima: dict[str, float] = {}
    for key in ("t_root", "t_branch", "t_nat", "watch_seconds"):
        measured = fixtures[:3] if key == "watch_seconds" else fixtures
        values = [item.get(key) for item in measured]
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value <= 0
            for value in values
        ):
            raise ResumeRefusal("Mac reducer requires measured primitive timing maxima")
        maxima[key] = max(float(value) for value in values)
    seen_attempts: set[str] = set()
    overhead = 0.0
    for attempt in attempts:
        if not isinstance(attempt, Mapping) or attempt.get("identity") != dict(binding):
            raise ResumeRefusal("Mac reducer attempt identity differs")
        name = attempt.get("attempt_id")
        if not isinstance(name, str) or not name or name in seen_attempts:
            raise ResumeRefusal("Mac reducer attempt accounting is duplicate or missing")
        seen_attempts.add(name)
        for key in ("setup_seconds", "io_seconds", "imbalance_seconds"):
            value = attempt.get(key)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value < 0
            ):
                raise ResumeRefusal("Mac reducer requires all measured attempt overhead")
            overhead += float(value)
    if not seen_attempts and accounting_history is None and attempts:
        raise ResumeRefusal("Mac reducer requires every attempt accounting record")
    accounting_evidence = None
    accounting_verified = False
    if accounting_history is not None:
        verified_overhead, accounting_evidence = _attempt_accounting(
            accounting_history, binding, records, ledger_path=path
        )
        accounting_verified = verified_overhead is not None
        if accounting_verified:
            overhead = verified_overhead
    archive_hours = (
        128 * maxima["t_nat"] + 14720 * maxima["t_root"] + 88320 * maxima["t_branch"]
    ) / 6 / 3600 + overhead / 3600
    calibration = records[CALIBRATION_KEY]
    calibration_max = max(
        float(arm[key])
        for arm in calibration["arms"].values()
        for key in ("full_seconds", "serialized_continuation_seconds")
    )
    covered = all(
        all(
            any(row.get(flag) is True for row in fixtures[start : start + 3])
            for flag in ("full_h300", "long_body", "full_h5000_natural")
        )
        for start in range(0, 12, 3)
    ) and any(row.get("watch_fixture_frames") == 14_400 for row in fixtures[:3])
    kernel_strata: dict[str, dict[str, Any]] = {}
    for key, row in records.items():
        provenance = row["kernel_provenance"]
        kernel = provenance["os_kernel"]
        label = json.dumps(
            [kernel["release"], kernel["version"], provenance["platform"]],
            separators=(",", ":"),
        )
        stratum = kernel_strata.setdefault(
            label,
            {
                "release": kernel["release"],
                "version": kernel["version"],
                "platform": provenance["platform"],
                "units": [],
            },
        )
        stratum["units"].append(
            {
                "key": key,
                "gpu_uuid": provenance["gpu_uuid"],
                "assignment_raw_sha256": provenance["assignment_raw_sha256"],
                "elapsed_seconds": row["elapsed_seconds"],
                **{
                    name: row[name]
                    for name in ("t_root", "t_branch", "t_nat", "watch_seconds", "arms")
                    if name in row
                },
            }
        )
    return {
        "schema": "bl-f5-timing-mac-reducer/v2",
        "status": (
            (
                "COMPLETE_UNITS_PENDING_MAC_FORECAST"
                if covered
                else "HOLD_REGISTERED_COVERAGE_FAILED"
            )
            if accounting_verified
            else "HOLD_UNVERIFIED_ATTEMPT_ACCOUNTING"
        ),
        "accounting_verified": accounting_verified,
        "accounting_evidence": accounting_evidence,
        "registered_coverage_passed": covered,
        "fixtures": fixtures,
        "calibration": calibration,
        "unit_count": 13,
        "max_seconds": maxima,
        "kernel_strata": list(kernel_strata.values()),
        "forecast_kernel_rule": "SLOWEST_OBSERVED_QUALIFYING_PRIMITIVE_NO_KERNEL_AVERAGING",
        "archive_registered_conservative_hours": archive_hours if accounting_verified else None,
        "coupled_calibration_max_seconds": calibration_max,
        "all_attempt_setup_io_imbalance_seconds": overhead if accounting_verified else None,
        "mac_probe_product_forecast": "HOLD_NOT_MEASURED",
    }
