"""Mac-only binding from a sealed C/T packet to its admitted live GPU variant.

The runtime never imports this module. Canonical packet lineage stays sealed;
only the dispatcher accounting snapshot and the supervisor lease default may
change the selected variant before execution.
"""

from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping

from research.compute.gpu_preference import variants


def resolve_dispatch_binding(
    packet: Path,
    canonical: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    paths: Mapping[str, Path] | None = None,
    binding: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Verify every linked file, returning execution identity/hash and sealed proof.

    Supplying a previously sealed binding reopens its files and requires exactly
    the same descriptors. No receipt or hardware identity is inferred from a
    preferred GPU, filename suffix, or mutable operator declaration.
    """
    from research.sandbox_a_resident.ct_preflight import _bounded_manifest_bytes
    from research.sandbox_a_resident.ct_resume import CTResumeRefused, _load_json
    from tools.sandbox.registry import digest

    def refuse(message: str) -> None:
        raise CTResumeRefused("dispatch binding: " + message)

    descriptors: dict[str, Any] = {}

    def read(name: str, path: Path) -> dict[str, Any]:
        if not path.is_absolute():
            refuse(f"{name} path must be absolute")
        raw = _bounded_manifest_bytes(path)
        value = _load_json(raw)
        if not isinstance(value, dict):
            refuse(f"{name} must be an object")
        descriptors[name] = {
            "path": str(path.resolve()),
            "size": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        return value

    if (paths is None) == (binding is None):
        refuse("supply either evidence paths or a sealed binding")
    if binding is not None:
        if not isinstance(binding, Mapping):
            refuse("binding must be an object")
        if binding.get("schema") != "t34-cf-dispatch-binding/v1":
            refuse("unknown binding schema")
        records = binding.get("files")
        if not isinstance(records, dict):
            refuse("missing evidence descriptors")
        try:
            names = (
                ("preference", "accounting", "gpu_receipt")
                if "preference" in records
                else ("accounting",)
            )
            paths = {name: Path(records[name]["path"]) for name in names}
        except (KeyError, TypeError) as error:
            raise CTResumeRefused("dispatch binding: missing evidence paths") from error
    assert paths is not None
    preferred = canonical.get("requirements", {}).get("prefer") is not None
    required = {"preference", "accounting", "gpu_receipt"} if preferred else {"accounting"}
    if set(paths) != required:
        refuse("evidence paths must match preferred or fixed-hardware dispatch")
    seal = read("packet_seal", packet / "SEAL.json")
    if seal.get("manifest_sha256") != digest(packet / "dispatch-manifest.json"):
        refuse("canonical manifest differs from existing seal")
    accounting = read("accounting", paths["accounting"])
    expected_variants = variants(dict(canonical))
    gpu = None
    if preferred:
        preference = read("preference", paths["preference"])
        if preference.get("schema") != "colab-gpu-preference/v1" or preference.get(
            "preference"
        ) != [
            "H100",
            "A100",
        ]:
            refuse("unknown preference evidence")
        rows = preference.get("variants")
        if len(expected_variants) != 2 or not isinstance(rows, list) or len(rows) != 2:
            refuse("preference must bind both admitted variants")
        selected = None
        names = []
        for index, (row, expected) in enumerate(zip(rows, expected_variants)):
            if not isinstance(row, dict):
                refuse("malformed variant descriptor")
            try:
                admitted = read(f"variant_{index}", Path(row["manifest"]))
            except (KeyError, TypeError) as error:
                raise CTResumeRefused("dispatch binding: variant path missing") from error
            if (
                admitted != expected
                or row.get("sha256") != descriptors[f"variant_{index}"]["sha256"]
                or row.get("gpu") != expected["requirements"]["gpu"]
                or row.get("budget") != expected["budget"]
                or not isinstance(row.get("name"), str)
                or not row["name"]
            ):
                refuse("variant differs from pure sealed derivation")
            names.append(row["name"])
            if receipt.get("job_id") == expected["job_id"]:
                selected = (index, row, expected)
        if len(set(names)) != 2 or selected is None:
            refuse("foreign job or ambiguous variant identity")
        index, row, expected = selected
        gpu = read("gpu_receipt", paths["gpu_receipt"])
        if str(paths["gpu_receipt"].resolve()) != row.get("receipt"):
            refuse("GPU receipt path differs from admitted variant")
    else:
        expected = dict(canonical)
        index = 0
        try:
            source_path = Path(accounting["source_manifest"])
            admitted = read("variant_0", source_path)
        except (KeyError, TypeError) as error:
            raise CTResumeRefused("dispatch binding: accounting source missing") from error
        if admitted != expected or receipt.get("job_id") != expected.get("job_id"):
            refuse("fixed-hardware source differs from sealed packet or foreign job")
        row = {"manifest": str(source_path), "sha256": descriptors["variant_0"]["sha256"]}
    try:
        live_path = Path(accounting["manifest"])
        live = read("live_manifest", live_path)
    except (KeyError, TypeError) as error:
        raise CTResumeRefused("dispatch binding: accounting manifest missing") from error
    if (
        accounting.get("source_manifest") != row["manifest"]
        or accounting.get("source_manifest_sha256") != row["sha256"]
        or accounting.get("manifest_sha256") != descriptors["live_manifest"]["sha256"]
    ):
        refuse("accounting source/live manifest hash mismatch")
    rates = accounting.get("other_assignments")
    if not isinstance(rates, dict) or any(
        not isinstance(endpoint, str)
        or not endpoint
        or type(rate) not in (int, float)
        or not math.isfinite(rate)
        or rate <= 0
        for endpoint, rate in rates.items()
    ):
        refuse("invalid accounting rates")
    refreshed = copy.deepcopy(expected)
    refreshed["budget"]["other_assignments"] = rates
    if live != refreshed:
        refuse("live manifest changes fields beyond accounting refresh")
    output = Path(expected["output_dir"])
    executed = read("executed_manifest", output / "manifest.json")
    normalized = copy.deepcopy(live)
    normalized.setdefault("lease_seconds", 180)
    if executed != normalized:
        refuse("retained manifest changes fields beyond supervisor lease default")
    disk_receipt = read("supervisor_receipt", output / "receipt.json")
    hardware = read("hardware", output / "hardware.json")
    if (
        dict(receipt) != disk_receipt
        or receipt.get("manifest_sha256") != descriptors["executed_manifest"]["sha256"]
        or receipt.get("commit") != expected["commit"]
        or hardware.get("hardware", {}).get("kind") != expected["requirements"]["gpu"]
    ):
        refuse("execution or observed hardware identity differs")
    if gpu is not None:
        if (
            gpu.get("schema") != "colab-gpu-dispatch-receipt/v1"
            or gpu.get("job_id") != expected["job_id"]
            or gpu.get("requested_gpu") != expected["requirements"]["gpu"]
            or gpu.get("observed_hardware") != hardware
        ):
            refuse("GPU receipt identity differs")
        for field, name in (
            ("manifest", "live_manifest"),
            ("hardware_evidence", "hardware"),
            ("supervisor_receipt", "supervisor_receipt"),
        ):
            desc = descriptors[name]
            if gpu.get(field) != {key: desc[key] for key in ("path", "sha256")}:
                refuse(f"GPU receipt {field} hash/path differs")
        for field in (
            "endpoint",
            "status",
            "stop_verified",
            "usage_before",
            "usage_after",
            "accounting",
        ):
            if gpu.get(field) != receipt.get(field):
                refuse(f"GPU receipt {field} differs from supervisor")
        if gpu.get("budget") != live["budget"]:
            refuse("GPU receipt budget differs")
    # The fallback is legal only after the dispatcher's qualified reconciliation.
    if index == 1:
        unavailable = read("unavailable", paths["preference"].parent / "unavailable.json")
        primary_receipt = read(
            "unavailable_receipt", Path(expected_variants[0]["output_dir"]) / "receipt.json"
        )
        desc = descriptors["unavailable_receipt"]
        if (
            unavailable.get("schema") != "colab-allocation-unavailable/v1"
            or unavailable.get("classification") != "ALLOCATION_UNAVAILABLE"
            or unavailable.get("mode") != "LIVE"
            or unavailable.get("endpoint_ever_assigned") is not False
            or unavailable.get("vm_compute_started") is not False
            or unavailable.get("recovery") != {"endpoint": None}
            or unavailable.get("receipt") != {key: desc[key] for key in ("path", "sha256")}
            or primary_receipt.get("job_id") != canonical["job_id"]
            or primary_receipt.get("commit") != canonical["commit"]
            or primary_receipt.get("endpoint") is not None
        ):
            refuse("fallback lacks bound pre-endpoint reconciliation")
    proof = {"schema": "t34-cf-dispatch-binding/v1", "files": descriptors}
    if binding is not None and proof != binding:
        refuse("sealed dispatch evidence changed")
    return executed, descriptors["executed_manifest"]["sha256"], proof
