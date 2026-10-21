"""Read-only scientific admission checks, shared by manifest preparation."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Mapping

from research.bl_ppo.exports import verify_export
from research.bl_prefix import seeds
from research.bl_prefix.controllers import B8Controller, E2Controller


def local_inputs(value: Any, files: Mapping[str, Path]) -> Any:
    """Resolve upload tags to hash-bound local files without copying data."""
    if isinstance(value, dict):
        return {key: local_inputs(child, files) for key, child in value.items()}
    if isinstance(value, list):
        return [local_inputs(child, files) for child in value]
    if isinstance(value, str) and (value.startswith("/") or "/drive/" in value):
        raise ValueError("scientific paths must name hash-bound uploaded inputs")
    if not isinstance(value, str) or not value.startswith("input:"):
        return value
    relative = value[6:]
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("unsafe input-relative path")
    if relative in files:
        return str(files[relative])
    parents = set()
    for dest, path in files.items():
        if dest.startswith(relative.rstrip("/") + "/"):
            suffix = Path(dest).relative_to(relative)
            parent = path
            for _ in suffix.parts:
                parent = parent.parent
            parents.add(parent)
    if len(parents) == 1:
        return str(parents.pop())
    raise ValueError(f"runtime input is unbound: {relative}")


def _delivery(plan: Mapping[str, Any], files: Mapping[str, Path]) -> None:
    """Check that dependencies exist in the upload inventory, not only on this Mac."""

    def tagged(value: Any) -> str:
        if not isinstance(value, str) or not value.startswith("input:"):
            raise ValueError("scientific file fields must use input: delivery tags")
        return value[6:]

    for key in ("forecast", "seed_audit", "checkpoint_dir"):
        tagged(plan.get(key))
    for key in ("controllers", "critic_receipts", "checkpoint_relocations"):
        for value in plan.get(key, {}).values():
            tagged(value)
    if plan.get("experiment") == "e7":
        tagged(plan.get("b8_checkpoint"))
    for key in ("seed_audit", "b8_checkpoint"):
        if plan.get("collection") is not None:
            tagged(plan["collection"].get(key))
    from research.apex_safety_20260926 import dev_screen
    from research.bl_prefix.controllers import B8_SHA256
    from research.redesign_scope_20261007 import grid_h5000_identity as gid

    folder = tagged(plan["checkpoint_dir"])
    expected = [*dev_screen.POOL, gid.FRP3_S12, ("b8_s3_t7_20261016.pth", B8_SHA256)]
    for name, digest in expected:
        path = files.get(f"{folder}/{name}")
        if path is None or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"required opponent/comparator upload is absent or changed: {name}")
    if plan.get("collection") is None:
        relative = tagged(plan.get("bank_bundle"))
        index = files.get(relative)
        if index is None:
            raise ValueError("scientific bank bundle is not delivered")
        verify_export(index)
        for chunk in json.loads(index.read_text())["chunks"]:
            dest = (Path(relative).parent / chunk["path"]).as_posix()
            path = files.get(dest)
            if (
                path is None
                or path.stat().st_size != chunk["size"]
                or hashlib.sha256(path.read_bytes()).hexdigest() != chunk["sha256"]
            ):
                raise ValueError("bank archive chunk is not in the bound upload inventory")


def validate_ready(plan: dict[str, Any], files: Mapping[str, Path]) -> None:
    """Refuse ready=true scientific plans whose actual inputs do not pass."""
    if plan.get("ready") is not True:
        return
    if plan.get("preflight") is True:
        raise ValueError("Colab preflight dispatch is outside this build; use local entrypoints")
    try:
        _delivery(plan, files)
        resolved = local_inputs(plan, files)
        if resolved.get("bank_bundle"):
            from research.bl_prefix.bundles import unpack_bank

            with tempfile.TemporaryDirectory(prefix="bl-prefix-admission-") as directory:
                resolved["bank"] = str(
                    unpack_bank(Path(resolved["bank_bundle"]), Path(directory) / "bank")
                )
                _validate_resolved(resolved)
        else:
            _validate_resolved(resolved)
    except (KeyError, TypeError, OSError) as exc:
        raise ValueError(f"scientific inputs are incomplete: {exc}") from exc


def _validate_resolved(resolved: dict[str, Any]) -> None:
    from research.bl_prefix.make_manifest import forecast as compute_forecast

    forecast_path = Path(resolved["forecast"])
    verify_export(forecast_path)
    forecast = json.loads(forecast_path.read_text())
    if (
        forecast.get("fits_with_two_times_margin") is not True
        or forecast.get("experiment") != resolved["experiment"]
    ):
        raise ValueError("scientific forecast cannot fit the frozen cap")
    recomputed = compute_forecast(resolved["experiment"], forecast["mac_seconds_per_world_step"])
    if any(forecast.get(key) != value for key, value in recomputed.items()):
        raise ValueError("forecast differs from the frozen work/cap calculation")
    audit_path = Path(resolved["seed_audit"])
    verify_export(audit_path)
    audit = json.loads(audit_path.read_text())
    seeds.verify_receipt(audit, round_index=int(audit["round_index"]))
    if type(resolved.get("workers", 12)) is not int or resolved.get("workers", 12) != 12:
        raise ValueError("scientific forecast requires exactly twelve single-thread workers")
    future = resolved["future_seeds"]
    count = 2 if resolved["experiment"] == "e6" else 3
    if (
        len(future) != count
        or len(set(future)) != count
        or any(type(seed) is not int for seed in future)
        or not set(future) <= set(audit["allocation"]["future"]["seeds"])
    ):
        raise ValueError("future RNG identities are incomplete or unaudited")
    if resolved["experiment"] == "e7":
        exploration = resolved["exploration_seeds"]
        if (
            len(exploration) != 128 * 24
            or len(set(exploration)) != len(exploration)
            or not set(exploration) <= set(audit["allocation"]["exploration"]["seeds"])
        ):
            raise ValueError("exploration RNG identities are incomplete or unaudited")
    if resolved["experiment"] == "e6":
        from research.bl_prefix.e6 import _critic_ok, _digest, _roots

        checkpoints = resolved["controllers"]
        if set(checkpoints) != {"R", "F", "B8"} or not _critic_ok(resolved, checkpoints):
            raise ValueError("final E2 endpoints or critic receipts are incomplete")
        B8Controller(checkpoints["B8"])
        for arm in ("R", "F"):
            E2Controller(checkpoints[arm], expected_arm=arm)
        if resolved.get("collection") is not None:
            from research.bl_prefix.bank import _audit, _b8_path, _workers

            collection = resolved["collection"]
            if collection.get("preflight") is not False or collection.get("experiment") not in {
                "bank",
                "e6e7-bank",
            }:
                raise ValueError("scientific collection cannot use preflight fixtures")
            _audit(collection)
            _workers(collection, preflight=False)
            if _digest(_b8_path(collection)) != _digest(checkpoints["B8"]):
                raise ValueError("collection/comparator checkpoint mismatch")
            if _digest(Path(collection["seed_audit"])) != _digest(audit_path):
                raise ValueError("collection/comparator audit mismatch")
        else:
            bank_path = Path(resolved["bank"])
            verify_export(bank_path)
            bank = json.loads(bank_path.read_text())
            if (
                bank.get("preflight") is not False
                or bank.get("audit_sha256") != _digest(audit_path)
                or bank.get("b8", {}).get("sha256") != _digest(checkpoints["B8"])
            ):
                raise ValueError("scientific bank admission identity failed")
            _roots(
                bank,
                bank_path.parent,
                resolved.get("checkpoint_relocations", {}),
                set(audit["allocation"]["source"]["seeds"]),
            )
    else:
        from research.bl_prefix.e6 import _digest
        from research.bl_prefix.e7 import _load_bank

        B8Controller(resolved["b8_checkpoint"])
        _load_bank(
            Path(resolved["bank"]),
            False,
            resolved["b8_checkpoint"],
            _digest(audit_path),
            resolved.get("checkpoint_relocations", {}),
            set(audit["allocation"]["source"]["seeds"]),
        )
