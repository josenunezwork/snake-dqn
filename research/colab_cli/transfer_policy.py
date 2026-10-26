"""Prospective, hash-bound transfer allowances; staging stays independent.

Inventory identities below are transcribed from the immutable d1024-pack-02
canary manifest, not inferred from fit outcomes. The acceptance uses synthetic
bytes with exactly these roles/sizes and cannot qualify a different inventory.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

RULING_SHA256 = "f57080bdc08b6f0dc439adb0082ea5e00725d0cfd2521fdaffde6b864f1cc809"
F5_TIMING_RULING_SHA256 = "4f10f264fda3c7190e07f486803fa02cc10645741fb95ef215b1d1b0cc123439"
INVENTORY_SHA256 = "b9eca552d1dc72349e7e303859a87da6317ce0abc0d5635dcecfb9a7151c72ea"
DEFAULT_TRANSFER_SECONDS = 150.0
EXCEPTION_TRANSFER_SECONDS = 300.0
MAX_STAGING_SECONDS = 300.0
FIT_SCRIPT = "research/bl_residual/f4_d256_colab.sh"
ACCEPTANCE_SCRIPT = "research/colab_cli/acceptance_probe.sh"
F5_TIMING_SCRIPT = "research/bl_residual/f5_colab.sh"
B8_URL = (
    "https://github.com/josenunezwork/snake-dqn/releases/download/"
    "s1-inputs-20261019/B8_s3_final.pth"
)
# dest: (bytes, production SHA); synthetic roles retain size but not payload SHA.
FROZEN_INPUTS = {
    "b8.pth": (6_562_005, "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"),
    "factory-verified-ledger.json": (
        2_405_362,
        "f5ec94ccdfae90133362a9072227d83125ae0ce661c44471904ce57333b610b8",
    ),
    "train-1024.npz": (
        45_397_643,
        "6906bffc7241de568965c1a91f4faebe799895ee0b4650e59b20e43db1dac494",
    ),
    "metadata.tar": (2_211_840, "d37d1a0426321502f4ef316f1f79ebedf5a77328e69829cf0f11880635b944f2"),
}


def authority_binding(context: str = "fit") -> dict[str, Any]:
    """Return an explicit immutable policy declaration, never inferred consent."""
    if not isinstance(context, str) or context not in {"fit", "acceptance"}:
        raise ValueError("foreign D1024 transfer exception context")
    return {
        "schema": "f4-d1024-prospective-transfer-exception/v1",
        "ruling_sha256": RULING_SHA256,
        "inventory_sha256": INVENTORY_SHA256,
        "context": context,
        "base_transfer_seconds": DEFAULT_TRANSFER_SECONDS,
        "effective_transfer_seconds": EXCEPTION_TRANSFER_SECONDS,
        "whole_staging_seconds": MAX_STAGING_SECONDS,
    }


def f5_timing_authority_binding(inventory_sha256: str, context: str = "timing") -> dict[str, Any]:
    """Return the narrow F5 recovery allowance for one sealed inventory.

    Unlike the historical F4 exception, this binding names a successor's exact
    frozen input inventory.  It is deliberately a separate manifest field so
    the F4 D1024 path cannot borrow it (and vice versa).
    """
    if context not in {"timing", "canary", "acceptance"} or not re.fullmatch(
        r"[0-9a-f]{64}", inventory_sha256
    ):
        raise ValueError("foreign F5 timing transfer exception context or inventory")
    return {
        "schema": "f5-timing-recovery-transfer-exception/v1",
        "ruling_sha256": F5_TIMING_RULING_SHA256,
        "inventory_sha256": inventory_sha256,
        "context": context,
        "base_transfer_seconds": DEFAULT_TRANSFER_SECONDS,
        "effective_transfer_seconds": EXCEPTION_TRANSFER_SECONDS,
        "whole_staging_seconds": MAX_STAGING_SECONDS,
    }


def f5_timing_recovery_binding(inventory_sha256: str, context: str = "timing") -> dict[str, Any]:
    """Named public constructor for the F5 timing04 recovery authority."""
    return f5_timing_authority_binding(inventory_sha256, context)


F5_TIMING_NAMESPACE = "bl-residual-f5-timing-recovery-20261026"
_F5_TIMING_BASE_DESTINATIONS = {
    "b8.pth",
    "timing-fixtures.json",
    "champion_a5_freespace_20260621.pth",
    "best_apex_fs.pth",
    "best_apex_stage1_fs.pth",
    "best_apex_pre_fs.pth",
    "frp3_m3_s12_u60000_20261005.pth",
    "control.json",
    "verified-ledger.json",
    "verified-ledger.json.sha256",
    "verified-ledger.json.complete",
}
_F5_TIMING_COMPLETED_UNIT_DESTINATION = re.compile(
    r"timing-units/(?:fixture-(?:67[2-9]|68[0-3])|calibration)\.json(?:\.(?:sha256|complete))?"
)


def validate_f5_timing_inventory(
    inventory: dict[str, Any],
    *,
    expected_commit: str | None = None,
    expected_tree: str | None = None,
) -> None:
    """Validate sealed F5 recovery metadata before it controls a filesystem path.

    The authority admits precisely the recovery's base inputs and, on a resumed
    package, whole deterministic unit payload/sidecar triples.  It cannot turn
    a timing allowance into a selected-science transport allowance.
    """
    supplied = inventory.get("sha256")
    unsigned = dict(inventory)
    unsigned.pop("sha256", None)
    canonical = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    rows = inventory.get("inputs")
    if (
        inventory.get("schema") != "f5-timing-recovery-synthetic-input-inventory/v1"
        or inventory.get("ruling_sha256") != F5_TIMING_RULING_SHA256
        or inventory.get("namespace") != F5_TIMING_NAMESPACE
        or supplied != canonical
        or not re.fullmatch(r"[0-9a-f]{40}", str(inventory.get("source_commit", "")))
        or not re.fullmatch(r"[0-9a-f]{40}", str(inventory.get("source_tree", "")))
        or not isinstance(rows, list)
        or not rows
        or any(not isinstance(row, dict) for row in rows)
    ):
        raise ValueError("F5 timing transfer exception inventory is malformed")
    if expected_commit is not None and inventory["source_commit"] != expected_commit:
        raise ValueError("F5 timing transfer exception source commit differs")
    if expected_tree is not None and inventory["source_tree"] != expected_tree:
        raise ValueError("F5 timing transfer exception source tree differs")
    names = [row.get("dest") for row in rows]
    if any(not isinstance(name, str) or not name for name in names) or len(names) != len(
        set(names)
    ):
        raise ValueError("F5 timing transfer exception inventory destinations differ")
    names_set = set(names)
    extra = names_set - _F5_TIMING_BASE_DESTINATIONS
    if not _F5_TIMING_BASE_DESTINATIONS <= names_set or any(
        _F5_TIMING_COMPLETED_UNIT_DESTINATION.fullmatch(name) is None for name in extra
    ):
        raise ValueError("F5 timing transfer exception inventory roles differ")
    completed_roots = {name.removesuffix(".sha256").removesuffix(".complete") for name in extra}
    if any({root, root + ".sha256", root + ".complete"} - names_set for root in completed_roots):
        raise ValueError("F5 timing transfer exception completed unit sidecars differ")
    for row in rows:
        if (
            row.get("transport") not in {"upload", "https"}
            or type(row.get("size_bytes")) is not int
            or row["size_bytes"] <= 0
            or not isinstance(row.get("sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
        ):
            raise ValueError("F5 timing transfer exception inventory input is malformed")
        if row["transport"] == "https" and not isinstance(row.get("url"), str):
            raise ValueError("F5 timing transfer exception public input is malformed")
    if sum(row["size_bytes"] for row in rows if row["transport"] == "upload") > 45_397_643:
        raise ValueError("F5 timing transfer exception exceeds the private input byte cap")


def _f5_inventory(manifest: dict[str, Any], binding: dict[str, Any]) -> None:
    inventory = manifest.get("f5_timing_synthetic_inventory")
    if not isinstance(inventory, dict):
        raise ValueError("F5 timing transfer exception requires a sealed inventory")
    validate_f5_timing_inventory(
        inventory,
        expected_commit=manifest.get("commit"),
        expected_tree=manifest.get("f5_timing_source_tree"),
    )
    if inventory["sha256"] != binding["inventory_sha256"]:
        raise ValueError("F5 timing transfer exception inventory digest differs")
    rows = inventory["inputs"]
    names = [row["dest"] for row in rows]
    actual = {row.get("dest"): row for row in manifest.get("inputs", []) if isinstance(row, dict)}
    if not set(names).issubset(actual):
        raise ValueError("F5 timing transfer exception input destinations differ")
    context = binding["context"]
    allowed_extra = {"acceptance_probe.py", "acceptance_probe.sh"}
    if context == "acceptance":
        # The combined final-only transport qualification intentionally includes
        # its independent D1024 synthetic inventory under the old bar.
        allowed_extra |= set(FROZEN_INPUTS)
    if context != "acceptance" and set(actual) != set(names):
        raise ValueError("F5 timing transfer exception has unbound input destinations")
    if context == "acceptance" and not set(actual) <= set(names) | allowed_extra:
        raise ValueError("F5 timing acceptance has unbound input destinations")
    for row in rows:
        observed = actual[row["dest"]]
        if observed.get("size_bytes") != row["size_bytes"]:
            raise ValueError("F5 timing transfer exception input size differs")
        if row["transport"] == "https" and (
            observed.get("url") != row.get("url") or observed.get("sha256") != row["sha256"]
        ):
            raise ValueError("F5 timing transfer exception public input differs")
        if row["transport"] == "upload":
            path = Path(str(observed.get("path", "")))
            if path.is_symlink() or not path.is_file() or path.stat().st_size != row["size_bytes"]:
                raise ValueError("F5 timing transfer exception private input is missing or unsafe")
            expected = (
                hashlib.sha256(bytes(row["size_bytes"])).hexdigest()
                if context == "acceptance"
                else row["sha256"]
            )
            with path.open("rb") as stream:
                observed_digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if observed.get("sha256") != expected or observed_digest != expected:
                raise ValueError("F5 timing transfer exception private input hash differs")


def _f5_timing_limit(manifest: dict[str, Any], binding: dict[str, Any]) -> float:
    context = binding.get("context")
    if binding != f5_timing_authority_binding(binding.get("inventory_sha256", ""), context):
        raise ValueError("foreign or malformed F5 timing transfer exception authority")
    requirements = manifest.get("requirements", {})
    if requirements.get("gpu") != "L4" or requirements.get("allowed_gpus", ["L4"]) != ["L4"]:
        raise ValueError("F5 timing transfer exception is L4-only")
    _f5_inventory(manifest, binding)
    env = manifest.get("env", {})
    if context == "acceptance":
        if (
            manifest.get("script") != ACCEPTANCE_SCRIPT
            or env.get("COLAB_ACCEPTANCE") != "1"
            or env.get("COLAB_ACCEPTANCE_PROFILE") != "final-only"
            or env.get("COLAB_ACCEPTANCE_ARM") != "L4"
        ):
            raise ValueError("F5 timing transfer exception requires final-only L4 acceptance")
    elif (
        manifest.get("script") != F5_TIMING_SCRIPT
        or manifest.get("device") != "cuda:0"
        or env.get("F5_DEVICE") != "cuda:0"
        or env.get("F5_MODE")
        != ("qualification-canary" if context == "canary" else "qualification")
    ):
        raise ValueError("F5 timing transfer exception requires timing-only L4 qualification")
    if context == "canary":
        canary = manifest.get("canary", {})
        if (
            not isinstance(canary, dict)
            or canary.get("mode") != "probe"
            or env.get("SNAKE_COLAB_CANARY") != "1"
            or env.get("SNAKE_COLAB_CANARY_PACKAGE") != "research/bl_residual"
        ):
            raise ValueError("F5 timing transfer exception requires the family canary")
    return EXCEPTION_TRANSFER_SECONDS


def _inventory(manifest: dict[str, Any], *, synthetic: bool) -> None:
    rows = manifest.get("inputs")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("D1024 transfer exception requires the immutable input inventory")
    by_dest = {row.get("dest"): row for row in rows}
    extras = {"acceptance_probe.py", "acceptance_probe.sh"} if synthetic else set()
    if len(by_dest) != len(rows) or set(by_dest) != set(FROZEN_INPUTS) | extras:
        raise ValueError("D1024 transfer exception input destinations differ")
    for dest, (size, sha) in FROZEN_INPUTS.items():
        row = by_dest[dest]
        if dest == "b8.pth":
            if row.get("url") != B8_URL or "path" in row or row.get("sha256") != sha:
                raise ValueError("D1024 transfer exception public B8 identity differs")
            observed_size = row.get("size_bytes")
        else:
            if "url" in row or not isinstance(row.get("path"), str):
                raise ValueError("D1024 transfer exception requires private uploads")
            path = Path(row["path"])
            if path.is_symlink() or not path.is_file():
                raise ValueError("D1024 transfer exception input is missing or unsafe")
            observed_size = path.stat().st_size
            if not re.fullmatch(r"[0-9a-f]{64}", str(row.get("sha256", ""))):
                raise ValueError("D1024 transfer exception requires bound input hashes")
            if not synthetic and row["sha256"] != sha:
                raise ValueError("D1024 transfer exception immutable input hash differs")
            if synthetic:
                # Sparse all-zero acceptance bytes, not arbitrary same-size data.
                expected = hashlib.sha256()
                block = bytes(1 << 20)
                remaining = size
                while remaining:
                    count = min(remaining, len(block))
                    expected.update(block[:count])
                    remaining -= count
                if row["sha256"] != expected.hexdigest():
                    raise ValueError("D1024 transfer exception synthetic input hash differs")
        if type(observed_size) is not int or observed_size != size:
            raise ValueError("D1024 transfer exception immutable input size differs")


def transfer_limit(manifest: dict[str, Any]) -> float:
    """Validate every claimed exception before returning its transfer limit.

    Absence grants no exception. A present declaration must exactly match this
    ruling and its lane; unrelated jobs may never opt into the larger allowance.
    """
    f5_binding = manifest.get("f5_timing_transfer_exception")
    if f5_binding is not None:
        if "transfer_exception" in manifest or not isinstance(f5_binding, dict):
            raise ValueError("F5 timing transfer exception cannot combine with another authority")
        return _f5_timing_limit(manifest, f5_binding)
    if "transfer_exception" not in manifest:
        return DEFAULT_TRANSFER_SECONDS
    binding = manifest["transfer_exception"]
    if not isinstance(binding, dict):
        raise ValueError("missing or malformed D1024 transfer exception authority")
    context = binding.get("context")
    if (
        not isinstance(context, str)
        or context not in {"fit", "acceptance"}
        or binding != authority_binding(context)
    ):
        raise ValueError("foreign or malformed D1024 transfer exception authority")
    requirements = manifest.get("requirements", {})
    if requirements.get("gpu") != "L4" or requirements.get("allowed_gpus", ["L4"]) != ["L4"]:
        raise ValueError("D1024 transfer exception is L4-only")
    env = manifest.get("env", {})
    if context == "acceptance":
        if (
            manifest.get("script") != ACCEPTANCE_SCRIPT
            or env.get("COLAB_ACCEPTANCE") != "1"
            or env.get("COLAB_ACCEPTANCE_PROFILE") != "final-only"
            or env.get("COLAB_ACCEPTANCE_ARM") != "L4"
        ):
            raise ValueError("D1024 transfer exception requires final-only L4 acceptance")
        _inventory(manifest, synthetic=True)
    else:
        mode = env.get("F4_MODE")
        f4 = manifest.get("f4", {})
        request = manifest.get("canary", {})
        probe = isinstance(request, dict) and request.get("mode") == "probe"
        excluded = (
            env.get("SNAKE_COLAB_CANARY") == "1"
            and env.get("SNAKE_COLAB_CANARY_PACKAGE") == "research/bl_residual"
        )
        if (
            manifest.get("script") != FIT_SCRIPT
            or env.get("F4_DOSE") != "1024"
            or env.get("F4_DEVICE") != "cuda:0"
            or manifest.get("device") != "cuda:0"
            or mode not in {"canary", "main"}
            or (mode == "canary" and (not probe or not excluded))
            or (mode == "main" and (probe or bool(env.get("SNAKE_COLAB_CANARY"))))
            or type(f4.get("dose")) is not int
            or f4.get("dose") != 1024
            or type(f4.get("registered_updates")) is not int
            or f4.get("registered_updates") != 2000
            or env.get("METADATA_ARCHIVE_SHA256") != FROZEN_INPUTS["metadata.tar"][1]
            or not re.fullmatch(
                rf"f4-d1024-{'canary' if mode == 'canary' else 'fit'}-[0-9a-f]{{8}}-[0-9]{{2}}",
                str(manifest.get("job_id", "")),
            )
        ):
            raise ValueError("D1024 transfer exception F4 package/mode/dose/recipe differs")
        _inventory(manifest, synthetic=False)
    return EXCEPTION_TRANSFER_SECONDS


def acceptance_limit(plan: dict[str, Any]) -> float:
    """Revalidate an exception from its saved plan and exact L4 manifest."""
    f5_binding = plan.get("f5_timing_transfer_exception")
    if f5_binding is not None:
        if "transfer_exception" in plan or not isinstance(f5_binding, dict):
            raise ValueError("F5 timing acceptance has conflicting transfer authority")
        inventory = plan.get("f5_timing_synthetic_inventory")
        if not isinstance(inventory, dict) or plan.get("profile") != "final-only":
            raise ValueError("F5 timing transfer exception requires final-only inventory")
        jobs = plan.get("jobs", [])
        if len(jobs) != 2 or {job.get("name") for job in jobs} != {"cpu", "l4"}:
            raise ValueError("F5 timing transfer exception requires exactly CPU/L4 acceptance")
        cap = None
        for job in jobs:
            manifest = json.loads(Path(job.get("manifest", "")).read_text())
            if manifest.get("commit") != plan.get("commit"):
                raise ValueError("F5 timing transfer exception acceptance commit differs")
            if job["name"] == "l4":
                if manifest.get("f5_timing_transfer_exception") != f5_binding:
                    raise ValueError("F5 timing acceptance manifest authority differs")
                if manifest.get("f5_timing_synthetic_inventory") != inventory:
                    raise ValueError("F5 timing acceptance manifest inventory differs")
                cap = transfer_limit(manifest)
            elif any(
                key in manifest for key in ("transfer_exception", "f5_timing_transfer_exception")
            ):
                raise ValueError("CPU acceptance cannot claim an L4 transfer exception")
        if cap is not None:
            return cap
        raise ValueError("F5 timing acceptance lacks L4 manifest")
    if "transfer_exception" not in plan:
        for job in plan.get("jobs", []):
            path = Path(job.get("manifest", ""))
            if path.is_file():
                manifest = json.loads(path.read_text())
                if "transfer_exception" in manifest:
                    raise ValueError("D1024 acceptance manifest has undeclared transfer authority")
                if "f5_timing_transfer_exception" in manifest:
                    raise ValueError(
                        "F5 timing acceptance manifest has undeclared transfer authority"
                    )
        return DEFAULT_TRANSFER_SECONDS
    if plan["transfer_exception"] != authority_binding("acceptance"):
        raise ValueError("foreign or malformed D1024 acceptance transfer authority")
    inventory = plan.get("d1024_synthetic_inventory", {})
    if plan.get("profile") != "final-only" or inventory.get("sha256") != INVENTORY_SHA256:
        raise ValueError("D1024 transfer exception requires its final-only inventory")
    jobs = plan.get("jobs", [])
    if len(jobs) != 2 or {job.get("name") for job in jobs} != {"cpu", "l4"}:
        raise ValueError("D1024 transfer exception requires exactly CPU/L4 acceptance")
    cap = DEFAULT_TRANSFER_SECONDS
    for job in jobs:
        manifest = json.loads(Path(job["manifest"]).read_text())
        if manifest.get("commit") != plan.get("commit"):
            raise ValueError("D1024 transfer exception acceptance commit differs")
        if job["name"] == "l4":
            if manifest.get("transfer_exception") != plan["transfer_exception"]:
                raise ValueError("D1024 acceptance manifest transfer authority differs")
            cap = transfer_limit(manifest)
        elif any(key in manifest for key in ("transfer_exception", "f5_timing_transfer_exception")):
            raise ValueError("CPU acceptance cannot claim an L4 transfer exception")
    return cap


def receipt_limit(manifest: dict[str, Any], receipt: dict[str, Any], output: Path) -> float:
    """Bind a claimed exception to the broker's actual immutable job manifest."""
    planned = transfer_limit(manifest)
    authority_key = (
        "f5_timing_transfer_exception"
        if "f5_timing_transfer_exception" in manifest
        else "transfer_exception"
    )
    if authority_key not in manifest:
        return planned
    saved = output / "manifest.json"
    if saved.is_symlink() or not saved.is_file():
        raise ValueError("transfer exception actual manifest is absent or unsafe")
    raw = saved.read_bytes()
    if hashlib.sha256(raw).hexdigest() != receipt.get("manifest_sha256"):
        raise ValueError("transfer exception actual manifest hash differs from receipt")
    actual = json.loads(raw)
    if (
        actual.get("commit") != manifest.get("commit")
        or actual.get("commit") != receipt.get("commit")
        or actual.get("job_id") != manifest.get("job_id")
        or actual.get("job_id") != receipt.get("job_id")
        or actual.get(authority_key) != manifest[authority_key]
    ):
        raise ValueError("transfer exception actual manifest authority/identity differs")
    observed = transfer_limit(actual)
    if observed != planned:
        raise ValueError("transfer exception actual manifest allowance differs")
    return observed
