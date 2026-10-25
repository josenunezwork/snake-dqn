"""Prospective, hash-bound F4 D1024 transfer allowance; staging stays independent.

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
INVENTORY_SHA256 = "b9eca552d1dc72349e7e303859a87da6317ce0abc0d5635dcecfb9a7151c72ea"
DEFAULT_TRANSFER_SECONDS = 150.0
EXCEPTION_TRANSFER_SECONDS = 300.0
MAX_STAGING_SECONDS = 300.0
FIT_SCRIPT = "research/bl_residual/f4_d256_colab.sh"
ACCEPTANCE_SCRIPT = "research/colab_cli/acceptance_probe.sh"
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
    if "transfer_exception" not in plan:
        for job in plan.get("jobs", []):
            path = Path(job.get("manifest", ""))
            if path.is_file() and "transfer_exception" in json.loads(path.read_text()):
                raise ValueError("D1024 acceptance manifest has undeclared transfer authority")
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
        elif "transfer_exception" in manifest:
            raise ValueError("CPU acceptance cannot claim the D1024 L4 transfer exception")
    return cap


def receipt_limit(manifest: dict[str, Any], receipt: dict[str, Any], output: Path) -> float:
    """Bind a claimed exception to the broker's actual immutable job manifest."""
    planned = transfer_limit(manifest)
    if "transfer_exception" not in manifest:
        return planned
    saved = output / "manifest.json"
    if saved.is_symlink() or not saved.is_file():
        raise ValueError("D1024 transfer exception actual manifest is absent or unsafe")
    raw = saved.read_bytes()
    if hashlib.sha256(raw).hexdigest() != receipt.get("manifest_sha256"):
        raise ValueError("D1024 transfer exception actual manifest hash differs from receipt")
    actual = json.loads(raw)
    if (
        actual.get("commit") != manifest.get("commit")
        or actual.get("commit") != receipt.get("commit")
        or actual.get("job_id") != manifest.get("job_id")
        or actual.get("job_id") != receipt.get("job_id")
        or actual.get("transfer_exception") != manifest["transfer_exception"]
    ):
        raise ValueError("D1024 transfer exception actual manifest authority/identity differs")
    observed = transfer_limit(actual)
    if observed != planned:
        raise ValueError("D1024 transfer exception actual manifest allowance differs")
    return observed
