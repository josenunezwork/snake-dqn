"""Follow-on unit envelope retaining actual hardware alongside immutable payloads."""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Any

import torch

from research.spatial_student_pilot.resume import UnitStore


def hardware(device: str) -> dict[str, Any]:
    dev = torch.device(device)
    if dev.type == "cuda":
        props = torch.cuda.get_device_properties(dev)
        return dict(
            device=str(dev),
            gpu_sku=torch.cuda.get_device_name(dev),
            capability=list(torch.cuda.get_device_capability(dev)),
            total_memory=props.total_memory,
            torch_version=str(torch.__version__),
            cuda_version=torch.version.cuda,
        )
    return dict(
        device=str(dev),
        gpu_sku=None,
        cpu=platform.machine(),
        torch_version=str(torch.__version__),
        cuda_version=torch.version.cuda,
    )


class HardwareUnitStore:
    """Wrap each publication without changing pilot ledger or payload semantics.

    Mixed GPU fit resume is rejected: preserving RNG/optimizer alone does not
    prove bitwise continuation across SKUs. Other units retain their own SKU.
    """

    def __init__(self, store: UnitStore, device: str) -> None:
        self.store = store
        self.runtime = hardware(device)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.store, name)

    def get(self, unit_id: str) -> Any | None:
        value = self.store.get(unit_id)
        if value is None:
            return None
        if (
            not isinstance(value, dict)
            or value.get("schema") != "spatial-exit-unit/v2"
            or "hardware" not in value
        ):
            raise ValueError("unit lacks per-unit hardware provenance")
        if unit_id.startswith("fit-") and value["hardware"] != self.runtime:
            raise ValueError("fit resume requires the same hardware/software identity")
        return value["payload"]

    def publish(self, unit_id: str, payload: Any) -> Path:
        return self.store.publish(
            unit_id, dict(schema="spatial-exit-unit/v2", hardware=self.runtime, payload=payload)
        )
