"""Optional train-only lossless integer raster cache (ideas 01.1 and 09.1)."""

from __future__ import annotations

from typing import Any, Sequence

import torch

from research.spatial_student_pilot.observation import LENGTH_SCALE, build_observation

from .search import tensor_identity


class IntegerRasterCache:
    """Eight binary planes in a byte, six clearance numerators in uint16.

    Built on the runtime from compact rows; never uploaded or used to store
    held-out data. Byte-verify every encoding before admitting it. Geometry is
    shared per shape; scalars/heads retain their exact original types and bytes.
    """

    def __init__(self, rows: Sequence[dict[str, Any]], device: str, max_bytes: int) -> None:
        self.device = device
        self.entries: dict[str, tuple[Any, ...]] = {}
        identities: dict[str, str] = {}
        self.geometry: dict[tuple[int, int], torch.Tensor] = {}
        self.bytes = 0
        for row in rows:
            if row["partition"] != "train" or row["parent_partition"] != "train":
                raise ValueError("training cache refuses held-out ancestry")
            key = row["uid"]
            content = tensor_identity(row["compact"])
            if key in self.entries:
                if identities[key] != content:
                    raise ValueError("cache UID has conflicting compact bytes")
                continue
            identities[key] = content
            planes, scalars, heads = build_observation(row["compact"], device=device)
            binary = planes[:, [0, 1, 2, 3, 4, 5, 12, 13]].to(torch.uint8)
            shifts = torch.arange(8, device=device).view(1, 8, 1, 1)
            bits = (binary.long() << shifts).sum(1).to(torch.uint8)
            numerators = (planes[:, 6:12] * LENGTH_SCALE).round().to(torch.uint16)
            shape = tuple(planes.shape[-2:])
            geometry = planes[:, 14:16].clone()
            if shape in self.geometry and not torch.equal(self.geometry[shape], geometry):
                raise ValueError("cache geometry differs for identical shape")
            tensors = [bits, numerators, scalars, heads]
            if shape not in self.geometry:
                tensors.append(geometry)
            required = sum(t.numel() * t.element_size() for t in tensors)
            if self.bytes + required > max_bytes:
                raise ValueError("frozen integer-cache byte cap exceeded")
            self.geometry.setdefault(shape, geometry)
            self.entries[key] = (bits, numerators, scalars, heads, shape)
            self.bytes += required
            actual = self._decode(self.entries[key])
            if not all(torch.equal(a, b) for a, b in zip(actual, (planes, scalars, heads))):
                raise ValueError("integer cache failed byte-identity admission")

    def _decode(self, entry: tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
        bits, numerators, scalars, heads, shape = entry
        shifts = torch.arange(8, device=self.device).view(1, 8, 1, 1)
        binary = ((bits[:, None].long() >> shifts) & 1).float()
        clearance = numerators.float() / LENGTH_SCALE
        planes = torch.cat((binary[:, :6], clearance, binary[:, 6:], self.geometry[shape]), 1)
        return planes, scalars, heads

    def batch(self, rows: Sequence[dict[str, Any]]) -> tuple[torch.Tensor, ...]:
        decoded = [self._decode(self.entries[row["uid"]]) for row in rows]
        return tuple(torch.cat([r[i] for r in decoded]) for i in range(3))
