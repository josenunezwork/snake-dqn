"""Geometry boundary tests for Q0's ego-raster open-state definition."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from research.q0q1_20261017.audit_episode_metrics import locally_open
from src.simd_env.ego_raster import EgoRasterConfig, _offset_table


class _Sim:
    def __init__(self, direction: int, *, alive: bool = True) -> None:
        self.cfg = EgoRasterConfig(version="bv8")
        self.pad = 1
        self.head_col = 100
        self.head_row = 100
        self.owner = np.full((1, 202, 202), -1, dtype=np.int16)
        self.view = SimpleNamespace(
            owner_pad=self.owner,
            alive=np.array([[alive, True]], dtype=bool),
            heads=np.array([[[self.head_col, self.head_row], [0, 0]]], dtype=np.int64),
            direction=np.array([[direction, 0]], dtype=np.int64),
            pad=self.pad,
        )

    def ego_view(self):
        return self.view

    def place_at_local_pixel(self, pixel: int, owner: int = 1) -> None:
        stride = self.owner.shape[2]
        center = (self.head_row + self.pad) * stride + self.head_col + self.pad
        flat = self.owner[0].reshape(-1)
        flat[center + pixel] = owner

    def place_ahead(self, distance: int, owner: int = 1) -> None:
        from src.simd_env.batch_sim import CARDINAL

        dx, dy = CARDINAL[int(self.view.direction[0, 0])]
        self.owner[
            0, self.head_row + self.pad + dy * distance, self.head_col + self.pad + dx * distance
        ] = owner


@pytest.mark.parametrize("heading", range(4))
@pytest.mark.parametrize("pixel", (15, 30 * 31 + 15, 23 * 31, 23 * 31 + 30))
def test_locally_open_uses_each_inclusive_ego_raster_boundary(heading: int, pixel: int) -> None:
    """Ahead 23, behind 7, and lateral 15 boundaries are all in the crop."""
    sim = _Sim(heading)
    table = _offset_table(31, 23, 15, sim.owner.shape[2])
    sim.place_at_local_pixel(int(table[heading, pixel]))
    assert not locally_open(sim, 0, 0)


@pytest.mark.parametrize("heading", range(4))
def test_locally_open_excludes_cell_just_beyond_forward_ego_extent(heading: int) -> None:
    sim = _Sim(heading)
    sim.place_ahead(24)
    assert locally_open(sim, 0, 0)


def test_locally_open_ignores_hero_body_but_requires_a_living_hero() -> None:
    sim = _Sim(0)
    table = _offset_table(31, 23, 15, sim.owner.shape[2])
    sim.place_at_local_pixel(int(table[0, 15]), owner=0)
    assert locally_open(sim, 0, 0)
    assert not locally_open(_Sim(0, alive=False), 0, 0)
