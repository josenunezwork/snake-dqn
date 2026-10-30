"""Public synthetic full-board late-body fixtures for tests and family canary."""

from __future__ import annotations

import torch

from research.spatial_student_pilot.observation import OBSERVATION_VERSION


def synthetic_dataset(height: int = 83, width: int = 145) -> dict:
    generator = torch.Generator().manual_seed(1234)
    compacts = []
    for i in range(12):
        span = min(600, height * width)
        order = torch.arange(span)
        cells = torch.stack((order % width, order // width), -1).short()
        body = cells[None, None].repeat(1, 6, 1, 1)
        compacts.append(
            dict(
                version=OBSERVATION_VERSION,
                grid_h=height,
                grid_w=width,
                boost_cost_frames=5,
                body_cells=body,
                body_count=torch.full((1, 6), span, dtype=torch.int32),
                length=torch.full((1, 6), max(600, span)),
                alive=torch.ones((1, 6), dtype=torch.bool),
                direction=torch.zeros((1, 6), dtype=torch.long),
                boost_frames=torch.zeros((1, 6)),
                hunger=torch.zeros((1, 6)),
                frame=torch.tensor([i]),
                heads=body[:, 0, 0].clone(),
                food_cells=torch.tensor([[[1, 1]]], dtype=torch.int16),
                food_kind=torch.ones((1, 1), dtype=torch.uint8),
            )
        )
    return dict(
        compact=compacts,
        phi=torch.randn((12, 256), generator=generator),
        legal=torch.ones((12, 6), dtype=torch.bool),
        train=torch.arange(8),
        assessment=torch.arange(8, 12),
        train_y=torch.randint(2, (8, 6), generator=generator).float(),
        root_uids=[f"synthetic-{i}" for i in range(12)],
    )
