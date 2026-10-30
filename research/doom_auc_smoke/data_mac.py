"""Mac-only conversion of existing C/T inputs; assessment labels stay on the Mac."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from research.doom_auc_smoke.evidence import digest
from research.spatial_student_pilot.observation import OBSERVATION_VERSION

BANK_SHA = "112a2aa4c369ecd0e0f735db410e9ac5a7d330aca09846a50d1a734e6b3175f0"
LABEL_SHAS = (
    "3b86db00fdbe11e7c20b82992599599a4265bebf47a68aa94be9b84fccb6162c",
    "44ef1bdf31274585eab2f6130e1913faadbf72f07b16d360a54c23c5e8bc1277",
    "6684e30e752ebbee00c302dd308750edacae648bf96ffc7e96b5704655076186",
    "1b2387507412ced28db804bb35c460d7a8e3a9dd62215691c34943768a1d26e3",
    "6fc27dbd38f1da9d03411e1ca1c61d067663c6d32daf2af20b9f1fdf2807a689",
    "d5f49020006e7b3f07d78ea6cb8b7263dba396714615dbcaf08270913dbd4da1",
    "e28d8637b9cd39501523984d94f7b1588905fa6b50998f2298d766b6aaa707eb",
    "cce958ae9a90b19d4e77b5fa471d5b09f8f08e269baed592f63b0512e2937d0b",
)
B8_SHA = "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"


def compact_snapshot(snapshot: Any, height: int, width: int) -> dict[str, Any]:
    """Head-first C/T segments already encode ring order; never restore/advance a sim."""
    counts = np.diff(snapshot.body_offsets)
    span = max(1, int(counts.max()))
    cells = torch.zeros((1, 6, span, 2), dtype=torch.int16)
    for slot, count in enumerate(counts):
        cells[0, slot, :count] = torch.from_numpy(
            snapshot.body_segments_xy[
                snapshot.body_offsets[slot] : snapshot.body_offsets[slot + 1]
            ].copy()
        )
    small = snapshot.small_tensors
    result = dict(
        version=OBSERVATION_VERSION,
        grid_h=height,
        grid_w=width,
        boost_cost_frames=3,
        body_cells=cells,
        body_count=torch.tensor(counts.copy(), dtype=torch.int32)[None],
        heads=cells[:, 0, 0].clone(),
        frame=small["frame"].reshape(1).clone(),
        food_cells=torch.from_numpy(snapshot.food_xy.copy())[None],
        food_kind=(1 + torch.from_numpy(snapshot.food_corpse.copy()))[None],
    )
    for dest, source in (
        ("length", "length"),
        ("alive", "alive"),
        ("direction", "direction"),
        ("boost_frames", "boost_frames"),
        ("hunger", "frames_since_food"),
    ):
        result[dest] = small[source][None].clone()
    if not snapshot.prepared or snapshot.num_snakes != 6:
        raise ValueError("expected six-snake prepared snapshot")
    return result


def audit_split(metadata: list[Any], partitions: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    if set(partitions) != {"train", "assessment"}:
        raise ValueError("foreign world partition table")
    uids = set()
    for index, meta in enumerate(metadata):
        if (
            meta.root_index != index
            or meta.root_uid in uids
            or not 0 <= meta.world_id < len(partitions)
            or meta.partition != partitions[meta.world_id]
        ):
            raise ValueError("duplicate/misaligned root or world partition leakage")
        uids.add(meta.root_uid)
    return tuple(
        torch.tensor([i for i, m in enumerate(metadata) if m.partition == p])
        for p in ("train", "assessment")
    )


def convert(
    bank: Path, labels: Path, checkpoint: Path
) -> tuple[dict[str, Any], dict[str, Any], dict]:
    # Preparation imports are deliberately local: GPU runtime has no simulator dependencies.
    from research.sandbox_a_resident.ct_snapshots import LosslessBoundedSnapshotStore
    from src.model.ego2s_network import load_ego2s_checkpoint

    if digest(bank) != BANK_SHA or digest(checkpoint) != B8_SHA:
        raise ValueError("foreign bank or B8 checkpoint")
    raw = torch.load(bank, map_location="cpu", weights_only=False)
    data = raw["dataset"]
    if raw["schema"] != "ct-stage-0001/v1" or data["control"]["branch_horizon"] != 300:
        raise ValueError("foreign bank schema/horizon")
    store = LosslessBoundedSnapshotStore.from_state_dict(data["store"])
    if len(store) != 2560:
        raise ValueError("expected all 2560 searched roots")
    train, assessment = audit_split(store.metadata, data["world_partitions"])
    if (len(train), len(assessment)) != (2048, 512):
        raise ValueError("frozen root split differs")
    obs = store.stacked_observations()
    legal, action = obs["legal_mask"].bool(), torch.tensor([m.b8_action for m in store.metadata])
    if not legal.any(1).all() or not legal.gather(1, action[:, None]).all():
        raise ValueError("invalid stored legal/B8 actions")
    train_y, score_y = torch.zeros((2560, 6)), torch.zeros((2560, 6, 4), dtype=torch.bool)
    seen = set()
    for index, sha in enumerate(LABEL_SHAS):
        path = labels / f"stage-{index+1:04d}.pt"
        if digest(path) != sha:
            raise ValueError("foreign scored chunk")
        row = torch.load(path, map_location="cpu", weights_only=True)
        ids = row["root_indices"].long()
        if (
            row["schema"] != "ct-scored-root-chunk/v1"
            or row["chunk_index"] != index
            or row["telemetry"]["horizon"] != 300
            or len(ids) != 320
            or ids.min() < 0
            or ids.max() >= 2560
            or len(set(ids.tolist())) != len(ids)
            or seen.intersection(ids.tolist())
        ):
            raise ValueError("incomplete/duplicate/foreign scored root coverage")
        seen.update(ids.tolist())
        if not torch.equal(row["legal_mask"].bool(), legal[ids]) or not torch.equal(
            row["b8_action"], action[ids]
        ):
            raise ValueError("outcome mask/B8-action differs")
        for field in ("survive_sel_raw", "survive_score_raw"):
            values = row[field]
            if (
                values.shape != (320, 6, 4)
                or not ((values[legal[ids]] == 0) | (values[legal[ids]] == 1)).all()
            ):
                raise ValueError("invalid binary H300 survival outcomes")
        train_y[ids] = (1 - row["survive_sel_raw"].float()).mean(-1)
        score_y[ids] = (1 - row["survive_score_raw"].float()).bool()
    if seen != set(range(2560)):
        raise ValueError("missing scored roots")
    compacts = [compact_snapshot(s, store.grid_h, store.grid_w) for s in store.snapshots]
    for snapshot, meta in zip(store.snapshots, store.metadata):
        if (
            snapshot.world_id != meta.world_id
            or snapshot.life_id != meta.life_id
            or snapshot.tick != meta.tick
            or int(snapshot.small_tensors["length"][0]) != meta.hero_length
            or not np.array_equal(
                np.diff(snapshot.body_offsets), snapshot.small_tensors["seg_count"].numpy()
            )
        ):
            raise ValueError("snapshot/root metadata alignment differs")
    # Config is canonical v1; check rather than assume the observation's cadence.
    from src.simd_env.batch_sim import BatchSimConfig

    if BatchSimConfig(num_envs=1, num_snakes=6).boost_length_cost_frames != 3:
        raise ValueError("canonical boost cadence changed")
    net = load_ego2s_checkpoint(str(checkpoint), "cpu")
    features, q = [], []
    with torch.no_grad():
        for start in range(0, 2560, 64):
            local = obs["local"][start : start + 64].float() / 255
            glob = obs["glob"][start : start + 64].float() / 255
            scalar = obs["scalars"][start : start + 64]
            phi = net.fuse(torch.cat((net.local(local), net.glob(glob), scalar), 1))
            adv = net.adv(phi)
            features.append(phi)
            q.append(net.value(phi) + adv - adv.mean(1, keepdim=True))
    if not torch.allclose(torch.cat(q), obs["q_b8"], atol=1e-3, rtol=0):
        raise ValueError("B8 checkpoint/observation Q parity failed")
    phi = torch.cat(features)
    mean, std = phi[train].mean(0), phi[train].std(0, unbiased=False).clamp_min(1e-6)
    phi = (phi - mean) / std
    runtime = dict(
        schema="doom-dataset/v1",
        compact=compacts,
        phi=phi,
        legal=legal,
        train=train,
        assessment=assessment,
        train_y=train_y[train],
        root_uids=[m.root_uid for m in store.metadata],
    )
    heldout = dict(
        root_uids=[store.metadata[i].root_uid for i in assessment],
        world=torch.tensor([store.metadata[i].world_id for i in assessment]),
        length=torch.tensor([store.metadata[i].hero_length for i in assessment]),
        legal=legal[assessment],
        action=action[assessment],
        doom=score_y[assessment],
    )
    audit = dict(
        bank_sha256=BANK_SHA,
        label_sha256=list(LABEL_SHAS),
        b8_sha256=B8_SHA,
        roots=2560,
        train_roots=len(train),
        assessment_roots=len(assessment),
        assessment_worlds=len(set(heldout["world"].tolist())),
        train_worlds=len({store.metadata[i].world_id for i in train}),
        assessment_labels_uploaded=False,
        anchors_excluded=50000,
        observation_version=OBSERVATION_VERSION,
        grid=[store.grid_h, store.grid_w],
        q_max_abs_error=float((torch.cat(q) - obs["q_b8"]).abs().max()),
    )
    return runtime, heldout, audit
