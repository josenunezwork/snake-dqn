"""Actual default Watch6 constructor and independently checked Grid mirror."""

from __future__ import annotations

import copy
import os
import random
import tarfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np


def open_watch(row: dict, fps: int, arm: str):
    """Use the shipped product configuration, never the gate parity config."""
    from research.bl_served_search.b2_engine import FRP3
    from research.bl_served_search.ceiling import B8_SHA256
    from research.t3_rollout_labels_20261015.bank_engine import CapturedWorld
    from research.t15_mixed_roster_20261017.episodes import new_session
    from research.t34_pathb.adapter_v2 import RunSession
    from src.core.game_config import GameConfig
    from src.simd_env.ego2s_policy import Ego2sSimdPolicy
    from src.simd_env.eval_engine import _config_from_game_config
    from src.simd_env.grid_sim import GridBatchSim
    from src.simd_env.vector61_policy import Vector61Runtime, Vector61SimdPolicy
    from tools.ops.package_spec import digest
    from web.backend import served_checkpoint

    saved = Path(os.environ["SNAKE_CKPT_DIR"])
    pins = served_checkpoint.load_pins()
    evidence_root = os.environ.get("WATCH6_RELEASE_EVIDENCE")
    if evidence_root:
        pins = copy.deepcopy(pins)
        for name, pin in pins["checkpoints"].items():
            pin["strict_receipt"]["root"] = str(Path(evidence_root) / name)
    with (
        patch.object(served_checkpoint, "SAVED_DIR", str(saved)),
        patch.object(served_checkpoint, "load_pins", return_value=pins),
        patch.dict(
            os.environ,
            {"SNAKE_SERVE_PATHB": "off", "SNAKE_MECHANICS_V2": "1", "SNAKE_WATCH_SOURCE": "live"},
        ),
    ):
        live = new_session(int(row["seed"]), 6)
    live.set_speed(fps or 24)
    live.policy.forward = "rowwise"
    if GameConfig.MECHANICS_VERSION != 2 or GameConfig.NUM_SNAKES != 6:
        raise ValueError("requires actual product Watch6 mechanics v2")
    b8 = Path(live.served_checkpoint["path"])
    frp3 = Path(live.watch_opponent_checkpoint["path"])
    if digest(b8) != B8_SHA256 or digest(frp3) != FRP3:
        raise ValueError("product checkpoint identity mismatch")
    rng = random.getstate()
    # Storage only: full session plus an H300 continuation can grow at most
    # one segment/frame. No physical length limit or world knob is changed.
    cfg = replace(_config_from_game_config(6, 1.0), max_capacity=14704)
    sim = GridBatchSim(cfg, seeds=[int(row["seed"])], train_mode=False, allow_respawn=True)
    runtime = Vector61Runtime(backend="numba")
    runtime.bind(sim, np.array([[False, True, True, True, True, True]]))
    hero = Ego2sSimdPolicy(str(b8), device="cpu", forward="rowwise", mask_mode="advisory")
    opponents = [
        Vector61SimdPolicy(
            str(frp3),
            runtime,
            veto_slots=(slot,),
            forward="rowwise",
            veto_variant="v8",
            veto_lambda=8.0,
        )
        for slot in range(1, 6)
    ]
    world = CapturedWorld(
        sim, runtime, hero, opponents, int(row["seed"]), "mixed", ["checkpoint"] * 5
    )
    random.setstate(rng)
    return RunSession(world, arm, live=live, specs=[("checkpoint", str(frp3))] * 5)


def stage_release_evidence(input_root: Path) -> None:
    """Unpack only the six exact immutable release anchors, never bypass pins."""
    import hashlib

    from web.backend import served_checkpoint

    pins = served_checkpoint.load_pins()
    expected = {}
    for name, pin in pins["checkpoints"].items():
        for ref in pin["strict_receipt"].values():
            if isinstance(ref, dict):
                expected[name + "/" + ref["path"]] = ref["sha256"]
    output = input_root / "release-evidence"
    output.mkdir(exist_ok=False)
    with tarfile.open(input_root / "release-evidence.tar.gz", "r:gz") as archive:
        if {m.name for m in archive.getmembers()} != set(expected):
            raise ValueError("release evidence inventory mismatch")
        for member in archive.getmembers():
            if not member.isfile() or member.size > 2**21:
                raise ValueError("unsafe release evidence")
            data = archive.extractfile(member).read()
            if hashlib.sha256(data).hexdigest() != expected[member.name]:
                raise ValueError("release evidence digest mismatch")
            target = output / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
