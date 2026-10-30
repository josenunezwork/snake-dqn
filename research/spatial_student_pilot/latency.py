"""Mac observation+actor probe from immutable, naturally reached assessment roots.

This measures live Snake/FoodManager extraction, CPU raster reconstruction,
input uploads, actor forward, logits readback and masked action selection.
It excludes opponents, dynamics and rendering, so it is not full-tick or v2
product qualification. Snapshots become real GameState/Snake/FoodManager
objects for this static read-only probe; they are not resumed live episodes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from research.sandbox_a_resident.ct_snapshots import CompactWorldSnapshot
from research.spatial_student_pilot.collect import select_dataset
from research.spatial_student_pilot.config import PilotConfig
from research.spatial_student_pilot.model import SpatialStudent, load_checkpoint
from research.spatial_student_pilot.observation import (
    build_observation_numpy,
    compact_from_live,
    compact_from_sim,
)

DIAGNOSTIC_PROTOCOL = {
    "id": "spatial-latency-amendment-04/v1",
    "device": "mps",
    "precision": "fp32",
    "threads": 1,
    "warmups_per_fixture": 12,
    "repeats_per_fixture": 100,
    "load": "single session; ordinary desktop load retained; no competing research job started",
    "scope": "live extraction+CPU observation+uploads+actor+readback+masked action",
}
DIAGNOSTIC_SOFTWARE = {
    "system": "Darwin",
    "platform": "macOS-27.2-arm64-arm-64bit",
    "python": "3.12.3",
    "torch": "2.9.1",
    "numpy": "1.26.4",
}


def source_identity() -> dict[str, str]:
    """Bind the frozen observation and actor implementation used by the probe."""
    return {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ("latency.py", "model.py", "observation.py")
    }


def verify_diagnostic(
    report: Mapping[str, Any],
    cfg: PilotConfig,
    roots: Sequence[Mapping[str, Any]],
    checkpoint_sha256: str,
    metadata: Mapping[str, Any],
) -> float:
    """Reject incomplete/foreign diagnostics before the scientific readout.

    A qualification flag or a scalar percentile cannot replace raw evidence.
    The historical 4ms comparison is reported, never an admission threshold.
    """
    expected = {
        **DIAGNOSTIC_SOFTWARE,
        "schema": "spatial-student-latency/v1",
        "system": "Darwin",
        "device": "mps",
        "threads": 1,
        "fixture_count": 128,
        "warmups_per_fixture": 12,
        "repeats_per_fixture": 100,
        "scope": DIAGNOSTIC_PROTOCOL["scope"],
        "diagnostic_protocol": DIAGNOSTIC_PROTOCOL,
        "diagnostic_complete": True,
        "source_sha256": source_identity(),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_metadata": dict(metadata),
        "config_sha256": cfg.sha256,
        "namespace": cfg.namespace,
        "allocation_index": cfg.allocation_index,
        "root_uids": [r["uid"] for r in roots],
        "parent_count": len({r["world_id"] for r in roots}),
    }
    # JSON comparison also rejects booleans masquerading as numeric identities
    # (Python otherwise considers True == 1), and nonfinite nested fields.
    actual_identity = {key: report.get(key) for key in expected}
    if len(roots) != 128 or json.dumps(
        actual_identity, sort_keys=True, allow_nan=False
    ) != json.dumps(expected, sort_keys=True, allow_nan=False):
        raise ValueError("latency diagnostic identity/protocol mismatch")
    for key in ("platform", "torch", "python", "numpy", "load_observation"):
        if not isinstance(report.get(key), str) or not report[key].strip():
            raise ValueError("latency diagnostic missing software/load identity")
    raw = report.get("sample_ms")
    if (
        not isinstance(raw, list)
        or len(raw) != 12800
        or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in raw)
    ):
        raise ValueError("malformed or incomplete raw latency timings")
    samples = np.asarray(raw, dtype=float)
    if not np.isfinite(samples).all() or (samples < 0).any():
        raise ValueError("nonfinite or negative latency timings")
    pooled = _stats(raw)
    per_fixture = [_stats(raw[i : i + 100]) for i in range(0, 12800, 100)]
    if json.dumps(
        [report.get("statistics"), report.get("per_fixture")], sort_keys=True, allow_nan=False
    ) != json.dumps([pooled, per_fixture], sort_keys=True, allow_nan=False):
        raise ValueError("latency statistics differ from retained raw timings")
    if report.get("within_p95_limit") is not (pooled["p95_ms"] <= 4.0):
        raise ValueError("latency historical comparison differs from raw timings")
    return pooled["p95_ms"]


def snapshot_to_live(snapshot: dict[str, Any], cfg: PilotConfig) -> Any:
    """Restore observed state into actual live object classes, without policies/RNG.

    The GameState constructor would initialize six unrelated Apex policies.
    Static restoration deliberately allocates the real GameState class without
    that constructor, and fills every field consumed by compact_from_live.
    This object is only an observation fixture, never a runnable game clone.
    """
    from src.game.food_manager import FoodManager
    from src.game.game_state import GameState
    from src.game.snake import Snake

    snap = CompactWorldSnapshot.from_state_dict(snapshot)
    sim_cfg = cfg.sim_config(1)
    if snap.num_snakes != 6:
        raise ValueError("live probe requires six snakes")
    small = snap.small_tensors
    game = GameState.__new__(GameState)
    game.cfg = sim_cfg
    game._game_width, game._game_height = sim_cfg.game_width, sim_cfg.game_height
    game.frame = int(small["frame"])
    game.num_snakes = 6
    game.headless = True
    game.snakes = []
    cardinal = ((0, -1), (1, 0), (0, 1), (-1, 0))
    for slot in range(6):
        start, end = map(int, snap.body_offsets[slot : slot + 2])
        cells = snap.body_segments_xy[start:end]
        if end - start != int(small["seg_count"][slot]):
            raise ValueError("snapshot body count identity failure")
        if len(cells) == 0 and bool(small["alive"][slot]):
            raise ValueError("living snapshot snake has no head")
        pixels = [tuple(int(v) * sim_cfg.segment_size for v in xy) for xy in cells]
        snake = Snake(
            id=slot,
            color=(slot * 35, 200, 100),
            start_pos=pixels[0] if pixels else (0, 0),
            segment_size=sim_cfg.segment_size,
            game_width=sim_cfg.game_width,
            game_height=sim_cfg.game_height,
            food_capacity=sim_cfg.max_food,
        )
        snake.segments = pixels
        snake.length = int(small["length"][slot])
        snake.is_alive = bool(small["alive"][slot])
        snake.direction = cardinal[int(small["direction"][slot])]
        snake.boost_frames = int(small["boost_frames"][slot])
        snake.frames_since_food = int(small["frames_since_food"][slot])
        snake.respawn_timer = int(small["respawn_timer"][slot])
        snake.is_boosting = bool(small.get("_boosted_this_step", torch.zeros(6))[slot])
        game.snakes.append(snake)
    fm = FoodManager(
        game_width=sim_cfg.game_width,
        game_height=sim_cfg.game_height,
        initial_food=0,
        max_food=sim_cfg.max_food,
        segment_size=sim_cfg.segment_size,
        wall_thickness=sim_cfg.wall_thickness,
    )
    fm.food = [tuple(int(v) * sim_cfg.segment_size for v in xy) for xy in snap.food_xy]
    fm._corpse_positions = {pos for pos, corpse in zip(fm.food, snap.food_corpse) if corpse}
    game.food_manager = fm
    game.alive_snakes = sum(s.is_alive for s in game.snakes)
    return game


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _stats(values: list[float]) -> dict[str, Any]:
    return {
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "max_ms": float(max(values)),
        "samples": len(values),
    }


def _probe(
    fixtures: Sequence[Any],
    student: SpatialStudent,
    *,
    device: str,
    warmups: int,
    repeats: int,
    live: bool,
    legal_masks: Sequence[Sequence[bool]] | None = None,
) -> dict[str, Any]:
    if not fixtures or min(warmups, repeats) < 1:
        raise ValueError("nonempty fixtures and positive warmups/repeats required")
    target = torch.device(device)
    student = student.to(target).eval()
    masks = (
        np.ones((len(fixtures), 6), dtype=bool)
        if legal_masks is None
        else np.asarray(legal_masks, dtype=bool)
    )
    if masks.shape != (len(fixtures), 6) or not masks.any(axis=1).all():
        raise ValueError("one nonempty legal mask per fixture required")
    samples, by_fixture = [], []
    with torch.inference_mode():
        for row, game in enumerate(fixtures):
            times = []
            for repeat in range(warmups + repeats):
                _sync(target)
                started = time.perf_counter_ns()
                compact = compact_from_live(game) if live else compact_from_sim(game, [0])
                inputs = build_observation_numpy(compact, target)
                # cpu() is the measured synchronization/readback boundary.
                logits = student(*inputs).float().cpu().numpy()[0]
                if not np.isfinite(logits).all():
                    raise ValueError("nonfinite logits in live latency probe")
                action = int(np.where(masks[row], logits, -np.inf).argmax())
                if not masks[row, action]:
                    raise ValueError("illegal action in live latency probe")
                elapsed = (time.perf_counter_ns() - started) / 1e6
                if repeat >= warmups:
                    times.append(elapsed)
            by_fixture.append(_stats(times))
            samples.extend(times)
    return {
        "schema": "spatial-student-latency/v1",
        "device": str(target),
        "platform": platform.platform(),
        "system": platform.system(),
        "python": platform.python_version(),
        "numpy": str(np.__version__),
        "torch": str(torch.__version__),
        "threads": torch.get_num_threads(),
        "fixture_count": len(fixtures),
        "warmups_per_fixture": warmups,
        "repeats_per_fixture": repeats,
        "scope": (
            "live extraction+CPU observation+uploads+actor+readback+masked action"
            if live
            else (
                "sim extraction+CPU observation+uploads+actor+readback+masked action; "
                "development only"
            )
        ),
        "full_tick_qualified": False,
        "threshold_qualified": False,
        "p95_limit_ms": 4.0,
        "statistics": _stats(samples),
        "per_fixture": by_fixture,
        "sample_ms": samples,
    }


def probe_live(
    games: Sequence[Any],
    student: SpatialStudent,
    *,
    device: str = "mps",
    warmups: int = 12,
    repeats: int = 100,
    legal_masks: Sequence[Sequence[bool]] | None = None,
) -> dict[str, Any]:
    """Measure complete static live path; caller must verify root provenance."""
    return _probe(
        games,
        student,
        device=device,
        warmups=warmups,
        repeats=repeats,
        live=True,
        legal_masks=legal_masks,
    )


def probe_sim(
    sims: Sequence[Any],
    student: SpatialStudent,
    *,
    device: str = "cpu",
    warmups: int = 2,
    repeats: int = 3,
) -> dict[str, Any]:
    """Tiny real-entrypoint development probe; never qualifies the 4ms threshold."""
    return _probe(sims, student, device=device, warmups=warmups, repeats=repeats, live=False)


def selected_assessment_roots(
    cfg: PilotConfig, batches: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Reproduce the frozen dataset selection from all complete collection units."""
    if any(batch.get("schema") != "spatial-collection/v1" for batch in batches):
        raise ValueError("latency requires verified collection units")
    dataset = select_dataset(cfg, batches)
    roots = [r for r in dataset["roots"] if r["partition"] == "assessment" and r["kind"] == "late"]
    if len({r["uid"] for r in roots}) != len(roots):
        raise ValueError("duplicate assessment latency root")
    parents: dict[int, int] = {}
    for root in roots:
        parent = int(root["world_id"])
        snap = root["snapshot"]
        if (
            not 0 <= parent < cfg.worlds
            or cfg.partitions()[parent] != "assessment"
            or int(snap["world_id"]) != parent
        ):
            raise ValueError("foreign assessment parent")
        if (
            not bool(snap["small_tensors"]["alive"][0])
            or int(snap["small_tensors"]["length"][0]) < 600
        ):
            raise ValueError("latency root is not naturally reached living L600+")
        parents[parent] = parents.get(parent, 0) + 1
    if (
        len(roots) != cfg.assessment_late
        or len(parents) < 64
        or max(parents.values(), default=0) > 2
    ):
        raise ValueError("latency requires all128 late roots from >=64 parents, max2 per parent")
    return roots


def load_final_candidate(
    checkpoint: Path, cfg: PilotConfig, device: str = "cpu"
) -> tuple[SpatialStudent, dict[str, Any], str]:
    """Require a sealed final8000-update candidate bound to the frozen allocation."""
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    marker = Path(str(checkpoint) + ".complete")
    sidecar = Path(str(checkpoint) + ".sha256")
    if not marker.is_file() or marker.read_bytes() != b"complete\n":
        raise ValueError("missing final candidate completion marker")
    words = sidecar.read_text().split() if sidecar.is_file() else []
    if not words or words[0] != digest:
        raise ValueError("final candidate digest mismatch")
    student, metadata = load_checkpoint(checkpoint, device)
    expected = {
        "config_sha256": cfg.sha256,
        "seed": cfg.seed,
        "updates": cfg.bc_updates + cfg.improvement_updates,
        "namespace": cfg.namespace,
    }
    if cfg.mode != "pilot" or cfg.tiny or any(metadata.get(k) != v for k, v in expected.items()):
        raise ValueError("latency candidate differs from frozen final8000-update allocation")
    return student, metadata, digest


def development_bank_probe(
    path: Path, *, device: str, warmups: int, repeats: int
) -> dict[str, Any]:
    """Historical fixtures plus random weights; never final-candidate qualification."""
    from research.sandbox_a_resident.ct_snapshots import LosslessBoundedSnapshotStore

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if Path(str(path) + ".sha256").read_text().split()[0] != digest:
        raise ValueError("historical fixture bank SHA mismatch")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if blob.get("schema") != "ct-stage-0001/v1":
        raise ValueError("expected historical C/T collection stage")
    store = LosslessBoundedSnapshotStore.from_state_dict(blob["dataset"]["store"])
    eligible = sorted(
        (
            i
            for i, snap in enumerate(store.snapshots)
            if bool(snap.small_tensors["alive"][0]) and int(snap.small_tensors["length"][0]) >= 600
        ),
        key=lambda i: (-int(store.snapshots[i].small_tensors["length"][0]), i),
    )
    indices, seen = [], set()
    for i in eligible:
        parent = store.metadata[i].world_id
        if parent not in seen:
            indices.append(i)
            seen.add(parent)
        if len(indices) == 16:
            break
    if len(indices) < 16:
        raise ValueError("need16 distinct natural long-body historical parents")
    cfg = PilotConfig()
    games = [snapshot_to_live(store.snapshots[i].to_state_dict(), cfg) for i in indices]
    torch.manual_seed(cfg.seed)
    report = probe_live(games, SpatialStudent(), device=device, warmups=warmups, repeats=repeats)
    report.update(
        development_only=True,
        random_initialization=True,
        fixture_bank_sha256=digest,
        fixture_bank=str(path),
        root_indices=indices,
        hero_lengths=[int(store.snapshots[i].small_tensors["length"][0]) for i in indices],
        within_p95_limit=report["statistics"]["p95_ms"] <= 4.0,
        source_sha256={
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("latency.py", "model.py", "observation.py")
        },
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--units-root", type=Path)
    parser.add_argument("--unit-id", action="append")
    parser.add_argument("--development-bank", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--warmups", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--load-observation", help="document desktop load for the frozen session")
    args = parser.parse_args(argv)
    if platform.system() != "Darwin":
        raise ValueError("latency must be measured on the owner's Mac")
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    if args.development_bank is not None:
        if any((args.checkpoint, args.units_root, args.unit_id, args.config)):
            raise ValueError("development bank cannot be combined with scientific inputs")
        report = development_bank_probe(
            args.development_bank, device=args.device, warmups=args.warmups, repeats=args.repeats
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
        print(json.dumps(report["statistics"], sort_keys=True), flush=True)
        return 0
    if not all((args.checkpoint, args.units_root, args.unit_id, args.config)):
        parser.error("scientific probe requires checkpoint, units-root, unit-id and config")
    if (
        args.device != "mps"
        or args.warmups != 12
        or args.repeats != 100
        or not args.load_observation
        or not args.load_observation.strip()
    ):
        parser.error("diagnostic requires MPS, exactly12 warmups/100 repeats and load-observation")
    from research.spatial_student_pilot.resume import load_verified_unit

    cfg = PilotConfig.from_dict(json.loads(args.config.read_text()))
    if cfg.mode != "pilot" or cfg.tiny:
        raise ValueError("threshold probe requires the registered production allocation")
    if platform.system() != "Darwin":
        raise ValueError("the registered latency threshold must be measured on the owner's Mac")
    if args.output.exists():
        raise FileExistsError(args.output)
    batches = []
    for unit_id in args.unit_id:
        payload = load_verified_unit(args.units_root, cfg, unit_id)
        batches.append(payload)
    roots = selected_assessment_roots(cfg, batches)
    torch.set_num_threads(1)
    student, metadata, checkpoint_digest = load_final_candidate(args.checkpoint, cfg, args.device)
    games = [snapshot_to_live(r["snapshot"], cfg) for r in roots]
    for root, game in zip(roots, games):
        source_inputs = build_observation_numpy(root["compact"])
        restored_inputs = build_observation_numpy(compact_from_live(game))
        if not all(torch.equal(a, b) for a, b in zip(source_inputs, restored_inputs)):
            raise ValueError("snapshot/live observation identity failure")
    report = probe_live(
        games,
        student,
        device=args.device,
        warmups=args.warmups,
        repeats=args.repeats,
        legal_masks=[r["mask"] for r in roots],
    )
    report.update(
        {
            "namespace": cfg.namespace,
            "config_sha256": cfg.sha256,
            "allocation_index": cfg.allocation_index,
            "checkpoint_metadata": metadata,
            "checkpoint_sha256": checkpoint_digest,
            "root_uids": [r["uid"] for r in roots],
            "parent_count": len({r["world_id"] for r in roots}),
            "diagnostic_complete": True,
            "diagnostic_protocol": DIAGNOSTIC_PROTOCOL,
            "source_sha256": source_identity(),
            "load_observation": args.load_observation,
            "within_p95_limit": report["statistics"]["p95_ms"] <= 4.0,
        }
    )
    verify_diagnostic(report, cfg, roots, checkpoint_digest, metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
