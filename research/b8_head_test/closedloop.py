"""Closed-loop development screen; remote endpoint units stay sealed until Mac readout."""

from __future__ import annotations

import argparse
import hashlib
import json
import signal
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from research.doom_auc_smoke.evidence import atomic, digest, json_once

NAMESPACE = "t34-b8-head-closedloop-20261030/v1"
MARGINS = (0, 10, 25, 50)
ARMS = ("b8", "m0", "m10", "m25", "m50")
MIXES = ("scripted", "frozen", "mixed")


def frozen_features(
    b8: nn.Module, local: torch.Tensor, glob: torch.Tensor, scalar: torch.Tensor
) -> torch.Tensor:
    """Use the fitted head's uint8-to-float normalization at deployment."""
    return b8.fuse(
        torch.cat(
            (b8.local(local.float() / 255), b8.glob(glob.float() / 255), scalar.float()), 1
        )
    )


def seed(domain: str, world: int, opponent: bool = False) -> int:
    """Independent counter-RNG keys; no replacement after a collision."""
    raw = f"{NAMESPACE}/a0/{domain}/{world}/{'opponent' if opponent else 'world'}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") & 0x7FFFFFFFFFFFFFFF


def actions(
    values: torch.Tensor, legal: torch.Tensor, b8: torch.Tensor, margin: torch.Tensor
) -> torch.Tensor:
    """Strict margin in raw mass units over the predicted B8-action component."""
    if values.shape != legal.shape or values.shape != (len(b8), 6):
        raise ValueError("unaligned action table")
    if not legal.gather(1, b8[:, None]).all() or not torch.isfinite(values[legal]).all():
        raise ValueError("illegal B8 action or nonfinite legal predictions")
    masked = values.masked_fill(~legal, -torch.inf)
    best, action = masked.max(1)
    gain = best - values.gather(1, b8[:, None]).flatten()
    return torch.where(gain > margin, action, b8)


def ledger(identity: str) -> dict:
    return dict(namespace=NAMESPACE, allocation_index=0, identity=identity, units=[])


def validate_ledger(value: dict, identity: str, expected: set[str]) -> None:
    if any(value.get(k) != v for k, v in ledger(identity).items() if k != "units"):
        raise ValueError("foreign namespace/allocation/identity")
    if type(value.get("allocation_index")) is not int or not isinstance(value.get("units"), list):
        raise ValueError("malformed ledger")
    seen = set()
    for row in value["units"]:
        unit = row["unit"]
        if unit not in expected or unit in seen or row["path"] != f"units/{unit}.json":
            raise ValueError("foreign or duplicate unit")
        seen.add(unit)
        if (
            type(row["size"]) is not int
            or not 0 < row["size"] <= 1024**2
            or len(row["sha256"]) != 64
        ):
            raise ValueError("invalid descriptor")


def publish(root: Path, unit: str, identity: str, rows: list[dict]) -> dict:
    path = root / f"units/{unit}.json"
    payload = dict(namespace=NAMESPACE, allocation_index=0, identity=identity, unit=unit, rows=rows)
    json_once(path, payload)
    descriptor = dict(
        unit=unit, path=f"units/{unit}.json", size=path.stat().st_size, sha256=digest(path)
    )
    atomic(Path(str(path) + ".sha256"), (descriptor["sha256"] + "\n").encode())
    json_once(Path(str(path) + ".complete"), dict(ledger(identity), **descriptor))
    return descriptor


def verified(root: Path, row: dict, identity: str) -> dict:
    validate_ledger(dict(ledger(identity), units=[row]), identity, {row["unit"]})
    path = root / row["path"]
    if path.is_symlink() or any(p.is_symlink() for p in path.parents):
        raise ValueError("symlink evidence")
    if (
        path.stat().st_size != row["size"]
        or digest(path) != row["sha256"]
        or Path(str(path) + ".sha256").read_text().strip() != row["sha256"]
        or json.loads(Path(str(path) + ".complete").read_text()) != dict(ledger(identity), **row)
    ):
        raise ValueError("unit hash/size/completion differs")
    value = json.loads(path.read_text())
    if any(
        value.get(k) != v
        for k, v in dict(ledger(identity), unit=row["unit"]).items()
        if k != "units"
    ):
        raise ValueError("unit identity differs")
    return value


def initial_hash(sim: Any, opponents: Any, index: int) -> str:
    h = hashlib.sha256()
    for name in (
        "bodies",
        "head_ptr",
        "seg_count",
        "length",
        "alive",
        "direction",
        "boost_frames",
        "frames_since_food",
        "respawn_timer",
        "frame",
        "rng_key",
        "rng_counter",
        "food_cells",
        "food_present",
        "corpse_food",
        "terminal_slots",
    ):
        h.update(name.encode())
        h.update(getattr(sim, name)[index].cpu().contiguous().numpy().tobytes())
    h.update(str(opponents.rosters[index].identity).encode())
    h.update(opponents.rng_key[index].cpu().numpy().tobytes())
    return h.hexdigest()


@torch.inference_mode()
def play(
    ids: list[int],
    *,
    domain: str,
    horizon: int,
    b8: nn.Module,
    head: nn.Module,
    mean: torch.Tensor,
    std: torch.Tensor,
    networks: dict,
    device: str,
    stop: Any,
    progress: Any,
    tiny: bool = False,
) -> list[dict]:
    from research.gpu_sim_spike import r2_cache
    from research.sandbox_a_resident.training_rosters import training_rosters
    from research.spatial_student_pilot.batching import BatchedOpponents, observe_hero_bv8
    from research.spatial_student_pilot.mechanics import PilotCTDeviceSim
    from research.spatial_student_pilot.search import _compact_sim_in_place
    from src.simd_env.batch_sim import BatchSimConfig

    n, count = len(ids), len(ids) * len(ARMS)
    cfg = BatchSimConfig(
        num_envs=count,
        num_snakes=6,
        max_capacity=10400,
        body_storage_capacity=10400,
        mechanics_version=1,
        **(dict(game_width=240, game_height=240, initial_food=10, max_food=18) if tiny else {}),
    )
    sim = PilotCTDeviceSim(
        cfg,
        device=torch.device(device),
        train_mode=False,
        allow_respawn=True,
        move_backend="indexed" if device == "cuda" else "reference",
    )
    r2_cache.enable(sim)
    paired = ids * len(ARMS)
    sim.rng_key.copy_(torch.tensor([seed(domain, i) for i in paired], device=device))
    sim.rng_counter.zero_()
    sim.alive.fill_(False)
    for name in ("length", "seg_count", "head_ptr", "boost_frames", "frames_since_food"):
        getattr(sim, name).zero_()
    sim.terminal_slots.fill_(False)
    sim.reset_fresh_worlds()
    sim.terminal_slots[:, 0] = True
    roster_table = training_rosters(max(3, max(ids) + 1))
    rosters = [roster_table[i] for i in paired]
    opponents = BatchedOpponents(
        sim,
        rosters,
        b8_network=b8,
        checkpoint_networks=networks,
        opp_rng_keys=[seed(domain, i, True) for i in paired],
    )
    hashes = [initial_hash(sim, opponents, i) for i in range(count)]
    if any(hashes[i] != hashes[i % n] for i in range(count)):
        raise ValueError("paired initial state differs")
    total_mass = torch.zeros(count, dtype=torch.float64, device=device)
    total_alive = torch.zeros_like(total_mass)
    deaths = torch.zeros(count, dtype=torch.long, device=device)
    overrides, decisions = torch.zeros_like(deaths), torch.zeros_like(deaths)
    slots = torch.arange(count, device=device)
    arm_margins = torch.tensor([float("inf"), *MARGINS], device=device).repeat_interleave(n)
    last = time.monotonic()
    for tick in range(horizon):
        if stop():
            raise InterruptedError("deadline/signal; no partial endpoint unit")
        if tick % 10 == 0:
            keep = sim.alive[:, 0].nonzero(as_tuple=False).flatten()
            if not keep.numel():
                break
            if keep.numel() < sim.E:
                _compact_sim_in_place(sim, keep)
                opponents.compact_rows(keep)
                slots = slots[keep]
        sim.prepare()
        local, glob, scalar, legal, _, base = observe_hero_bv8(sim, b8)
        phi = frozen_features(b8, local, glob, scalar)
        predicted = head((phi - mean) / std)
        chosen = actions(predicted, legal, base, arm_margins[slots])
        living = sim.alive[:, 0].clone()
        decisions.index_add_(0, slots, living.long())
        overrides.index_add_(0, slots, ((chosen != base) & living).long())
        sim.step(opponents.actions(sim, chosen))
        opponents.after_step(sim)
        torch._assert_async(~sim.capacity_overflow.any(), "body overflow invalidates screen")
        alive = sim.alive[:, 0]
        deaths.index_add_(0, slots, (living & ~alive).long())
        total_mass.index_add_(0, slots, torch.where(alive, sim.length[:, 0], 0).double())
        total_alive.index_add_(0, slots, alive.double())
        if time.monotonic() - last > 60:
            progress(
                dict(
                    stage="gameplay", domain=domain, tick=tick + 1, horizon=horizon, parent_worlds=n
                )
            )
            last = time.monotonic()
    rows = []
    for j, arm in enumerate(ARMS):
        for k, world in enumerate(ids):
            i = j * n + k
            rows.append(
                dict(
                    world_id=world,
                    mix=rosters[k].mix,
                    arm=arm,
                    initial_sha256=hashes[i],
                    mass=float(total_mass[i] / horizon),
                    alive_fraction=float(total_alive[i] / horizon),
                    deaths=int(deaths[i]),
                    decisions=int(decisions[i]),
                    overrides=int(overrides[i]),
                )
            )
    return rows


def statistics(rows: list[dict]) -> dict:
    """Whole paired parents, stratified by mix; equal-weight three-mix effect."""
    table = {(r["world_id"], r["arm"]): r for r in rows}
    if len(table) != len(rows):
        raise ValueError("duplicate world/arm")
    worlds = sorted({r["world_id"] for r in rows})
    if set(table) != {(w, a) for w in worlds for a in ARMS}:
        raise ValueError("incomplete paired arms")
    for w in worlds:
        if len({table[w, a]["initial_sha256"] for a in ARMS}) != 1:
            raise ValueError("unpaired initial states")
    result = {}
    for arm in ARMS:
        output = {}
        for mix in (*MIXES, "all"):
            groups = MIXES if mix == "all" else (mix,)
            rng = np.random.default_rng(343014)
            draws = np.zeros(10000)
            points = []
            selected = []
            for group in groups:
                ws = [w for w in worlds if table[w, "b8"]["mix"] == group]
                if not ws:
                    raise ValueError("missing mix support")
                selected += ws
                delta = np.asarray([table[w, arm]["mass"] - table[w, "b8"]["mass"] for w in ws])
                points.append(float(delta.mean()))
                draws += delta[rng.integers(0, len(ws), (10000, len(ws)))].mean(1) / len(groups)
            rs = [table[w, arm] for w in selected]
            output[mix] = dict(
                worlds=len(rs),
                mass_delta=float(np.mean(points)),
                lb90=float(np.quantile(draws, 0.10)),
                ci95=np.quantile(draws, [0.025, 0.975]).tolist(),
                mass=float(np.mean([r["mass"] for r in rs])),
                deaths=sum(r["deaths"] for r in rs),
                alive_fraction=float(np.mean([r["alive_fraction"] for r in rs])),
                overrides=sum(r["overrides"] for r in rs),
                decisions=sum(r["decisions"] for r in rs),
                override_rate=sum(r["overrides"] for r in rs)
                / max(1, sum(r["decisions"] for r in rs)),
            )
        result[arm] = output
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--mode", choices=("preflight", "canary", "main"), required=True)
    args = parser.parse_args()
    config = json.loads((args.input_dir / "config.json").read_text())
    for name, sha in config["hashes"].items():
        if digest(args.input_dir / name) != sha:
            raise ValueError(f"input hash differs: {name}")
    state = torch.load(args.input_dir / "head.pt", map_location=args.device, weights_only=True)
    head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 6)).to(args.device).eval()
    head.load_state_dict(state["model"])
    from src.model.ego2s_network import load_ego2s_checkpoint
    from src.model.inference_agent import InferenceAgent

    b8 = load_ego2s_checkpoint(str(args.input_dir / "b8-s3.pth"), args.device).eval()
    networks = {
        config["hashes"][name]: InferenceAgent.from_checkpoint(
            str(args.input_dir / name), device=torch.device(args.device)
        ).network.eval()
        for name in ("frp3-s12.pth", "champion.pth")
    }
    domain = args.mode
    worlds, horizon, batch = (192, 5000, 64) if domain == "main" else (3, 64, 3)
    if domain == "preflight":
        horizon = 3
    identity = config["identity"] + "/" + domain
    expected = {f"batch-{i:04d}" for i in range((worlds + batch - 1) // batch)}
    old = (
        json.loads((args.input_dir / "verified-ledger.json").read_text())
        if domain == "main"
        else ledger(identity)
    )
    validate_ledger(old, identity, expected)
    args.output.mkdir(parents=True, exist_ok=False)
    done = {}
    for row in old["units"]:
        done[row["unit"]] = verified(args.input_dir / "resume", row, identity)
    interrupted = False

    def halt(*_: Any) -> None:
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGTERM, halt)
    signal.signal(signal.SIGUSR1, halt)
    deadline = time.monotonic() + (10200 if domain == "main" else 1200)

    def progress(value: dict) -> None:
        print(json.dumps(value), flush=True)

    for index, start in enumerate(range(0, worlds, batch)):
        unit = f"batch-{index:04d}"
        if unit in done:
            continue
        mark = time.monotonic()
        rows = play(
            list(range(start, min(start + batch, worlds))),
            domain=domain,
            horizon=horizon,
            b8=b8,
            head=head,
            mean=state["mean"],
            std=state["std"],
            networks=networks,
            device=args.device,
            stop=lambda: interrupted or time.monotonic() > deadline,
            progress=progress,
            tiny=domain == "preflight",
        )
        publish(args.output, unit, identity, rows)
        progress(dict(stage="complete-unit", unit=unit, seconds=time.monotonic() - mark))
    # This terminal marker contains no efficacy outcomes. Only Mac aggregator opens endpoint rows.
    summary = dict(
        status="COMPLETE",
        mode=domain,
        identity=identity,
        worlds=worlds,
        horizon=horizon,
        arms=list(ARMS),
        units=sorted(expected),
    )
    path = args.output / "summary.json"
    json_once(path, summary)
    atomic(Path(str(path) + ".sha256"), (digest(path) + "\n").encode())
    atomic(Path(str(path) + ".complete"), b"complete\n")


if __name__ == "__main__":
    main()
