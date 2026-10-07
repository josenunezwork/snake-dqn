"""In-process Apex actor episodes for the perf-sim identity harness, profiler and bench.

Two worlds, chosen by what the ``--root`` source tree has:

* ``--world gate``: the FRP-v3 gate-world training actor (``src/training/gate_world.py``,
  branches frp-v3 / frp-v4): deployment world, mix-sampled strict rosters, hero on v8 with
  veto-routed exploration, hero-only transitions. Each episode is forced to a (world seed,
  mix) pair from this harness's own namespace (``perf-sim-identity/v1``), horizon
  ``--horizon``.
* ``--world train``: the default Apex actor arena (``ApexActor.run`` with ``max_episodes``),
  as on main.

The actor is the real ``ApexActor`` object driven in-process (no subprocess, no IPC) with a
recording buffer client; the network weights are the frp3-s12 checkpoint's. ``identity``
mode writes, per episode, digests of every row the actor sends (state / next-state float32
bytes, action, reward, done, bootstrap steps, next-action mask and mode, priority) plus the
per-frame world digest stream; ``time`` mode only times (``--profile`` adds cProfile).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import episode_harness as eh  # noqa: E402

FRP3_CONFIG = "research/frp_v3_20261005/configs/arm_M3.yaml"
TRAIN_CONFIG = "configs/free_space_v2.yaml"


def _row_digest(h: "hashlib._Hash", value: Any) -> None:
    import numpy as np
    import torch

    if value is None:
        h.update(b"N")
    elif isinstance(value, torch.Tensor):
        h.update(value.detach().cpu().numpy().tobytes())
        h.update(repr((str(value.dtype), tuple(value.shape))).encode())
    elif isinstance(value, np.ndarray):
        h.update(value.tobytes())
        h.update(repr((str(value.dtype), value.shape)).encode())
    else:
        h.update(repr(value).encode())


class DigestClient:
    """Stand-in for ``ActorBufferClient``: digests every row sent (in order)."""

    def __init__(self) -> None:
        self.h = hashlib.blake2b(digest_size=16)
        self.rows = 0
        self.actions: List[int] = []

    def add_batch(self, **batch: Any) -> None:
        keys = sorted(batch)
        n = len(batch["rewards"])
        for i in range(n):
            for key in keys:
                column = batch[key]
                self.h.update(key.encode())
                _row_digest(self.h, None if column is None else column[i])
            self.actions.append(int(batch["actions"][i]))
            self.rows += 1

    def get_stats(self) -> Dict[str, Any]:
        return {}

    def flush(self) -> None:
        return None

    def take(self) -> Dict[str, Any]:
        out = {
            "rows": self.rows,
            "rows_digest": self.h.hexdigest(),
            "actions_digest": hashlib.sha256(bytes(self.actions)).hexdigest()[:24],
        }
        self.h = hashlib.blake2b(digest_size=16)
        self.rows = 0
        self.actions = []
        return out


def build_actor(world: str, horizon: int, actor_id: int, client: Any) -> Any:
    import torch

    from research.apex_safety_20260926 import dev_screen
    from src.model.apex_network import ApexNetwork
    from src.training.apex_actor import ApexActor

    blob = torch.load(eh.checkpoint_path(*eh.FRP3_S12), map_location="cpu", weights_only=False)
    net = ApexNetwork(61, 512, 6)
    net.load_state_dict(blob["dqn_state_dict"])
    kwargs: Dict[str, Any] = dict(
        actor_id=actor_id,
        num_actors=5,
        shared_network=net,
        buffer_client=client,
        weight_queue=queue.Queue(),
        stats_queue=queue.Queue(),
        stop_event=threading.Event(),
        gamma=0.99,
        n_step=3,
        base_epsilon=0.1,
        epsilon_alpha=1.0,
        boost_exploration_rate=0.0,
        danger_exploration_rate=0.0,
        base_seed=20261007,
    )
    if world == "gate":
        from src.training.gate_world import GateWorldSpec

        spec = GateWorldSpec(
            horizon=int(horizon),
            pool=tuple((eh.checkpoint_path(name, sha), sha) for name, sha in dev_screen.POOL),
            seed_domain=eh.DOMAIN,
            seed_purpose="actor{actor_id}",
            roster_width=5,
            veto="v8",
            veto_lambda=8.0,
            explore_through_veto=True,
            profile_digest=dev_screen.PROFILE_DIGEST,
            explore_filter="route",
            roster_mode="mix_sampled",
        )
        kwargs.update(config_path=FRP3_CONFIG, gate_world=spec.to_dict())
    else:
        kwargs.update(config_path=TRAIN_CONFIG)
    return ApexActor(**kwargs)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--world", choices=("gate", "train"), required=True)
    parser.add_argument("--mode", choices=("identity", "time"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--horizon", type=int, default=1500, help="gate world horizon")
    parser.add_argument("--actor-id", type=int, default=0, help="0 = epsilon 0.1")
    parser.add_argument("--namespace", default="actor", help="gate world seed namespace")
    parser.add_argument("--profile", type=Path, default=None)
    args = parser.parse_args(argv)

    out = args.out.resolve()
    prof_path = args.profile.resolve() if args.profile else None
    eh._bootstrap(args.root)
    import torch

    torch.set_num_threads(1)
    client = DigestClient()
    actor = build_actor(args.world, args.horizon, args.actor_id, client)
    tracer = eh.Tracer() if args.mode == "identity" else None
    if tracer:
        tracer.install()
    profiler = None
    if prof_path is not None:
        import cProfile

        profiler = cProfile.Profile()
    out.parent.mkdir(parents=True, exist_ok=True)
    mixes = ("frozen", "scripted", "mixed")
    with out.open("w", encoding="utf-8") as stream:
        if args.world == "gate":
            actor._initialize_runtime()
            seeds = eh.world_seeds(args.episodes, namespace=args.namespace)
            for index, seed in enumerate(seeds):
                mix = mixes[index % 3]
                started, cpu0 = time.perf_counter(), time.process_time()
                if profiler:
                    profiler.enable()
                _, steps, leftovers = actor._run_gate_world_episode(
                    stream_to_buffer=True, world_seed=int(seed), gate_mix=mix
                )
                if leftovers:
                    actor._send_experience_batch(leftovers)
                if profiler:
                    profiler.disable()
                entry = {
                    "episode": index,
                    "mix": mix,
                    "world_seed": int(seed),
                    "steps": int(steps),
                    "wall_seconds": time.perf_counter() - started,
                    "cpu_seconds": time.process_time() - cpu0,
                    "transitions": client.take(),
                }
                if tracer:
                    entry["trace"] = tracer.take()
                stream.write(json.dumps(entry, sort_keys=True) + "\n")
                stream.flush()
                print(
                    f"episode {index} {mix} steps={steps} {entry['wall_seconds']:.2f}s",
                    file=sys.stderr,
                    flush=True,
                )
        else:
            actor.max_episodes = int(args.episodes)
            actor.log_interval = 10**9
            started, cpu0 = time.perf_counter(), time.process_time()
            if profiler:
                profiler.enable()
            actor.run()
            if profiler:
                profiler.disable()
            entry = {
                "episodes": int(args.episodes),
                "wall_seconds": time.perf_counter() - started,
                "cpu_seconds": time.process_time() - cpu0,
                "agent_transitions": int(actor.agent_transition_count),
                "transitions": client.take(),
            }
            if tracer:
                entry["trace"] = tracer.take()
            stream.write(json.dumps(entry, sort_keys=True) + "\n")
    if profiler:
        profiler.dump_stats(str(prof_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
