"""Replay-only sealer review, preserving every original intervention outcome.

The original rollout workers run a frozen implementation. This separately timed
pass applies the reviewed joint-seal classification to exact prepared t0 worlds;
it performs no hero forwards or counterfactual rollouts. Original rows are retained.
"""

from __future__ import annotations

import concurrent.futures
import json
import multiprocessing
import shutil
import time
from pathlib import Path
from typing import Any

from research.b8_death_census_20261014 import census
from research.t1_commitment_20261015.parity import write_json
from research.t1_commitment_20261015.replay import (
    ARTIFACTS,
    build_replay,
    configure_cpu,
    load_inputs,
)
from research.t1_commitment_20261015.topology import sealer_identity


def refresh_arm(arm: str, out: Path) -> dict:
    """Recompute sealers on unchanged recorded paths, with explicit cost receipts."""
    configure_cpu()
    started = time.perf_counter()
    canonical = out / "rows" / f"{arm}.jsonl"
    backup = out / "rows-before-sealer-review" / f"{arm}.jsonl"
    backup.parent.mkdir(parents=True, exist_ok=True)
    if not backup.exists():
        shutil.copyfile(canonical, backup)
    rows = [json.loads(line) for line in backup.read_text().splitlines()]
    _, series = load_inputs(arm)
    context = census.load_context()
    corrections, env_frames = [], 0
    for row in rows:
        onset = row["trap_onset_census"]
        if onset is None:
            continue
        t0, env = int(onset["t0"]), row["env"]
        replay = build_replay(arm, [env], context)
        assert replay.seeds[0] == row["world_seed"]
        before = None
        reviewed = {"owner": "unresolved", "reason": "no previous decision"}
        individual_started = time.perf_counter()
        for t in range(t0 + 1):

            def observe(r: Any) -> None:
                nonlocal before, reviewed
                assert int(r.sim.length[0, 0]) == int(series["length"][env, t]), (arm, env, t)
                assert r.sim.alive[0, 0]
                if t == t0 - 1:
                    before = r.clone(prepared=True)
                if t == t0 and before is not None:
                    reviewed = sealer_identity(before.sim, r.sim, r.kinds)

            replay.step([int(series["action"][env, t])], observe)
        extra_seconds = time.perf_counter() - individual_started
        corrections.append(
            {
                "env": env,
                "t0": t0,
                "old": row["sealer"],
                "new": reviewed,
                "classification_changed": row["sealer"]["owner"] != reviewed["owner"],
                "replayed_env_frames": replay.env_frames,
            }
        )
        row["sealer"] = reviewed
        row["sealer_review_replay_env_frames"] = replay.env_frames
        row["env_frames_before_sealer_review"] = row["env_frames"]
        row["env_frames"] += replay.env_frames
        row["wall_seconds_before_sealer_review"] = row["wall_seconds"]
        row["wall_seconds"] += extra_seconds
        env_frames += replay.env_frames
        print(
            json.dumps({"sealer_review": arm, "env": env, "owner": reviewed["owner"]}), flush=True
        )
    temporary = canonical.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    temporary.replace(canonical)
    receipt = {
        "status": "PASS",
        "arm": arm,
        "rows": len(rows),
        "reviewed": len(corrections),
        "env_frames": env_frames,
        "wall_seconds": time.perf_counter() - started,
        "corrections": corrections,
    }
    write_json(out / "sealer-review" / f"{arm}.json", receipt)
    return receipt


def main() -> None:
    out = ARTIFACTS
    run = json.loads((out / "run.json").read_text())
    assert run["status"] == "PASS" and not run["partial_pilot"]
    start = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=2, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        jobs = [pool.submit(refresh_arm, arm, out) for arm in census.B8_ARMS]
        receipts = [job.result() for job in jobs]
    write_json(
        out / "sealer-review.json",
        {
            "status": "PASS",
            "workers": 2,
            "wall_seconds": time.perf_counter() - start,
            "env_frames": sum(r["env_frames"] for r in receipts),
            "arms": receipts,
            "git": census._git(),
            "topology_sha256": census.sha256_file(Path(__file__).with_name("topology.py")),
            "rollout_launch_commit": (out / "launch-head.txt").read_text().strip(),
        },
    )


if __name__ == "__main__":
    main()
