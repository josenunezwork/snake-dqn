#!/usr/bin/env python3
"""Build a fan-out job file from a registered wrapper's world recipe.

Worlds are ``dev_screen.screen_seeds(count, domain, namespace)`` (default: the wrapper
module's ``DOMAIN`` and ``"worlds"``) with the strict balanced rosters of
``dev_screen._design_rows``, i.e. exactly the worlds/rosters a local screen plays.

Example::

    ./venv/bin/python -m research.runpod_fanout.jobgen --job-id v8probe-a \
        --wrapper apex-veto-v8-screen-v1 --arms A,B --mixes scripted,frozen,mixed \
        --worlds 4 --horizon 1000 --engine live --max-wall-minutes 60 \
        --purpose "Tier-1 probe" --out /tmp/job.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.runpod_fanout import jobspec  # noqa: E402
from research.runpod_fanout.wrappers import (  # noqa: E402
    HERO_SHA256,
    SCRIPTED_SHA256,
    WRAPPERS,
)


def build_job(a: argparse.Namespace) -> dict:
    from research.apex_safety_20260926 import dev_screen as ds

    wrapper = WRAPPERS[a.wrapper]
    domain = a.domain or importlib.import_module(wrapper["module"]).DOMAIN
    seeds = ds.screen_seeds(a.world_start + a.worlds, domain, a.namespace)
    index = {s: i for i, s in enumerate(seeds)}
    mixes = [m for m in a.mixes.split(",") if m]
    arms = [x for x in a.arms.split(",") if x]
    rows = [r for r in ds._design_rows(seeds[a.world_start :]) if r["mix"] in mixes]
    episodes, used = [], {HERO_SHA256}
    for row in rows:
        roster = [slot["member_sha256"] for slot in row["slots"]]
        used.update(s for s in roster if s not in SCRIPTED_SHA256)
        for arm in arms:
            episodes.append(
                {
                    "wrapper": a.wrapper,
                    "arm": arm,
                    "mix": row["mix"],
                    "world_seed": int(row["world_seed"]),
                    "world_index": index[int(row["world_seed"])],
                    "horizon": a.horizon,
                    "engine": a.engine,
                    "roster_member_sha256s": roster,
                }
            )
    commit = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", a.commit],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return {
        "schema": jobspec.JOB_SCHEMA,
        "job_id": a.job_id,
        "tier": "tier1",
        "purpose": a.purpose,
        "repo_commit": commit,
        "max_wall_minutes": a.max_wall_minutes,
        "checkpoints": sorted(used),
        "episodes": episodes,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--job-id", required=True)
    p.add_argument("--wrapper", required=True, choices=sorted(WRAPPERS))
    p.add_argument("--arms", required=True)
    p.add_argument("--mixes", default="scripted")
    p.add_argument("--worlds", type=int, required=True, help="worlds per mix")
    p.add_argument("--world-start", type=int, default=0)
    p.add_argument("--domain", default=None)
    p.add_argument("--namespace", default="worlds")
    p.add_argument("--horizon", type=int, default=5000)
    p.add_argument("--engine", choices=jobspec.ENGINES, default="live")
    p.add_argument("--commit", default="HEAD")
    p.add_argument("--max-wall-minutes", type=float, default=60)
    p.add_argument("--purpose", required=True)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args(argv)
    job = build_job(a)
    jobspec.validate_job(job, jobspec.load_policy(), jobspec.load_allowlist())
    with a.out.open("x") as fh:
        json.dump(job, fh, indent=1)
        fh.write("\n")
    print(json.dumps(jobspec.summarize_job(job), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
