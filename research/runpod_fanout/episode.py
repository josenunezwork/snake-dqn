#!/usr/bin/env python3
"""Run ONE fan-out episode and write its record (pod side and local side, same code).

Usage (from a checkout or ``git archive`` of the job's commit)::

    python -m research.runpod_fanout.episode --spec SPEC.json --out-root DIR \
        --ckpt-dir DIR --provider runpod:cpu3c --job-id ID --commit SHA

``SPEC.json`` is one job-file episode. Live episodes call the wrapper's own
``dev_screen.run_episode`` with the wrapper's ``ScreenSpec``, so the entry equals what a
local screen writes; the file additionally carries ``platform`` and ``fanout`` stamps.
SIMD episodes mirror the RunPod x86 check's ``run_simd_eval`` call. A horizon below
5000 uses the promotion-v2-watch-rect profile with only ``scored_horizon`` shortened
(its digest therefore differs from the pinned H5000 digest; it is recorded).

The record is written create-only to ``<out-root>/<group>/<arm>-<mix>-<seed>.json``.
If that file already exists, the new result must equal it after
:func:`research.runpod_fanout.jobspec.deterministic_bytes` (exit 0) or the run exits 3.
"""

from __future__ import annotations

import os

os.environ.setdefault("SNAKE_DQN_DEVICE", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import hashlib  # noqa: E402
import importlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, Mapping  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.runpod_fanout import jobspec  # noqa: E402
from research.runpod_fanout.wrappers import (  # noqa: E402
    HERO_SHA256,
    SCRIPTED_SHA256,
    WRAPPERS,
)

RECORD_STAMP_SCHEMA = "runpod-fanout-record/v1"
EXIT_DIVERGENT = 3


def build_spec(wrapper_id: str) -> Any:
    entry = WRAPPERS[wrapper_id]
    module = importlib.import_module(entry["module"])
    factory = entry["spec"]
    if factory == "make_spec":
        return module.make_spec(module.DOMAIN, {})
    if factory.startswith("build_spec:"):
        return module.build_spec(module.DOMAIN, float(factory.split(":", 1)[1]), {})
    if factory == "SPEC":
        return module.SPEC
    raise ValueError(f"unknown spec factory {factory!r}")


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    try:
        return subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip()
    except OSError:
        return "unknown"


def platform_stamp(provider: str) -> Dict[str, Any]:
    import numpy
    import torch

    platform_id = "|".join(
        [
            f"{platform.system().lower()}-{platform.machine()}",
            f"py{sys.version_info.major}.{sys.version_info.minor}",
            f"torch-{torch.__version__}",
            f"numpy-{numpy.__version__}",
            f"isa_cap-{os.environ.get('ONEDNN_MAX_CPU_ISA', 'none')}",
        ]
    )
    return {
        "platform_id": platform_id,
        "provider": provider,
        "cpu_model": cpu_model(),
        "torch_threads": torch.get_num_threads(),
        "mkl_cbwr": os.environ.get("MKL_CBWR"),
    }


def load_profile(ds: Any, horizon: int) -> Any:
    from src.core.config_loader import load_and_initialize_config
    from src.scripts.tournament_eval import evaluation_profile_for_name

    if ds.sha256_file(ds.DEFAULT_CONFIG) != ds.CONFIG_SHA256:
        raise SystemExit("deployment config sha256 differs from the pinned config")
    load_and_initialize_config(str(ds.DEFAULT_CONFIG))
    profile = evaluation_profile_for_name(ds.PROFILE_NAME, ds.HORIZON)
    if profile.digest != ds.PROFILE_DIGEST:
        raise SystemExit("profile digest differs from the pinned promotion-v2 profile")
    if horizon != ds.HORIZON:
        profile = dataclasses.replace(profile, scored_horizon=int(horizon))
    return profile


def run(spec_ep: Mapping[str, Any], ckpt_dir: Path) -> Dict[str, Any]:
    """Play the episode; return the entry (without platform/fanout stamps)."""
    from research.apex_safety_20260926 import dev_screen as ds

    ds._configure_torch()
    horizon = int(spec_ep["horizon"])
    profile = load_profile(ds, horizon)
    spec = build_spec(spec_ep["wrapper"])
    needed = {HERO_SHA256} | {
        s for s in spec_ep["roster_member_sha256s"] if s not in SCRIPTED_SHA256
    }
    snapshots = {}
    for sha in sorted(needed):
        path = Path(ckpt_dir) / f"{sha}.pth"
        if jobspec.sha256_file(path) != sha:
            raise SystemExit(f"checkpoint {path} does not match its sha256")
        snapshots[sha] = str(path)
    lookup = ds.agent_lookup(snapshots)
    row = {
        "world_seed": int(spec_ep["world_seed"]),
        "mix": spec_ep["mix"],
        "slots": [{"member_sha256": s} for s in spec_ep["roster_member_sha256s"]],
    }
    if spec_ep["engine"] == "live":
        # run_episode reads the module-global HORIZON; this process plays one episode.
        ds.HORIZON = horizon
        with tempfile.TemporaryDirectory() as tmp:
            return ds.run_episode(
                spec_ep["arm"],
                row,
                int(spec_ep["world_index"]),
                lookup,
                profile,
                Path(tmp),
                spec=spec,
            )
    from src.simd_env.eval_engine import run_simd_eval

    variant, lam, ref = WRAPPERS[spec_ep["wrapper"]]["simd"][spec_ep["arm"]]
    roster = [lookup[s] for s in spec_ep["roster_member_sha256s"]]
    seed = int(spec_ep["world_seed"])
    started = time.monotonic()
    records = run_simd_eval(
        lookup[HERO_SHA256],
        roster,
        horizon,
        [seed],
        profile=profile,
        opponent_specs_by_world={seed: roster},
        mix_id=spec_ep["mix"],
        vector61=True,
        hero_safety_veto=variant,
        vector61_forward="rowwise",
        hero_safety_veto_lambda=lam,
        hero_safety_veto_reference_lambda=ref,
    )
    record = json.loads(json.dumps(records[0]))
    return {
        "schema_version": spec.schema,
        "authority": spec.authority,
        "arm": spec_ep["arm"],
        "mix": spec_ep["mix"],
        "world_index": int(spec_ep["world_index"]),
        "world_seed": seed,
        "roster_member_sha256s": list(spec_ep["roster_member_sha256s"]),
        "hero_sha256": HERO_SHA256,
        "safety_veto": True,
        "wall_seconds": time.monotonic() - started,
        "record": record,
        "screen": spec.name,
        "safety_veto_method": record["probes"]["safety_veto"]["method"],
        "veto_diagnostics": record.get("veto_diagnostics"),
        "engine": "simd",
    }


def entry_bytes(entry: Mapping[str, Any]) -> bytes:
    """The on-disk format of the screens' create-only writer."""
    return (json.dumps(entry, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def write_record(out_root: Path, key: str, entry: Mapping[str, Any]) -> str:
    """Create-only write; an existing file must match deterministically. Returns status."""
    dest = Path(out_root) / key
    dest.parent.mkdir(parents=True, exist_ok=True)
    data = entry_bytes(entry)
    tmp = dest.with_name(f".{dest.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    try:
        os.link(tmp, dest)  # atomic create-only: readers never see a partial record
        return "written"
    except FileExistsError:
        old = json.loads(dest.read_text(encoding="utf-8"))
        if jobspec.deterministic_bytes(old) == jobspec.deterministic_bytes(entry):
            return "duplicate-verified"
        return "DIVERGENT"
    finally:
        tmp.unlink(missing_ok=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--spec", required=True, type=Path)
    p.add_argument("--out-root", required=True, type=Path)
    p.add_argument("--ckpt-dir", required=True, type=Path)
    p.add_argument("--provider", required=True)
    p.add_argument("--job-id", required=True)
    p.add_argument("--commit", required=True)
    a = p.parse_args(argv)
    spec_ep = json.loads(a.spec.read_text(encoding="utf-8"))
    key = jobspec.episode_key(spec_ep)
    entry = dict(run(spec_ep, a.ckpt_dir))
    entry["platform"] = platform_stamp(a.provider)
    entry["fanout"] = {
        "schema": RECORD_STAMP_SCHEMA,
        "job_id": a.job_id,
        "episode_key": key,
        "repo_commit": a.commit,
        "wrapper": spec_ep["wrapper"],
        "horizon": int(spec_ep["horizon"]),
        "engine": spec_ep["engine"],
        "spec_sha256": hashlib.sha256(
            json.dumps(spec_ep, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    status = write_record(a.out_root, key, entry)
    print(json.dumps({"key": key, "status": status}), flush=True)
    return EXIT_DIVERGENT if status == "DIVERGENT" else 0


if __name__ == "__main__":
    raise SystemExit(main())
