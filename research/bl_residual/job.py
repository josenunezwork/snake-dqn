"""Real F3 entrypoint: synthetic plumbing now, reviewed factory fits later.

Only training partitions are uploaded. Held-out outcome files remain on the Mac
until immutable final fit units have been verified. Remote outputs are sealed.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from research.bl_ppo.exports import digest, export_json, export_torch, limit_exports
from research.bl_residual.data import allocate_lives, tensor_cache, validate_rows, world_fold
from research.bl_residual.fit import FitRecipe, fit_residual
from research.bl_residual.model import FrozenB8Residual, fuse_residual
from research.bl_residual.protocol import (
    B8_SHA,
    DOSES,
    allocation_hash,
    canonical_sha,
    read_ledger,
    require_judgment,
)
from src.model.ego2s_network import ARCH, load_ego2s_checkpoint, spec_for


def synthetic_table(base: Any) -> dict[str, np.ndarray]:
    """Synthetic labels are isolated from every factory allocation and decision."""
    rng = np.random.default_rng(3403)
    n = 12
    raw = {
        "local": rng.integers(0, 256, (n, 7, 31, 31), dtype=np.uint8),
        "global": rng.integers(0, 256, (n, 4, 37, 37), dtype=np.uint8),
        "scalars": rng.normal(size=(n, 60)).astype(np.float32),
    }
    with torch.no_grad():
        raw["q"] = base(*[torch.as_tensor(raw[k]) for k in ("local", "global", "scalars")]).numpy()
    raw.update(
        legal=np.ones((n, 6), bool),
        action_mask=np.ones((n, 6), bool),
        bare=raw["q"].argmax(axis=1),
        died=rng.random((n, 6)) < 0.5,
        status=np.full((n, 6), "EXACT"),
        mass=rng.random((n, 6)).astype(np.float32) * 10,
        ordinary=np.arange(n) % 2 == 0,
        sampling_weight=np.ones(n, np.float32),
        natural_stream=np.arange(n) % 2 == 0,
        parent_world=np.asarray([f"synthetic-{i // 3}" for i in range(n)]),
        life_id=np.asarray([f"synthetic-life-{i // 3}" for i in range(n)]),
        row_id=np.asarray([f"synthetic-row-{i}" for i in range(n)]),
    )
    return raw


def run(input_dir: Path, output: Path, *, mode: str, device: str = "cpu") -> dict:
    """Fit complete deterministic units, skipping exactly Mac-verified units."""
    if mode not in ("smoke", "canary", "main"):
        raise ValueError("unrecognized mode")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    control = json.loads((input_dir / "control.json").read_text())
    if control.get("schema") != "bl-residual-control/v1":
        raise ValueError("foreign control")
    for name, sha in control["input_hashes"].items():
        if Path(name).name != name or digest(input_dir / name) != sha:
            raise ValueError("input hash mismatch")
    if digest(input_dir / "b8.pth") != B8_SHA:
        raise ValueError("B8 identity mismatch")
    recipe = FitRecipe()
    recipe_sha = canonical_sha(dataclasses.asdict(recipe))
    if control.get("recipe_sha256") != recipe_sha:
        raise ValueError("recipe binding mismatch")
    synthetic = mode != "main"
    if synthetic != bool(control.get("synthetic")):
        raise ValueError("synthetic/main namespace mismatch")
    if not synthetic:
        require_judgment(input_dir / "judgment.json", input_dir / "registration.md", recipe_sha)
    namespace = control["namespace"]
    requested = control.get("doses")
    if (
        not isinstance(requested, list)
        or not requested
        or len(set(requested)) != len(requested)
        or any(type(d) is not int or d not in DOSES for d in requested)
    ):
        raise ValueError("frozen requested doses required")
    requested_indices = {DOSES.index(d) for d in requested}
    allocation = allocation_hash(control)
    if control.get("allocation") != allocation:
        raise ValueError("allocation mismatch")
    skip = read_ledger(input_dir / "verified-ledger.json", namespace, allocation)
    if not set(skip) <= requested_indices:
        raise ValueError("ledger outside requested dose allocation")
    # Ledger is bound separately to avoid a self-referential allocation hash.
    if digest(input_dir / "verified-ledger.json") != control["ledger_sha256"]:
        raise ValueError("ledger binding mismatch")
    life_allocation = None
    if not synthetic:
        if "life-allocation.json" not in control["input_hashes"]:
            raise ValueError("bound life allocation required")
        life_allocation = json.loads((input_dir / "life-allocation.json").read_text())
        if allocate_lives(life_allocation["catalog"]) != life_allocation:
            raise ValueError("life allocation differs from frozen allocator")
        for dose in requested:
            if not life_allocation["doses"][str(dose)]["ready"]:
                raise ValueError("INSUFFICIENT: requested dose not ready")
    if output.exists():
        raise FileExistsError("fresh output required")
    output.mkdir(parents=True)
    export_json(output / "allocation.json", {"namespace": namespace, "allocation": allocation})
    started = time.monotonic()
    stop = threading.Event()
    counts = {"done": len(skip)}

    def pulse() -> None:
        while not stop.wait(110):
            print(
                json.dumps(
                    {
                        "done": counts["done"],
                        "total": len(requested),
                        "elapsed_seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )

    thread = threading.Thread(target=pulse, daemon=True)
    thread.start()
    completed = []
    try:
        with limit_exports(output, total_bytes=128 * 1024**2, file_bytes=32 * 1024**2):
            for index, dose in enumerate(DOSES):
                if dose not in requested or index in skip:
                    continue
                base = load_ego2s_checkpoint(str(input_dir / "b8.pth"))
                if synthetic:
                    table = synthetic_table(base)
                    actual_recipe = dataclasses.replace(recipe, updates=3, batch_size=4)
                else:
                    filename = f"train-{dose}.npz"
                    if filename not in control["input_hashes"]:
                        raise ValueError("training partition must be a bound input")
                    with np.load(input_dir / filename, allow_pickle=False) as archive:
                        table = {k: archive[k] for k in archive.files}
                    validate_rows(table)
                    if any(world_fold(str(p)) == 0 for p in table["parent_world"]):
                        raise ValueError("held-out parent in training partition")
                    expected_lives = set(life_allocation["doses"][str(dose)]["train_lives"])
                    catalog = {r["life_id"]: r for r in life_allocation["catalog"]}
                    actual = set(table["life_id"].tolist())
                    rescue_ids = {
                        life for life in actual if catalog.get(life, {}).get("rescue_bearing")
                    }
                    if rescue_ids != expected_lives:
                        raise ValueError("training life allocation mismatch")
                    parents = {catalog[life]["parent_world"] for life in expected_lives}
                    for life, parent, ordinary in zip(
                        table["life_id"], table["parent_world"], table["ordinary"]
                    ):
                        if (
                            life not in catalog
                            or catalog[life]["parent_world"] != parent
                            or parent not in parents
                            or (life not in expected_lives and not ordinary)
                        ):
                            raise ValueError(
                                "foreign parent or non-rescue row outside ordinary archive"
                            )
                    if len(expected_lives) != dose - 64:
                        raise ValueError("training dose would shrink N")
                    actual_recipe = recipe
                cache = tensor_cache(table, base)
                model = FrozenB8Residual(base).to(device)
                state = fit_residual(model, cache, actual_recipe)
                fused = fuse_residual(model)
                values = {
                    "obs_spec": spec_for(fused),
                    "arch": ARCH,
                    "config": fused.config,
                    "state_dict": fused.state_dict(),
                    "meta": {
                        "schema": "bl-residual-fit/v1",
                        "synthetic": synthetic,
                        "dose": dose,
                        "namespace": namespace,
                        "allocation": allocation,
                        "index": index,
                        "recipe": dataclasses.asdict(actual_recipe),
                        "recipe_sha256": recipe_sha,
                        "updates": state.updates_completed,
                    },
                    "fit_state": {
                        "residual": model.residual.state_dict(),
                        "optimizer": state.optimizer_state,
                        "rng": state.rng_state,
                        "updates": state.updates_completed,
                    },
                }
                result = export_torch(output / f"fit-{index:04d}.pth", values)
                completed.append({"index": index, "dose": dose, **result})
                counts["done"] += 1
            receipt = {
                "schema": "bl-residual-job/v1",
                "mode": mode,
                "synthetic": synthetic,
                "namespace": namespace,
                "allocation": allocation,
                "completed": completed,
                "skipped": sorted(skip),
                "expected_units": len(requested),
                "done": counts["done"],
                "elapsed_seconds": time.monotonic() - started,
                "scientific_result": False,
                "heldout_outcomes_opened": False,
            }
            export_json(output / "receipt.json", receipt)
            return receipt
    finally:
        stop.set()
        thread.join(timeout=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "canary", "main"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    result = run(args.input_dir, args.output, mode=args.mode, device=args.device)
    print(json.dumps({"done": result["done"], "synthetic": result["synthetic"]}))


if __name__ == "__main__":
    main()
