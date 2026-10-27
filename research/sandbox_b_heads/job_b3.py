"""Finite, resumable retrieval and direct-survival sweep for Sandbox Job B-3.

This module deliberately consumes an already registry-admitted NPZ projection.
It never searches the artifact tree or imports F5 code/data.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import tarfile
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from research.sandbox_b_heads.core_b3 import (
    ADVANTAGES,
    CONFIGS,
    TAU_VETO,
    SurvivalHead,
    pareto_frontier,
    promising,
    survival_loss,
    veto_actions,
)
from research.sandbox_b_heads.geometry_b3 import geometry_features
from research.sandbox_b_heads.job_b2 import (
    _cache_features,
    _length_and_margin,
    _load_table,
    _publish_bundle,
    _restore_bundle,
    _seal,
    metrics,
    parent_split,
    smoke_indices,
)
from research.sandbox_b_heads.resume_b3 import (
    CONFIG_IDS,
    atomic_json,
    empty_ledger,
    identity,
    recipe,
    sha256,
    verify_ledger,
)
from research.sandbox_b_heads.retrieval_b3 import retrieval

SALT = "a100-dev/sandbox-b/split-v1"
CRITERION = {
    "label": "simulator-D",
    "promising_iff": (
        "micro retention >= 0.70 AND ordinary flip_rate <= 0.01 " "AND ordinary harm count <= 1"
    ),
    "requires_nonempty_decisive_and_ordinary_denominators": True,
    "ordinary_flip_rate": "closed-loop proxy only; no rollout or release claim",
    "retrieval": "empirical finite-sample ceiling diagnostic, not an information-theoretic bound",
}


def _context(
    features: torch.Tensor, tensors: dict, length: torch.Tensor, margin: torch.Tensor, kind: str
) -> torch.Tensor:
    if kind == "phi":
        return features
    return torch.cat(
        (
            features,
            tensors["q"],
            torch.nn.functional.one_hot(tensors["bare"].long(), 6).float(),
            tensors["action_mask"].float(),
            length[:, None],
            margin[:, None].log1p(),
        ),
        dim=1,
    )


def run(args: argparse.Namespace) -> Path:
    started = time.monotonic()
    if args.updates < 0 or args.batch_size <= 0 or getattr(args, "smoke_rows", 0) < 0:
        raise ValueError("updates/smoke rows must be nonnegative and batch size positive")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning rate must be finite and positive")
    if (
        args.updates
        and not getattr(args, "synthetic_fixture", False)
        and not getattr(args, "smoke_rows", 0)
    ):
        raise ValueError("update overrides are restricted to explicit development smoke runs")
    if args.namespace != "a100-dev/sandbox-b/heads-v3" and not getattr(
        args, "synthetic_fixture", False
    ):
        raise ValueError("foreign B-3 namespace")
    if getattr(args, "allocation", "split-v1") != "split-v1":
        raise ValueError("foreign B-3 allocation")
    if args.salt != SALT:
        raise ValueError("B-3 must retain the comparable B-1 split salt")
    input_path, output = Path(args.input).resolve(), Path(args.output).resolve()
    registry_path = getattr(args, "registry", None)
    if registry_path:
        from tools.sandbox.registry import digest, load_registry, validate_input, validate_npz

        registry = load_registry(registry_path)
        validate_npz(input_path, registry=registry, dose="D1024")
        validate_input(
            {
                "kind": "frozen-checkpoint",
                "path": str(Path(args.b8_checkpoint).resolve()),
                "sha256": digest(args.b8_checkpoint),
            },
            registry,
        )
    elif not getattr(args, "synthetic_fixture", False):
        raise ValueError("runtime requires --registry; synthetic fixtures must opt in explicitly")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("output must be a fresh empty directory")
    ledger_path = output / "verified-ledger.json"
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir()
    unit_dir = output / "units"
    unit_dir.mkdir()
    allocation = getattr(args, "allocation", "split-v1")
    frozen_recipe = recipe(
        input_path,
        Path(args.b8_checkpoint),
        salt=args.salt,
        seed=args.seed,
        updates=args.updates,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        smoke_rows=getattr(args, "smoke_rows", 0),
    )
    input_ledger = getattr(args, "ledger", None)
    ledger = (
        verify_ledger(Path(input_ledger), frozen_recipe, args.namespace, allocation)
        if input_ledger
        else empty_ledger(frozen_recipe, args.namespace, allocation)
    )
    done, rows, units = set(), [], []
    for unit in ledger["units"]:
        source = Path(unit["bundle"]["path"])
        if not source.is_absolute():
            source = Path(input_ledger).parent / source
        destination = unit_dir / f"{unit['allocation_index']:02d}-{unit['id']}.tar"
        row = _restore_bundle(source, unit, checkpoint_dir)
        shutil.copyfile(source, destination)
        _seal(destination, {k: v for k, v in unit.items() if k != "bundle"})
        units.append(
            {
                **{k: v for k, v in unit.items() if k != "bundle"},
                "bundle": {
                    "path": destination.name,
                    "size": destination.stat().st_size,
                    "sha256": sha256(destination),
                },
            }
        )
        rows.append(row)
        done.add(unit["id"])
    last_progress = time.monotonic()

    def emit_progress(state: str) -> None:
        """Write outcome-blind runtime telemetry without mutating sealed exports."""
        elapsed = time.monotonic() - started
        completed = len(done)
        eta = (elapsed / completed) * (len(CONFIG_IDS) - completed) if completed else None
        print(
            json.dumps(
                {
                    "progress": state,
                    "completed_configs": completed,
                    "total_configs": len(CONFIG_IDS),
                    "elapsed_seconds": elapsed,
                    "eta_seconds": eta,
                },
                allow_nan=False,
            ),
            flush=True,
        )

    def emit_if_due(state: str) -> None:
        nonlocal last_progress
        if time.monotonic() - last_progress >= 120:
            emit_progress(state)
            last_progress = time.monotonic()

    emit_progress("started")
    table = _load_table(input_path)
    smoke_rows = getattr(args, "smoke_rows", 0)
    if smoke_rows:
        selected = smoke_indices(table["parent_world"], smoke_rows)
        table = {key: value[selected] for key, value in table.items()}
    val = parent_split(table["parent_world"], args.salt)
    if not val.any() or val.all():
        raise ValueError("salt split has no train or validation parent rows")
    feature_started = time.monotonic()
    resolved_device = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    device = torch.device(resolved_device)
    features = (
        _cache_features(
            table, Path(args.b8_checkpoint), device, lambda: emit_if_due("feature-cache")
        )
        if len(done) < len(CONFIG_IDS)
        else None
    )
    feature_seconds = time.monotonic() - feature_started
    training_started = time.monotonic()
    length, margin = _length_and_margin(table)
    tensors = {
        key: torch.as_tensor(value)
        for key, value in table.items()
        if key in {"q", "legal", "action_mask", "bare", "died", "ordinary"}
    }
    tensors["status"] = torch.as_tensor(table["status"] == "EXACT")
    life_map = {life: index for index, life in enumerate(sorted(set(table["life_id"].tolist())))}
    tensors["life_id"] = torch.as_tensor([life_map[life] for life in table["life_id"].tolist()])
    train_index, val_index = (
        torch.where(~torch.as_tensor(val))[0],
        torch.where(torch.as_tensor(val))[0],
    )
    # The frozen feature/label cache fits easily in A100 memory; transfer once.
    # The CPU route follows the same minibatch operations without device copies.
    if features is not None:
        features = features.to(device)
    length, margin = length.to(device), margin.to(device)
    tensors = {key: value.to(device) for key, value in tensors.items()}
    train_index, val_index = train_index.to(device), val_index.to(device)
    train_table = {k: v[train_index] for k, v in tensors.items()}
    validation_table = {k: v[val_index].cpu() for k, v in tensors.items()}
    retrieval_cache = {}
    geometry = None
    for config in CONFIGS:
        checkpoint = checkpoint_dir / f"{config.id}.pt"
        if config.id in done:
            continue
        diagnostics = None
        if config.k:
            # Diagnostic 0 runs before any fitting. Only training labels populate neighbors.
            if config.inputs not in retrieval_cache:
                vectors = (
                    features if config.inputs == "phi" else torch.cat((features, tensors["q"]), 1)
                )
                retrieval_cache[config.inputs] = retrieval(
                    vectors[train_index],
                    vectors[val_index],
                    train_table,
                    {key: value[val_index] for key, value in tensors.items()},
                    k_values=tuple(
                        c.k
                        for c in CONFIGS
                        if c.k and c.inputs == config.inputs and c.id not in done
                    ),
                    feature_name=config.inputs,
                    progress=lambda: emit_if_due("retrieval"),
                )
            retrieved = retrieval_cache[config.inputs]
            probabilities = retrieved["probabilities"][config.k].cpu()
            diagnostics = retrieved["diagnostics"][config.k]
            model_payload = {
                "method": "train-only cosine retrieval with matching base action",
                "train_row_indices": train_index.cpu(),
                "input_sha256": frozen_recipe["data_sha256"],
                "diagnostics": diagnostics,
            }
            updates = 0
        else:
            if config.inputs == "phi_q_geometry" and geometry is None:
                geometry = torch.as_tensor(
                    geometry_features(table, progress=lambda: emit_if_due("geometry")),
                    device=device,
                )
            context = _context(features, tensors, length, margin, config.inputs)
            mean, std = context[train_index].mean(0), context[train_index].std(
                0, unbiased=False
            ).clamp_min(1e-5)
            context = (context - mean) / std
            geo = (
                geometry
                if config.inputs == "phi_q_geometry"
                else context.new_zeros((len(context), 6, 0))
            )
            # Geometry statistics are fitted on training rows only, per feature across actions.
            geo_mean = geo[train_index].mean((0, 1))
            geo_std = (
                geo[train_index].std((0, 1), unbiased=False).clamp_min(1e-5)
                if geo.shape[2]
                else context.new_empty((0,))
            )
            geo = (geo - geo_mean) / geo_std
            torch.manual_seed(args.seed)
            model = SurvivalHead(context.shape[1], geo.shape[2]).to(device).train()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
            updates = args.updates or config.updates
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=updates)
            for step in range(updates):
                picks = train_index[
                    torch.randint(
                        len(train_index), (min(args.batch_size, len(train_index)),), device=device
                    )
                ]
                logits = model(context[picks], geo[picks])
                loss = survival_loss(logits, {key: value[picks] for key, value in tensors.items()})
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss for {config.id}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if any(
                    p.grad is not None and not torch.isfinite(p.grad).all()
                    for p in model.parameters()
                ):
                    raise RuntimeError(f"non-finite gradient for {config.id}")
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
                scheduler.step()
                emit_if_due("fitting")
            model.eval()
            with torch.no_grad():
                probabilities = torch.cat(
                    [
                        model(context[picks], geo[picks]).sigmoid().cpu()
                        for picks in val_index.split(1024)
                    ]
                )
            model_payload = {
                "state_dict": model.cpu().state_dict(),
                "context_mean": mean.cpu(),
                "context_std": std.cpu(),
                "geometry_mean": geo_mean.cpu(),
                "geometry_std": geo_std.cpu(),
                "context_dim": context.shape[1],
                "geometry_dim": geo.shape[2],
                "objective": "row-normalized exact-legal-action survival BCE",
            }
        if not torch.isfinite(probabilities).all():
            raise RuntimeError(f"non-finite survival probabilities for {config.id}")
        evaluations = []
        for tau in TAU_VETO:
            for advantage in ADVANTAGES:
                point = {
                    "tau_veto": tau,
                    "survival_advantage": advantage,
                    **metrics(
                        probabilities,
                        validation_table,
                        veto_actions(probabilities, validation_table, tau, advantage),
                    ),
                }
                point["promising"] = promising(point)
                evaluations.append(point)
        torch.save(
            {
                "schema": "sandbox-b-survival/v3",
                "label": "simulator-D",
                "config": asdict(config),
                "b8_checkpoint_sha256": frozen_recipe["b8_sha256"],
                "salt": args.salt,
                "recipe_fingerprint": ledger["fingerprint"],
                "updates_completed": updates,
                "tau_veto": TAU_VETO,
                "survival_advantages": ADVANTAGES,
                **model_payload,
            },
            checkpoint,
        )
        digest = sha256(checkpoint)
        _seal(checkpoint)
        row = {
            "config": config.id,
            "status": "COMPLETE",
            "checkpoint": checkpoint.name,
            "sha256": digest,
            "settings": asdict(config),
            "evaluations": evaluations,
            "diagnostics": diagnostics,
        }
        index = CONFIG_IDS.index(config.id)
        unit_identity = identity(index, args.namespace, allocation, ledger["fingerprint"])
        bundle_path = unit_dir / f"{index:02d}-{config.id}.tar"
        _publish_bundle(bundle_path, checkpoint, row, unit_identity)
        units.append(
            {
                **unit_identity,
                "bundle": {
                    "path": bundle_path.name,
                    "size": bundle_path.stat().st_size,
                    "sha256": sha256(bundle_path),
                },
            }
        )
        rows.append(row)
        done.add(config.id)
        # Each complete configuration already has an immutable bundle,
        # sidecar, and completion marker before we report it.  A
        # watchdog timeout can therefore preserve all preceding work.
        emit_progress("unit-exported")
        last_progress = time.monotonic()
    training_seconds = time.monotonic() - training_started
    export_started = time.monotonic()
    rows.sort(key=lambda row: CONFIG_IDS.index(row["config"]))
    units.sort(key=lambda unit: unit["allocation_index"])
    # Ledger is sealed only once; immutable unit bundles provide incremental progress.
    final_ledger = {
        **ledger,
        "units": [
            {**unit, "bundle": {**unit["bundle"], "path": "units/" + unit["bundle"]["path"]}}
            for unit in units
        ],
    }
    atomic_json(ledger_path, final_ledger)
    _seal(ledger_path)
    evaluation_rows = [
        {
            **{k: v for k, v in record.items() if k not in {"evaluations", "settings"}},
            **record["settings"],
            **point,
        }
        for record in rows
        for point in record["evaluations"]
    ]
    atomic_json(
        output / "pareto.json",
        {
            "label": "simulator-D",
            "objectives": [
                "max micro retention",
                "max life-macro retention",
                "min ordinary harm count",
                "min ordinary flip rate",
            ],
            "rows": pareto_frontier(evaluation_rows),
        },
    )
    _seal(output / "pareto.json")
    result = output / "results.json"
    result.write_text(
        json.dumps(
            {
                "schema": "sandbox-b-results/v3",
                "criterion": CRITERION,
                "metric_definition": (
                    "same decisive exact/legal base-death denominator as B-2; "
                    "life macro averages decisive lives only"
                ),
                "label": "simulator-D",
                "input_sha256": sha256(input_path),
                "salt": args.salt,
                "train_rows": int((~val).sum()),
                "val_rows": int(val.sum()),
                "configs": rows,
                "rows": evaluation_rows,
                "thresholds": TAU_VETO,
                "survival_advantages": ADVANTAGES,
                "pareto": pareto_frontier(evaluation_rows),
            },
            indent=2,
        )
        + "\n"
    )
    fields = [
        "config",
        "tau_veto",
        "survival_advantage",
        "promising",
        "inputs",
        "k",
        "updates",
        "status",
        "checkpoint",
        "sha256",
        "rescue_retention",
        "life_macro_rescue_retention",
        "rescue_decisive_count",
        "ordinary_harm_proxy",
        "ordinary_harm_count",
        "ordinary_count",
        "ordinary_flip_rate",
        "life_macro_ordinary_harm",
        "rescue_life_count",
    ]
    with (output / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(evaluation_rows)
    _seal(output / "results.csv")
    archive = output / "checkpoints.tar"
    with tarfile.open(archive, "w") as bundle:
        for item in sorted(checkpoint_dir.iterdir()):
            bundle.add(item, arcname=f"checkpoints/{item.name}")
    _seal(archive)
    _seal(result)
    timing = {
        "label": "simulator-D",
        "feature_cache_seconds": feature_seconds,
        "training_seconds": training_seconds,
        "export_seconds": time.monotonic() - export_started,
        "total_seconds": time.monotonic() - started,
        "rows": len(table["q"]),
        "total_updates": sum((args.updates or c.updates) if not c.k else 0 for c in CONFIGS),
        "completed_configs": len(done),
        "restored_configs": len(ledger["units"]),
    }
    atomic_json(output / "timing.json", timing)
    _seal(output / "timing.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--b8-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--namespace", default="a100-dev/sandbox-b/heads-v3")
    parser.add_argument("--allocation", default="split-v1")
    parser.add_argument("--ledger", help="immutable input ledger; output gets the updated ledger")
    parser.add_argument(
        "--registry", help="canonical sandbox registry; required outside synthetic tests"
    )
    parser.add_argument("--synthetic-fixture", action="store_true")
    parser.add_argument("--salt", default=SALT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--updates", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=3403)
    parser.add_argument("--smoke-rows", type=int, default=0)
    args = parser.parse_args()
    if args.updates < 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        raise SystemExit("updates, batch-size and learning-rate must be positive")
    started = time.monotonic()
    result = run(args)
    print(json.dumps({"result": str(result), "seconds": round(time.monotonic() - started, 3)}))


if __name__ == "__main__":
    main()
