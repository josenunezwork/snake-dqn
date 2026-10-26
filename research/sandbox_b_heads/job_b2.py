"""Finite, resumable frozen-B8 successor head/loss sweep for Sandbox Job B-2.

This module deliberately consumes an already registry-admitted NPZ projection.
It never searches the artifact tree or imports F5 code/data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import tarfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from research.a100_a2.model import b8_features
from research.bl_residual.data import validate_rows
from research.sandbox_b_heads.core_b2 import CONFIGS, build_sampling_weights, masks, training_loss
from research.sandbox_b_heads.job import metrics as base_metrics
from research.sandbox_b_heads.model_b2 import make_head, optimizer_groups
from research.sandbox_b_heads.resume_b2 import (
    CONFIG_IDS,
    atomic_json,
    empty_ledger,
    identity,
    recipe,
    sha256,
    verify_ledger,
)

SALT = "a100-dev/sandbox-b/split-v1"
THRESHOLDS = (0.0, 0.1, 0.2, 0.4, 0.8)


def _served(q: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    fallback = torch.zeros_like(mask)
    fallback[:, :3] = True
    return q.masked_fill(
        ~torch.where(mask.any(1, keepdim=True), mask, fallback), -torch.inf
    ).argmax(1)


def parent_split(parents: np.ndarray, salt: str = SALT) -> np.ndarray:
    """Return a parent-stable fresh 80/20 split; True means development validation."""
    if parents.ndim != 1 or parents.dtype.kind not in "US":
        raise ValueError("parent_world must be a one-dimensional string array")
    return np.array(
        [
            int.from_bytes(hashlib.sha256(f"{salt}/{p}".encode()).digest()[:8], "big") % 5 == 0
            for p in parents
        ],
        dtype=np.bool_,
    )


def _length_and_margin(table: dict[str, np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    # Scalar 0 is log1p(length)/log1p(4096). Margin is served top-1 minus top-2 Q.
    length = torch.as_tensor(table["scalars"][:, 0], dtype=torch.float32)
    q = torch.as_tensor(table["q"], dtype=torch.float32)
    mask = torch.as_tensor(table["action_mask"], dtype=torch.bool)
    fallback = torch.zeros_like(mask)
    fallback[:, :3] = True
    top = torch.topk(
        q.masked_fill(~torch.where(mask.any(1, keepdim=True), mask, fallback), -torch.inf), 2
    ).values
    return length, (top[:, 0] - top[:, 1]).clamp(0, 32)


def _cache_features(
    table: dict[str, np.ndarray],
    checkpoint: Path,
    device: torch.device,
    progress: Callable[[], None] | None = None,
) -> torch.Tensor:
    from src.model.ego2s_network import Ego2sNet

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload.get("state_dict", payload))
    net = Ego2sNet(7, 60).to(device).eval()
    net.load_state_dict(state)
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(table["q"]), 1024):
            end = start + 1024
            chunks.append(
                b8_features(
                    net,
                    torch.as_tensor(table["local"][start:end], device=device),
                    torch.as_tensor(table["global"][start:end], device=device),
                    torch.as_tensor(table["scalars"][start:end], device=device),
                ).cpu()
            )
            if progress is not None:
                progress()
    return torch.cat(chunks)


def metrics(q: torch.Tensor, table: dict[str, torch.Tensor], selected: torch.Tensor) -> dict:
    """B-1 compatible harm plus denominators, flip rate and life-macro harm."""
    report = base_metrics(q, table, selected)
    ordinary = table["ordinary"]
    base = table["bare"].long()
    row = torch.arange(len(base))
    known_selected = table["status"][row, selected] & table["legal"][row, selected]
    known_base = table["status"][row, base] & table["legal"][row, base]
    harmful = (selected != base) & (
        ~known_selected | ~known_base | (table["died"][row, selected] & ~table["died"][row, base])
    )
    life_harm = [
        harmful[ordinary & (table["life_id"] == life)].float().mean()
        for life in torch.unique(table["life_id"][ordinary])
    ]
    selectable = table["action_mask"].clone()
    selectable[~selectable.any(1), :3] = True
    report.update(
        ordinary_count=int(ordinary.sum()),
        ordinary_flip_rate=(
            float((selected != base)[ordinary].float().mean()) if ordinary.any() else None
        ),
        life_macro_ordinary_harm=float(torch.stack(life_harm).mean()) if life_harm else None,
        rescue_life_count=int(
            torch.unique(
                table["life_id"][
                    (
                        known_base
                        & table["died"][row, base]
                        & (table["status"] & table["legal"] & selectable & ~table["died"]).any(1)
                    )
                ]
            ).numel()
        ),
    )
    return report


def threshold_actions(q: torch.Tensor, table: dict[str, torch.Tensor], tau: float) -> torch.Tensor:
    """Apply the post-hoc B8 bonus; exact ties prefer the base action."""
    base = table["bare"].long()
    chosen = _served(q, table["action_mask"])
    row = torch.arange(len(base), device=q.device)
    # The base is the served B8 action and must remain selectable.
    return torch.where(q[row, chosen] - q[row, base] > tau, chosen, base)


def pareto_frontier(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Retain every point not dominated on micro, life-macro and harm count."""
    eligible = [
        r
        for r in rows
        if all(
            r.get(k) is not None
            for k in ("rescue_retention", "life_macro_rescue_retention", "ordinary_harm_count")
        )
    ]

    def dominates(a: dict, b: dict) -> bool:
        av = (a["rescue_retention"], a["life_macro_rescue_retention"], -a["ordinary_harm_count"])
        bv = (b["rescue_retention"], b["life_macro_rescue_retention"], -b["ordinary_harm_count"])
        return all(x >= y for x, y in zip(av, bv)) and any(x > y for x, y in zip(av, bv))

    return [r for r in eligible if not any(dominates(other, r) for other in eligible)]


def _load_table(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        table = {key: archive[key] for key in archive.files}
    validate_rows(table)
    return table


def _seal(path: Path, completion: dict[str, Any] | None = None) -> None:
    """Publish SHA then the marker last; sealed bytes are never modified."""
    checksum = path.with_name(path.name + ".sha256")
    marker = path.with_name(path.name + ".complete")
    for target, content in (
        (checksum, f"{sha256(path)}  {path.name}\n"),
        (marker, json.dumps(completion, sort_keys=True) + "\n" if completion else "complete\n"),
    ):
        temporary = target.with_name(target.name + ".tmp")
        with temporary.open("x") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(target)


def _restore_bundle(path: Path, unit: dict[str, Any], checkpoint_dir: Path) -> dict[str, Any]:
    """Runtime only: verify internal hashes before restoring a complete unit."""
    with tarfile.open(path, "r") as archive:
        members = archive.getmembers()
        if (
            {m.name for m in members} != {"checkpoint.pt", "report.json", "metadata.json"}
            or len(members) != 3
            or any(not m.isfile() for m in members)
        ):
            raise ValueError("unsafe or incomplete unit bundle")
        contents = {m.name: archive.extractfile(m).read() for m in members}
    metadata = json.loads(contents["metadata.json"])
    expected = {k: v for k, v in unit.items() if k != "bundle"}
    if metadata.get("identity") != expected:
        raise ValueError("bundle identity mismatch")
    for name in ("checkpoint.pt", "report.json"):
        if metadata.get("sha256", {}).get(name) != hashlib.sha256(contents[name]).hexdigest():
            raise ValueError("bundle internal SHA mismatch")
    row = json.loads(contents["report.json"])
    if (
        row.get("config") != unit["id"]
        or row.get("status") != "COMPLETE"
        or row.get("sha256") != metadata["sha256"]["checkpoint.pt"]
    ):
        raise ValueError("bundle report mismatch")
    checkpoint = checkpoint_dir / f"{unit['id']}.pt"
    checkpoint.write_bytes(contents["checkpoint.pt"])
    _seal(checkpoint)
    return row


def _publish_bundle(
    path: Path, checkpoint: Path, row: dict[str, Any], unit_identity: dict[str, Any]
) -> None:
    report = (json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode()
    contents = {"checkpoint.pt": checkpoint.read_bytes(), "report.json": report}
    metadata = {
        "identity": unit_identity,
        "sha256": {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()},
    }
    contents["metadata.json"] = (json.dumps(metadata, sort_keys=True) + "\n").encode()
    temporary = path.with_name(path.name + ".tmp")
    with tarfile.open(temporary, "w") as archive:
        for name, data in contents.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    temporary.replace(path)
    _seal(path, unit_identity)


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
    if args.salt != SALT:
        raise ValueError("B-2 must retain the comparable B-1 split salt")
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
        # Deterministic parent-first prefix; never select using outcome labels.
        _, first = np.unique(table["parent_world"], return_index=True)
        selected = np.sort(first)[:smoke_rows]
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
    for config in CONFIGS:
        checkpoint = checkpoint_dir / f"{config.id}.pt"
        if config.id in done:
            continue
        torch.manual_seed(args.seed)
        model = make_head(config.head).to(device).train()
        optimizer = torch.optim.Adam(optimizer_groups(model, args.learning_rate))
        updates = args.updates or config.updates
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=updates)
        weights = build_sampling_weights(train_table, config)
        strata = None
        if weights is not None:
            decisive = masks(train_table)[3]
            if decisive.any() and (~decisive).any():
                strata = [(torch.where(part)[0], weights[part]) for part in (decisive, ~decisive)]
        for step in range(updates):
            batch = min(args.batch_size, len(train_index))
            if strata is not None:
                parts = []
                for (pool, probabilities), size in zip(strata, (batch // 2, batch - batch // 2)):
                    if size:
                        parts.append(pool[torch.multinomial(probabilities, size, replacement=True)])
                local_picks = torch.cat(parts)
            elif weights is not None:
                local_picks = torch.multinomial(weights, batch, replacement=True)
            else:
                local_picks = torch.randint(len(train_index), (batch,), device=device)
            picks = train_index[local_picks]
            sampled = {k: v[picks] for k, v in tensors.items()}
            q = tensors["q"][picks] + model(features[picks], length[picks], margin[picks])
            gate = (
                model.gate_logits(features[picks], length[picks], margin[picks])
                if hasattr(model, "gate_logits")
                else None
            )
            loss = training_loss(q, tensors["q"][picks], sampled, config, gate)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss for {config.id}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(
                parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ):
                raise RuntimeError(f"non-finite gradient for {config.id}")
            optimizer.step()
            scheduler.step()
            if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
                raise RuntimeError(f"non-finite parameter for {config.id}")
            emit_if_due("running")
        model.eval()
        with torch.no_grad():
            corrected = (
                tensors["q"][val_index]
                + model(features[val_index], length[val_index], margin[val_index])
            ).cpu()
        if not torch.isfinite(corrected).all():
            raise RuntimeError(f"non-finite final corrected Q for {config.id}")
        evaluations = [
            {
                "tau": tau,
                **metrics(
                    corrected, validation_table, threshold_actions(corrected, validation_table, tau)
                ),
            }
            for tau in THRESHOLDS
        ]
        torch.save(
            {
                "schema": "sandbox-b-head/v2",
                "label": "simulator-D",
                "config": asdict(config),
                "state_dict": model.cpu().state_dict(),
                "b8_checkpoint_sha256": sha256(Path(args.b8_checkpoint)),
                "salt": args.salt,
                "recipe_fingerprint": ledger["fingerprint"],
                "updates_completed": updates,
                "gate_threshold": 0.5,
                "thresholds": THRESHOLDS,
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
            ],
            "rows": pareto_frontier(evaluation_rows),
        },
    )
    _seal(output / "pareto.json")
    result = output / "results.json"
    result.write_text(
        json.dumps(
            {
                "schema": "sandbox-b-results/v2",
                "label": "simulator-D",
                "input_sha256": sha256(input_path),
                "salt": args.salt,
                "train_rows": int((~val).sum()),
                "val_rows": int(val.sum()),
                "configs": rows,
                "rows": evaluation_rows,
                "thresholds": THRESHOLDS,
                "pareto": pareto_frontier(evaluation_rows),
            },
            indent=2,
        )
        + "\n"
    )
    fields = [
        "config",
        "tau",
        "head",
        "loss",
        "ratio",
        "updates",
        "sampling",
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
        "total_updates": sum(args.updates or c.updates for c in CONFIGS),
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
    parser.add_argument("--namespace", default="a100-dev/sandbox-b/heads-v2")
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
