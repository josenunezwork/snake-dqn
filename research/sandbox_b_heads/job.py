"""Finite, resumable frozen-B8 successor head/loss sweep for Sandbox Job B.

This module deliberately consumes an already registry-admitted NPZ projection.
It never searches the artifact tree or imports F5 code/data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional

from research.a100_a2.model import b8_features
from research.bl_residual.data import validate_rows
from research.sandbox_b_heads.model import make_head

SALT = "a100-dev/sandbox-b/split-v1"
HEADS = ("linear", "gated_mlp", "cosine")
KL_WEIGHTS = (1, 2, 4, 8, 16)
SURVIVAL_WEIGHTS = (1, 2, 4, 8)


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


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
    # Scalar 0 is B8 normalized length.  Margin is served top-1 minus top-2 Q.
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
    table: dict[str, np.ndarray], checkpoint: Path, device: torch.device
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
    return torch.cat(chunks)


def _survival_pair_loss(q: torch.Tensor, table: dict[str, torch.Tensor]) -> torch.Tensor:
    selectable = table["action_mask"].clone()
    selectable[~selectable.any(1), :3] = True
    known = table["status"] & table["legal"] & selectable
    survivor, dying = known & ~table["died"], known & table["died"]
    pairs = survivor[:, :, None] & dying[:, None, :]
    if not pairs.any():
        return q.sum() * 0
    # Survivor-reference target: every exact survivor ranks above each exact death.
    return functional.softplus(q[:, None, :] - q[:, :, None])[pairs].mean()


def metrics(
    corrected: torch.Tensor, table: dict[str, torch.Tensor]
) -> dict[str, float | int | None]:
    selected, base = _served(corrected, table["action_mask"]), table["bare"].long()
    row = torch.arange(len(selected))
    exact = table["status"]
    selected_ok = exact[row, selected] & table["legal"][row, selected]
    base_ok = exact[row, base] & table["legal"][row, base]
    selectable = table["action_mask"].clone()
    selectable[~selectable.any(1), :3] = True
    decisive = (
        base_ok
        & table["died"][row, base]
        & (exact & table["legal"] & selectable & ~table["died"]).any(1)
    )
    retained = decisive & selected_ok & ~table["died"][row, selected]
    ordinary = table["ordinary"]
    harmful = (
        (selected != base)
        & selected_ok
        & table["died"][row, selected]
        & base_ok
        & ~table["died"][row, base]
    )
    count = int(decisive.sum())
    return {
        "rescue_retention": float(retained[decisive].float().mean()) if count else None,
        "rescue_decisive_count": count,
        "ordinary_harm_proxy": float(harmful[ordinary].float().mean()) if ordinary.any() else None,
        "ordinary_harm_count": int(harmful[ordinary].sum()),
    }


@dataclass(frozen=True)
class Config:
    head: str
    kl_weight: int
    survival_weight: int

    @property
    def id(self) -> str:
        return f"{self.head}-kl{self.kl_weight}-survival{self.survival_weight}"


def _load_table(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        table = {key: archive[key] for key in archive.files}
    validate_rows(table)
    return table


def _load_ledger(path: Path) -> set[str]:
    if not path.exists():
        return set()
    payload = json.loads(path.read_text())
    units = payload.get("units", payload.get("verified_units", []))
    if not isinstance(units, list):
        raise ValueError("verified ledger units must be a list")
    return {item["id"] if isinstance(item, dict) else item for item in units}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Publish a completed-unit ledger only after its complete JSON bytes exist."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def run(args: argparse.Namespace) -> Path:
    input_path, output = Path(args.input).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    table = _load_table(input_path)
    val = parent_split(table["parent_world"], args.salt)
    if not val.any() or val.all():
        raise ValueError("salt split has no train or validation parent rows")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    features = _cache_features(table, Path(args.b8_checkpoint), device)
    length, margin = _length_and_margin(table)
    tensors = {
        key: torch.as_tensor(value)
        for key, value in table.items()
        if key in {"q", "legal", "action_mask", "bare", "died", "ordinary"}
    }
    tensors["status"] = torch.as_tensor(table["status"] == "EXACT")
    train_index, val_index = (
        torch.where(~torch.as_tensor(val))[0],
        torch.where(torch.as_tensor(val))[0],
    )
    ledger_path = output / "verified-ledger.json"
    input_ledger = getattr(args, "ledger", None)
    done = _load_ledger(Path(input_ledger)) if input_ledger else _load_ledger(ledger_path)
    rows: list[dict[str, Any]] = []
    for head in HEADS:
        for kl in KL_WEIGHTS:
            for survival in SURVIVAL_WEIGHTS:
                config = Config(head, kl, survival)
                checkpoint = output / f"{config.id}.pt"
                if config.id in done:
                    rows.append(
                        {
                            "config": config.id,
                            "status": "SKIPPED_VERIFIED",
                            "checkpoint": checkpoint.name,
                        }
                    )
                    continue
                torch.manual_seed(args.seed)
                model = make_head(head).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
                for _ in range(args.updates):
                    picks = train_index[
                        torch.randint(len(train_index), (min(args.batch_size, len(train_index)),))
                    ]
                    q = tensors["q"][picks].to(device) + model(
                        features[picks].to(device),
                        length[picks].to(device),
                        margin[picks].to(device),
                    )
                    ordinary = tensors["ordinary"][picks].to(device)
                    kl_loss = (
                        functional.kl_div(
                            functional.log_softmax(q[ordinary], 1),
                            functional.softmax(tensors["q"][picks][ordinary].to(device), 1),
                            reduction="batchmean",
                        )
                        if ordinary.any()
                        else q.sum() * 0
                    )
                    loss = kl * kl_loss + survival * _survival_pair_loss(
                        q, {k: v[picks].to(device) for k, v in tensors.items() if k != "ordinary"}
                    )
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"non-finite loss for {config.id}")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                with torch.no_grad():
                    corrected = (
                        tensors["q"][val_index]
                        + model(
                            features[val_index].to(device),
                            length[val_index].to(device),
                            margin[val_index].to(device),
                        ).cpu()
                    )
                report = metrics(corrected, {k: v[val_index] for k, v in tensors.items()})
                torch.save(
                    {
                        "schema": "sandbox-b-head/v1",
                        "config": asdict(config),
                        "state_dict": model.cpu().state_dict(),
                        "b8_checkpoint_sha256": sha256(Path(args.b8_checkpoint)),
                        "salt": args.salt,
                    },
                    checkpoint,
                )
                digest = sha256(checkpoint)
                (checkpoint.with_suffix(".pt.sha256")).write_text(f"{digest}  {checkpoint.name}\n")
                rows.append(
                    {
                        "config": config.id,
                        "status": "COMPLETE",
                        "checkpoint": checkpoint.name,
                        "sha256": digest,
                        **report,
                    }
                )
                done.add(config.id)
                _atomic_json(
                    ledger_path,
                    {
                        "schema": "sandbox-b-verified-ledger/v1",
                        "namespace": args.namespace,
                        "units": sorted(done),
                        "verified_units": sorted(done),
                    },
                )
    result = output / "results.json"
    result.write_text(
        json.dumps(
            {
                "schema": "sandbox-b-results/v1",
                "input_sha256": sha256(input_path),
                "salt": args.salt,
                "train_rows": int((~val).sum()),
                "val_rows": int(val.sum()),
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )
    fields = [
        "config",
        "status",
        "checkpoint",
        "sha256",
        "rescue_retention",
        "rescue_decisive_count",
        "ordinary_harm_proxy",
        "ordinary_harm_count",
    ]
    with (output / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (result.with_suffix(".json.sha256")).write_text(f"{sha256(result)}  {result.name}\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--b8-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--namespace", default="a100-dev/sandbox-b/heads-v1")
    parser.add_argument("--ledger", help="immutable input ledger; output gets the updated ledger")
    parser.add_argument("--salt", default=SALT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--updates", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=3403)
    args = parser.parse_args()
    if args.updates <= 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        raise SystemExit("updates, batch-size and learning-rate must be positive")
    started = time.monotonic()
    result = run(args)
    print(json.dumps({"result": str(result), "seconds": round(time.monotonic() - started, 3)}))


if __name__ == "__main__":
    main()
