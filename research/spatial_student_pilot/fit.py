"""Random spatial actor BC/improvement fit with full-state deterministic restart."""

from __future__ import annotations

import copy
import time
from typing import Any, Callable, Sequence

import torch
from torch.nn import functional as F

from research.spatial_student_pilot.config import PilotConfig
from research.spatial_student_pilot.model import SpatialStudent
from research.spatial_student_pilot.observation import build_observation, compact_cat
from research.spatial_student_pilot.resume import UnitStore


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(val) for val in value]
    return copy.deepcopy(value)


def fit_student(
    cfg: PilotConfig,
    ordinary: Sequence[dict[str, Any]],
    searched: Sequence[dict[str, Any]],
    *,
    device: str,
    store: UnitStore,
    stop: Callable[[], bool],
    progress: Callable[[dict[str, Any]], None],
) -> tuple[SpatialStudent, dict[str, Any]]:
    """Resume only the latest verified complete optimizer/RNG/update boundary."""
    if not ordinary or not searched:
        raise ValueError("both ordinary and search training rows required")
    if any(row["partition"] != "train" for row in [*ordinary, *searched]):
        raise ValueError("fit refuses assessment-world data")
    torch.manual_seed(cfg.seed)
    student = SpatialStudent().to(device).train()
    optimizer = torch.optim.Adam(student.parameters(), lr=3e-4)
    generator = torch.Generator().manual_seed(cfg.seed)
    total = cfg.bc_updates + cfg.improvement_updates
    latest = max(
        (int(i.split("-")[1]) for i in store.list_ids() if i.startswith("fit-")), default=0
    )
    if latest:
        snapshot = store.get(f"fit-{latest:07d}")
        if snapshot["update"] != latest or snapshot["seed"] != cfg.seed:
            raise ValueError("fit snapshot identity/counter mismatch")
        student.load_state_dict(snapshot["model"])
        optimizer.load_state_dict(snapshot["optimizer"])
        generator.set_state(snapshot["generator"])
        torch.set_rng_state(snapshot["torch_rng"])
        if str(device).startswith("cuda"):
            torch.cuda.set_rng_state_all(snapshot["cuda_rng"])
    started = time.monotonic()
    completed = 0
    for update in range(latest, total):
        if stop():
            raise InterruptedError("fit interrupted; only latest verified snapshot is eligible")
        if update < cfg.bc_updates:
            indices = torch.randint(len(ordinary), (cfg.batch_size,), generator=generator).tolist()
            rows = [ordinary[i] for i in indices]
            targets = [int(row["b8_action"]) for row in rows]
            weights = [1.0 / cfg.batch_size] * cfg.batch_size
        else:
            half = cfg.batch_size // 2
            if half * 2 != cfg.batch_size:
                raise ValueError("improvement batch must be even")
            a = torch.randint(len(searched), (half,), generator=generator).tolist()
            b = torch.randint(len(ordinary), (half,), generator=generator).tolist()
            rows = [searched[i] for i in a] + [ordinary[i] for i in b]
            targets = [int(searched[i]["target_action"]) for i in a]
            targets += [int(ordinary[i]["b8_action"]) for i in b]
            weights = [1.0 / half] * cfg.batch_size  # equal CE means, summed
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        for begin in range(0, len(rows), cfg.microbatch):
            end = min(len(rows), begin + cfg.microbatch)
            compact = compact_cat([row["compact"] for row in rows[begin:end]])
            logits = student(*build_observation(compact, device=device))
            mask = torch.stack([row["mask"] for row in rows[begin:end]]).to(device).bool()
            labels = torch.tensor(targets[begin:end], device=device)
            if not bool(mask.gather(1, labels[:, None]).all()) or not bool(
                torch.isfinite(logits).all()
            ):
                raise ValueError("nonfinite logits or illegal fit target")
            losses = F.cross_entropy(
                logits.masked_fill(~mask, -torch.inf), labels, reduction="none"
            )
            loss = (losses * torch.tensor(weights[begin:end], device=device)).sum()
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite fit loss")
            loss.backward()
            loss_value += float(loss.detach())
        if not all(
            p.grad is None or bool(torch.isfinite(p.grad).all()) for p in student.parameters()
        ):
            raise ValueError("nonfinite gradient")
        optimizer.step()
        completed += 1
        if not all(bool(torch.isfinite(p).all()) for p in student.parameters()):
            raise ValueError("nonfinite parameters")
        # No training loss/action quality in operational progress (one-look).
        progress({"stage": "fit", "updates": update + 1, "total": total})
        if (update + 1) % cfg.checkpoint_every == 0 or update + 1 == total:
            store.publish(
                f"fit-{update + 1:07d}",
                {
                    "schema": "spatial-fit-snapshot/v1",
                    "update": update + 1,
                    "seed": cfg.seed,
                    "model": _cpu_tree(student.state_dict()),
                    "optimizer": _cpu_tree(optimizer.state_dict()),
                    "generator": generator.get_state(),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": (
                        torch.cuda.get_rng_state_all() if str(device).startswith("cuda") else []
                    ),
                    "final": update + 1 == total,
                },
            )
    student.eval()
    return student, {
        "updates_this_attempt": completed,
        "final_update": total,
        "elapsed_seconds": time.monotonic() - started,
    }
