"""Warm learner trajectories with ordinary BC and unbiased queried supervision."""

from __future__ import annotations

from typing import Any, Callable, Sequence

import torch
from torch.nn import functional as F

from research.spatial_student_pilot.fit import _cpu_tree
from research.spatial_student_pilot.model import SpatialStudent
from research.spatial_student_pilot.observation import build_observation, compact_cat
from research.spatial_student_pilot.resume import UnitStore

from .config import RoundConfig, identity
from .data import validate_rows
from .guards import require_training_scope


def row_loss(
    logits: torch.Tensor, rows: Sequence[dict[str, Any]], cfg: RoundConfig, *, searched: bool
) -> torch.Tensor:
    """Hard CE anchor is always present; soft/centered advantage losses are opt-in."""
    masks = torch.stack([r["mask"] for r in rows]).to(logits.device).bool()
    actions = torch.tensor(
        [r["target_action"] if searched else r["b8_action"] for r in rows], device=logits.device
    )
    if not bool(torch.isfinite(logits).all()) or not bool(masks.gather(1, actions[:, None]).all()):
        raise ValueError("nonfinite logits or illegal target")
    masked = logits.masked_fill(~masks, -torch.inf)
    loss = F.cross_entropy(masked, actions, reduction="none")
    if searched and cfg.margin_threshold:
        margins = torch.tensor([r["top2_margin"] for r in rows], device=logits.device)
        if not bool(torch.isfinite(margins).all()) or bool((margins < 0).any()):
            raise ValueError("invalid label margins")
        # Near-ties smoothly reduce hard-target pressure; default is unchanged.
        loss = loss * (margins / cfg.margin_threshold).clamp(max=1)
    if searched and cfg.soft_advantage_weight:
        advantages = torch.stack([torch.as_tensor(r["action_advantages"]) for r in rows]).to(
            device=logits.device, dtype=logits.dtype
        )
        if not bool(torch.isfinite(advantages[masks]).all()):
            raise ValueError("invalid per-action advantages")
        target = (advantages / cfg.search_temperature).masked_fill(~masks, -torch.inf).softmax(1)
        log_probs = (masked / cfg.search_temperature).log_softmax(1).masked_fill(~masks, 0)
        loss = loss - cfg.soft_advantage_weight * cfg.search_temperature**2 * (
            target * log_probs
        ).sum(1)
    if cfg.soft_weight or (searched and cfg.advantage_weight):
        q = torch.stack(
            [torch.as_tensor(r["teacher_q"] if searched else r["b8_q"]) for r in rows]
        ).to(device=logits.device, dtype=torch.float32)
        if not bool(torch.isfinite(q[masks]).all()):
            raise ValueError("nonfinite legal teacher Q")
        q = q.masked_fill(~masks, 0)
        if cfg.soft_weight:
            temperature = cfg.search_temperature if searched else cfg.b8_temperature
            target = (q / temperature).masked_fill(~masks, -torch.inf).softmax(1)
            log_probs = (masked / temperature).log_softmax(1).masked_fill(~masks, 0)
            loss = loss - cfg.soft_weight * temperature**2 * (target * log_probs).sum(1)
        if searched and cfg.advantage_weight:
            denom = masks.sum(1, keepdim=True)
            center_q = q - q.sum(1, keepdim=True) / denom
            predicted = logits.masked_fill(~masks, 0)
            predicted = predicted - predicted.sum(1, keepdim=True) / denom
            lengths = torch.tensor([max(1, r["hero_length"]) for r in rows], device=logits.device)
            advantage = center_q / lengths[:, None]
            error = F.smooth_l1_loss(predicted, advantage, reduction="none").masked_fill(~masks, 0)
            loss = loss + cfg.advantage_weight * error.sum(1) / denom[:, 0]
    return loss


def fit(
    cfg: RoundConfig,
    ordinary: Sequence[dict[str, Any]],
    searched: Sequence[dict[str, Any]],
    *,
    initial: dict[str, Any],
    population: int,
    device: str,
    store: UnitStore,
    training_scope: str = "research",
    stop: Callable[[], bool] = lambda: False,
    progress: Callable[[dict[str, Any]], None] = lambda _: None,
) -> tuple[SpatialStudent, dict[str, Any]]:
    """Carry model, Adam, sampler and framework RNG through rounds and retries.

    Search rows are sampled uniformly from queried roots. n_query/(N*p) is
    required to recover the full ungated candidate mean in expectation. Do not
    normalize by observed weight sums or clip large weights silently.
    """
    require_training_scope(cfg, searched, training_scope)
    validate_rows(ordinary, "train")
    validate_rows(searched, "train")
    if not ordinary or population < len(searched) or population < 1:
        raise ValueError("invalid fixed training population")
    if any(not r.get("queried") or not 0 < r["inclusion_probability"] <= 1 for r in searched):
        raise ValueError("unqueried roots cannot carry expert labels")
    if initial.get("schema") != "spatial-exit-learner/v1":
        raise ValueError("full warm learner state required")
    fit_identity = identity(
        dict(
            ordinary=[r["uid"] for r in ordinary],
            searched=[r["label_key"] for r in searched],
            probabilities=[r["inclusion_probability"] for r in searched],
            population=population,
            config_sha256=cfg.sha256,
        )
    )
    latest = store.latest_fit()
    state = store.get(latest) if latest else initial
    if latest and state.get("fit_identity") != fit_identity:
        raise ValueError("resume training dataset identity differs")
    student = SpatialStudent().to(device).train()
    student.load_state_dict(state["model"])
    optimizer = torch.optim.Adam(student.parameters(), lr=3e-4)
    optimizer.load_state_dict(state["optimizer"])
    generator = torch.Generator()
    generator.set_state(state["generator"])
    torch.set_rng_state(state["torch_rng"])
    if str(device).startswith("cuda"):
        if state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        else:
            torch.cuda.manual_seed_all(cfg.seed)
    cache = None
    if cfg.observation_cache == "integer":
        from .cache import IntegerRasterCache

        cache = IntegerRasterCache([*ordinary, *searched], device, cfg.cache_max_bytes)
    start = int(state["round_update"]) if latest else 0
    if latest and int(latest.split("-")[1]) != start:
        raise ValueError("trajectory counter mismatch")
    global_start = state["global_update"] - start if latest else initial["global_update"]
    for update in range(start, cfg.updates):
        if stop():
            raise InterruptedError("only published full learner states can resume")
        half = cfg.batch_size // 2
        ordinary_ids = torch.randint(len(ordinary), (half,), generator=generator).tolist()
        search_ids = (
            torch.randint(len(searched), (half,), generator=generator).tolist() if searched else []
        )
        optimizer.zero_grad(set_to_none=True)
        for pool, ids, is_search in ((ordinary, ordinary_ids, False), (searched, search_ids, True)):
            for begin in range(0, len(ids), cfg.microbatch):
                rows = [pool[i] for i in ids[begin : begin + cfg.microbatch]]
                inputs = (
                    cache.batch(rows)
                    if cache is not None
                    else build_observation(compact_cat([r["compact"] for r in rows]), device=device)
                )
                losses = row_loss(student(*inputs), rows, cfg, searched=is_search)
                if is_search:
                    weights = [
                        len(searched) / (population * r["inclusion_probability"]) for r in rows
                    ]
                else:
                    weights = [cfg.bc_weight] * len(rows)
                loss = (losses * torch.tensor(weights, device=device)).sum() / half
                if not bool(torch.isfinite(loss)):
                    raise ValueError("nonfinite learner loss")
                loss.backward()
        flags = [torch.isfinite(p.grad).all() for p in student.parameters() if p.grad is not None]
        if not bool(torch.stack(flags).all()):
            raise ValueError("nonfinite learner gradients")
        optimizer.step()
        if not bool(torch.stack([torch.isfinite(p).all() for p in student.parameters()]).all()):
            raise ValueError("nonfinite learner parameters")
        progress(dict(stage="fit", updates=update + 1, total=cfg.updates))
        if (update + 1) % cfg.checkpoint_every == 0 or update + 1 == cfg.updates:
            state = dict(
                schema="spatial-exit-learner/v1",
                model=_cpu_tree(student.state_dict()),
                optimizer=_cpu_tree(optimizer.state_dict()),
                generator=generator.get_state(),
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if str(device).startswith("cuda") else [],
                round_update=update + 1,
                global_update=global_start + update + 1,
                round_index=cfg.round_index,
                fit_identity=fit_identity,
                final=update + 1 == cfg.updates,
                quarantined=bool(searched) or bool(initial.get("quarantined", True)),
                training_scope=training_scope,
            )
            store.publish(f"fit-{update + 1:07d}", state)
    return student.eval(), state


def initial_state(student: SpatialStudent, seed: int) -> dict[str, Any]:
    """Initialize a smoke-only learner; scientific imports preserve pilot Adam/RNG."""
    optimizer = torch.optim.Adam(student.parameters(), lr=3e-4)
    return dict(
        schema="spatial-exit-learner/v1",
        model=_cpu_tree(student.state_dict()),
        optimizer=_cpu_tree(optimizer.state_dict()),
        generator=torch.Generator().manual_seed(seed).get_state(),
        torch_rng=torch.Generator().manual_seed(seed).get_state(),
        cuda_rng=[],
        global_update=0,
        round_update=0,
        round_index=0,
        final=True,
    )
