"""P4 ExIt-B8 fit loop and reusable two-world/200-update preflight API."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from .data import ReplayDivergenceError, StreamingReplayLoader, to_device, verify_world_proof
from .diagnostics import FitTripwire, ProbeHook, enforce_tripwires, selection_key, validate_probe
from .interface import FakeReplaySource, ReplaySource, forward_batch
from .loss import LossConfig, exit_fit_loss


def _cpu_guard() -> None:
    """Do not steal Phase-0 CPU capacity during Mac-side wiring/preflight."""
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMBA_NUM_THREADS",
    ):
        os.environ.setdefault(name, "1")
    torch.set_num_threads(1)


def optimizer_for(model: torch.nn.Module, pretrained: bool) -> torch.optim.Adam:
    """Use 2.5e-5 for warm-started parameters and 1e-4 for genuinely new ones."""
    if not pretrained:
        return torch.optim.Adam(model.parameters(), lr=1e-4)
    declared = set(getattr(model, "new_parameter_names", ()))
    old, new = [], []
    for name, parameter in model.named_parameters():
        is_new = name in declared
        if not hasattr(model, "new_parameter_names"):
            is_new = name.startswith(("tower.", "tower_", "aux_", "hindsight", "free_region_head"))
        (new if is_new else old).append(parameter)
    groups = []
    if old:
        groups.append({"params": old, "lr": 2.5e-5})
    if new:
        groups.append({"params": new, "lr": 1e-4})
    return torch.optim.Adam(groups)


def _heldout_metrics(
    model: torch.nn.Module, source: ReplaySource, device: torch.device, epoch: int
) -> dict[str, float]:
    """Measure decisive agreement from a separate replay pass; source owns split identity."""
    # Production sources must expose a genuine held-out stream.  The fake uses
    # its deterministic stream only when an explicit fake preflight is requested.
    replay = getattr(source, "replay_heldout", None)
    if not callable(replay):
        if not bool(getattr(source, "is_fake", False)):
            raise ValueError(
                "production source lacks required replay_heldout(epoch) decisive labels"
            )
        worlds = source.replay_epoch(epoch)
    else:
        worlds = replay(epoch)
    agree = total = 0
    chunks: dict[str, list[int]] = {}
    model.eval()
    with torch.no_grad():
        for world in worlds:
            chunk = world.chunk_id or "unpartitioned"
            chunks.setdefault(chunk, [0, 0])[0] += 1
            if not verify_world_proof(world, allow_fake=bool(getattr(source, "is_fake", False))):
                chunks[chunk][1] += 1
                continue
            rows = [r for r in world.rows if bool(r["decisive"])]
            from .interface import validate_row

            for row in rows:
                validate_row(row)
            if not rows:
                continue
            for start in range(0, len(rows), 1024):
                part = rows[start : start + 1024]
                batch = {
                    key: np.stack([np.asarray(r[key]) for r in part])
                    for key in ("local", "global", "scalars")
                }
                if all("full_sight" in r for r in part):
                    batch["full_sight"] = np.stack([np.asarray(r["full_sight"]) for r in part])
                q = forward_batch(model, to_device(batch, device)).detach().cpu().numpy()
                if not np.isfinite(q).all():
                    raise FitTripwire("non-finite Q in held-out decisive scoring")
                pred = q.argmax(1)
                target = np.asarray([r["action_star"] for r in part])
                agree += int((pred == target).sum())
                total += len(part)
    if not total:
        raise ValueError("held-out decisive stream is empty; refusing vacuous selection")
    bad = {name: count for name, count in chunks.items() if count[1] / count[0] > 0.02}
    if bad:
        raise ReplayDivergenceError(f"held-out replay hash mismatch rate exceeds 2%: {bad}")
    return {"heldout_decisive_agreement": agree / total, "heldout_decisive_n": total}


def fit(
    *,
    source: ReplaySource,
    model: torch.nn.Module,
    previous_model: torch.nn.Module,
    device: str | torch.device = "cpu",
    epochs: int = 5,
    batch_size: int = 1024,
    pretrained: bool = True,
    loss_config: LossConfig = LossConfig(),
    probe_hook: ProbeHook | None = None,
    allow_fake: bool = False,
    seed: int = 0,
    max_updates: int | None = None,
) -> dict[str, Any]:
    """Fit for four to six epochs; returns only development metrics, never gate evidence."""
    if not 4 <= epochs <= 6 and max_updates is None:
        raise ValueError("P4 fitting requires four to six epochs")
    if bool(getattr(source, "is_fake", False)) and not allow_fake:
        raise ValueError("FakeReplaySource is smoke-only; pass allow_fake=True explicitly")
    if probe_hook is None and not bool(getattr(source, "is_fake", False)):
        raise ValueError("production fit requires the 32-world held-out H5000 probe hook")
    _cpu_guard()
    torch.manual_seed(seed)
    dev = torch.device(device)
    model, previous_model = model.to(dev).train(), previous_model.to(dev).eval()
    opt = optimizer_for(model, pretrained)
    loader = StreamingReplayLoader(source, batch_size=batch_size, seed=seed)
    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_optimizer_state: dict[str, Any] | None = None
    best_rng_state: torch.Tensor | None = None
    best_updates: int | None = None
    updates = 0
    last_progress = perf_counter()
    epoch_limit = 10000 if max_updates is not None else epochs
    for epoch in range(1, epoch_limit + 1):
        losses = []
        stop_training = False
        for raw in loader.epoch(epoch):
            if stop_training:
                # Consume EOF so full-world hash/drop accounting still decides
                # the preflight after its requested update count.
                continue
            batch = to_device(raw, dev)
            q = forward_batch(model, batch)
            loss, pieces = exit_fit_loss(q, batch, loss_config)
            # S1a-only hindsight supervision: only rows with a generated
            # target participate, with Huber beta=0.1 and the registered 0.1
            # coefficient. S1b has neither fields nor this head.
            if "hindsight_log_free_region" in batch:
                head = getattr(model, "aux_free_region", None)
                if not callable(head):
                    raise ValueError("hindsight target supplied to a model without aux_free_region")
                mask = batch["hindsight_log_free_region_mask"].bool()
                if mask.any():
                    prediction = head(
                        batch["local"], batch["global"], batch["scalars"], batch["full_sight"]
                    )
                    per_row = torch.nn.functional.smooth_l1_loss(
                        prediction[mask],
                        batch["hindsight_log_free_region"][mask],
                        beta=0.1,
                        reduction="none",
                    )
                    aux = (per_row * batch["sample_weight"][mask]).sum() / float(q.shape[0])
                    loss = loss + 0.1 * aux
                    pieces["aux_huber"] = aux
            if not torch.isfinite(loss):
                raise FitTripwire("non-finite loss")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            updates += 1
            losses.append(float(loss.detach()))
            if perf_counter() - last_progress >= 120:
                print(
                    json.dumps(
                        {"phase": "fit", "epoch": epoch, "updates": updates, "loss": losses[-1]}
                    ),
                    flush=True,
                )
                last_progress = perf_counter()
            if max_updates is not None and updates >= max_updates:
                stop_training = True
        if bool(getattr(source, "is_fake", False)):
            # A fake is only allowed for wiring.  It emits no quality claim.
            heldout = {"heldout_decisive_agreement": 1.0, "heldout_decisive_n": 0}
        else:
            heldout = _heldout_metrics(model, source, dev, epoch)
        if probe_hook is None:  # explicit fake smoke: cannot imply real H5000 evidence
            probe = {
                "survival": 0.0,
                "max_decile_disagreement": 0.0,
                "boost_share": 0.0,
                "previous_boost_share": 1.0,
                "max_action_share": 0.0,
            }
        else:
            probe = validate_probe(probe_hook(model, epoch))
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        if not bool(getattr(source, "is_fake", False)):
            enforce_tripwires(
                loss=mean_loss,
                heldout_decisive_agreement=heldout["heldout_decisive_agreement"],
                probe=probe,
            )
        row = {
            "epoch": epoch,
            "updates": updates,
            "loss": mean_loss,
            "regen_seconds": loader.last_stats.elapsed_seconds if loader.last_stats else None,
            "drop_rate": loader.last_stats.drop_rate if loader.last_stats else None,
            **heldout,
            "probe_survival": probe["survival"],
            "probe": probe,
        }
        history.append(row)
        if best is None or selection_key(row) > selection_key(best):
            best = row
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            best_optimizer_state = copy.deepcopy(opt.state_dict())
            best_rng_state = torch.get_rng_state().cpu().clone()
            best_updates = updates
        if max_updates is not None and updates >= max_updates:
            break
        model.train()
    return {
        "schema": "exit-fit-result/v1",
        "completed": True,
        "tripwire": None,
        "probes_ok": not bool(getattr(source, "is_fake", False)),
        "smoke_only_fake_engine": bool(getattr(source, "is_fake", False)),
        "history": history,
        "selected": best,
        "selected_state_dict": best_state,
        "selected_optimizer_state": best_optimizer_state,
        "selected_torch_rng_state": best_rng_state,
        "selected_updates": best_updates,
        "updates": updates,
    }


def run_fit_preflight(
    *,
    source: ReplaySource,
    model: torch.nn.Module,
    previous_model: torch.nn.Module,
    device: str,
    updates: int | None = 200,
    worlds: int = 2,
    batch_size: int = 1024,
    allow_fake: bool = False,
    epochs: int | None = None,
    probe_hook: ProbeHook | None = None,
    seed: int = 0,
) -> dict[str, object]:
    """P6's exact reusable 2-world/200-update preflight callable."""
    actual_worlds = getattr(source, "worlds", worlds)
    if actual_worlds != worlds:
        raise ValueError(f"preflight requires exactly {worlds} source worlds")
    if epochs is not None and not 4 <= epochs <= 6:
        raise ValueError("production fit epochs must be in [4, 6]")
    try:
        return fit(
            source=source,
            model=model,
            previous_model=previous_model,
            device=device,
            epochs=epochs or 4,
            batch_size=batch_size,
            allow_fake=allow_fake,
            max_updates=updates,
            probe_hook=probe_hook,
            seed=seed,
        )
    except (FitTripwire, ReplayDivergenceError) as error:
        return {
            "schema": "exit-fit-result/v1",
            "completed": False,
            "tripwire": str(error),
            "probes_ok": False,
            "smoke_only_fake_engine": bool(getattr(source, "is_fake", False)),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fake", action="store_true", help="explicit local fake smoke only")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if not args.fake:
        raise SystemExit(
            "production launcher is supplied by P6; this module only permits --fake locally"
        )
    from src.model.ego2s_network import Ego2sNet

    # Four rows/world keeps the mandated 2-world/200-update wiring check tiny.
    source = FakeReplaySource(rows_per_world=4)
    model, previous = Ego2sNet(7, 60), Ego2sNet(7, 60)
    result = run_fit_preflight(
        source=source,
        model=model,
        previous_model=previous,
        device=args.device,
        batch_size=4,
        allow_fake=True,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    # The caller that promotes a selected epoch consumes this CPU state dict;
    # receipts intentionally record only JSON metrics and identities.
    receipt = dict(result)
    receipt.pop("selected_state_dict", None)
    receipt.pop("selected_optimizer_state", None)
    receipt.pop("selected_torch_rng_state", None)
    (args.out / "preflight.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
