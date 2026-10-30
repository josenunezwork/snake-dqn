"""Workload entrypoint. Never scores assessment labels or allocates a runtime."""

from __future__ import annotations

import argparse
import faulthandler
import json
import os
import signal
import time
from pathlib import Path
from typing import Any

import torch

from research.doom_auc_smoke.data import load_gzip
from research.doom_auc_smoke.evidence import (
    ARMS,
    empty_ledger,
    json_once,
    latest,
    load_unit,
    publish,
    validate_ledger,
    verified,
)
from research.doom_auc_smoke.model import make_model, masked_bce
from research.spatial_student_pilot.observation import build_observation, compact_cat


def hardware(device: str) -> dict[str, Any]:
    return dict(
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        device=device,
        gpu=torch.cuda.get_device_name() if device == "cuda" else "none",
        source=os.environ.get("DOOM_SOURCE_COMMIT", "local"),
    )


def forward(
    model: torch.nn.Module, arm: str, data: dict, ids: torch.Tensor, device: str
) -> torch.Tensor:
    if arm == "b8":
        return model(data["phi"][ids].to(device))
    compact = compact_cat([data["compact"][i] for i in ids.tolist()])
    return model(*build_observation(compact, device))


def diagnostic_decision(logits: torch.Tensor, legal: torch.Tensor) -> dict[str, Any]:
    """Record a lowest-doom legal action, without influencing fitting or scoring.

    Negated doom logits preserve the lowest-probability ordering. Masking and
    Tensor.argmax stay on the logits' device so the qualified canary observer
    can attribute a real six-action CUDA decision to this entrypoint.
    """
    if logits.ndim != 2 or logits.shape[-1] != 6 or legal.shape != logits.shape:
        raise ValueError("diagnostic requires matching [batch, 6] logits and legal mask")
    legal = legal.to(device=logits.device, dtype=torch.bool)
    rows = legal.any(dim=-1).nonzero().flatten()
    if not len(rows):
        raise ValueError("diagnostic has no legal action")
    row = int(rows[0])
    values = logits.detach()[row : row + 1]
    mask = legal[row : row + 1]
    if not torch.isfinite(values).all():
        raise ValueError("nonfinite diagnostic logits")
    action = int((-values).masked_fill(~mask, -torch.inf).argmax(dim=-1).item())
    return dict(
        rule="lowest-doom-legal-first-index-tie",
        device=str(logits.device),
        row=row,
        action=action,
        legal=mask[0].tolist(),
        doom_logit=float(values[0, action]),
    )


def train_arm(
    arm: str,
    data: dict,
    output: Path,
    input_dir: Path,
    ledger: dict,
    identity: str,
    *,
    device: str,
    updates: int,
    batch: int,
    deadline: float,
    stopped: list[bool],
    checkpoint_every: int = 250,
) -> bool:
    """Only train indices and train targets enter the fitting closure."""
    prediction = next((r for r in ledger["units"] if r["unit"] == arm + "-prediction"), None)
    if prediction:
        load_unit(input_dir, prediction, identity)
        print(json.dumps(dict(stage="skip-verified-prediction", arm=arm)), flush=True)
        return True
    torch.manual_seed(343010)
    if device == "cuda":
        torch.cuda.manual_seed_all(343010)
    model = make_model(arm).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sampler = torch.Generator().manual_seed(343011)
    step = 0
    row = latest(ledger, arm)
    if row:
        state = load_unit(input_dir, row, identity)
        if state["hardware"] != hardware(device):
            raise ValueError("cross-hardware/library/source resume refused")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        sampler.set_state(state["sampler"])
        torch.set_rng_state(state["rng"])
        if device == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        step = state["step"]
        if step != int(row["unit"][-4:]) or not 0 < step <= updates:
            raise ValueError("ineligible saved training counter")
    published = {r["unit"] for r in ledger["units"]}

    def checkpoint() -> None:
        unit = f"{arm}-fit-{step:04d}"
        if step == 0 or unit in published:
            return
        publish(
            output,
            unit,
            identity,
            dict(
                model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                optimizer=optimizer.state_dict(),
                sampler=sampler.get_state(),
                rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if device == "cuda" else [],
                step=step,
                hardware=hardware(device),
            ),
        )
        published.add(unit)

    # Physical fitting view excludes assessment compacts/features and all assessment labels.
    train_ids = data["train"]
    fit = dict(compact=[data["compact"][i] for i in train_ids], phi=data["phi"][train_ids])
    targets, legal = data["train_y"], data["legal"][train_ids]
    last_print = time.monotonic()
    durations = []
    decision = None
    while step < updates:
        if stopped[0] or time.monotonic() >= deadline:
            checkpoint()
            return False
        start = time.monotonic()
        ids = torch.randint(len(train_ids), (batch,), generator=sampler)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = forward(model, arm, fit, ids, device)
        loss = masked_bce(logits, targets[ids].to(device), legal[ids].to(device))
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
        optimizer.step()
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError("nonfinite fitted parameters")
        step += 1
        if device == "cuda":
            torch.cuda.synchronize()
        durations.append(time.monotonic() - start)
        if decision is None:
            decision = diagnostic_decision(logits, legal[ids])
            print(json.dumps(dict(stage="diagnostic-decision", arm=arm, **decision)), flush=True)
        faulthandler.dump_traceback_later(900, exit=True)
        if step % checkpoint_every == 0 or step == updates:
            checkpoint()
        if step <= 3 or time.monotonic() - last_print >= 30 or step == updates:
            print(
                json.dumps(
                    dict(
                        stage="fit",
                        arm=arm,
                        done=step,
                        total=updates,
                        loss=float(loss.detach()),
                        grad_norm=float(norm),
                        step_seconds=durations[-1],
                        eta_seconds=(updates - step) * sum(durations[-20:]) / len(durations[-20:]),
                    )
                ),
                flush=True,
            )
            last_print = time.monotonic()
    checkpoint()
    if stopped[0] or time.monotonic() >= deadline:
        return False
    model.eval()
    predictions = []
    with torch.no_grad():
        for ids in data["assessment"].split(batch):
            if stopped[0] or time.monotonic() >= deadline:
                return False
            logits = forward(model, arm, data, ids, device)
            if decision is None:
                decision = diagnostic_decision(logits, data["legal"][ids])
                print(
                    json.dumps(dict(stage="diagnostic-decision", arm=arm, **decision)), flush=True
                )
            predictions.append(logits.cpu())
            faulthandler.dump_traceback_later(900, exit=True)
    scores = torch.cat(predictions)
    if not torch.isfinite(scores).all():
        raise ValueError("nonfinite final assessment logits")
    publish(
        output,
        arm + "-prediction",
        identity,
        dict(
            logits=scores,
            steps=step,
            hardware=hardware(device),
            root_uids=[data["root_uids"][i] for i in data["assessment"]],
            parameter_count=sum(p.numel() for p in model.parameters()),
            measured_step_seconds=durations[:3],
            diagnostic_decision=decision,
        ),
    )
    return True


def run(
    data: dict,
    output: Path,
    *,
    identity: str,
    ledger: dict,
    input_dir: Path,
    device: str,
    updates: int,
    batch: int,
    seconds: float,
    checkpoint_every: int = 250,
) -> dict[str, Any]:
    started = time.monotonic()
    validate_ledger(ledger, identity)
    for row in ledger["units"]:
        verified(input_dir, row, identity)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA required, no CPU fallback")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.reset_peak_memory_stats()
    faulthandler.enable()
    faulthandler.dump_traceback_later(900, exit=True)
    stopped = [False]
    previous = signal.signal(signal.SIGUSR1, lambda *_: stopped.__setitem__(0, True))
    completed = []
    try:
        for arm in ARMS:
            if train_arm(
                arm,
                data,
                output,
                input_dir,
                ledger,
                identity,
                device=device,
                updates=updates,
                batch=batch,
                deadline=started + seconds,
                stopped=stopped,
                checkpoint_every=checkpoint_every,
            ):
                completed.append(arm)
            else:
                break
    finally:
        signal.signal(signal.SIGUSR1, previous)
        faulthandler.cancel_dump_traceback_later()
    report = dict(
        status="PREDICTIONS_COMPLETE" if len(completed) == 2 else "INCOMPLETE",
        completed_arms=completed,
        identity=identity,
        wall_seconds=time.monotonic() - started,
        hardware=hardware(device),
        peak_vram_bytes=torch.cuda.max_memory_allocated() if device == "cuda" else 0,
        assessment_scored=False,
        updates=updates,
        batch=batch,
    )
    # Summary is operational telemetry, never a science decision/look.
    json_once(output / "summary.json", report)
    from research.doom_auc_smoke.evidence import atomic, digest

    atomic(output / "summary.json.sha256", (digest(output / "summary.json") + "\n").encode())
    atomic(output / "summary.json.complete", b"complete\n")
    return report


def tiny_view(data: dict, count: int = 8) -> dict:
    """Preflight uses real train roots; inference-only assessment has no outcome access."""
    result = dict(data)
    result["train"] = data["train"][:count]
    result["train_y"] = data["train_y"][:count]
    result["assessment"] = data["assessment"][:count]
    return result


def synthetic() -> dict:
    """Same arena, late bodies, public synthetic targets; never real canary fitting."""
    from research.doom_auc_smoke.fixtures import synthetic_dataset

    return synthetic_dataset()


def main() -> None:
    entered = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("canary", "fit", "preflight"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packet = args.input_dir.resolve()
    config = json.loads((packet / "config.json").read_text())
    from research.doom_auc_smoke.evidence import digest

    if digest(packet / "dataset.pt.gz") != config["dataset_sha256"]:
        raise ValueError("dataset identity differs")
    if args.mode == "canary":
        data, identity = synthetic(), "canary-" + config["identity"]
        ledger, updates, batch, seconds = empty_ledger(identity), 3, 32, 600
    else:
        data, identity = load_gzip(packet / "dataset.pt.gz"), config["identity"]
        ledger = json.loads((packet / "verified-ledger.json").read_text())
        updates, batch, seconds = 3000, 32, 2700
        if args.mode == "preflight":
            data, identity = tiny_view(data), "preflight-" + identity
            ledger, updates, batch, seconds = empty_ledger(identity), 3, 2, 120
    report = run(
        data,
        args.output.resolve(),
        identity=identity,
        ledger=ledger,
        input_dir=packet,
        device=args.device,
        updates=updates,
        batch=batch,
        seconds=max(0, seconds - (time.monotonic() - entered)),
    )
    print(json.dumps(report, sort_keys=True), flush=True)
    if report["status"] != "PREDICTIONS_COMPLETE":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
