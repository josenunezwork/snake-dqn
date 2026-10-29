"""Collect shared complete lives; numerical analysis is a separate Mac-only command."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from research.sandbox_a_resident.training_compact import decode_stream
from research.sandbox_a_resident.training_learner import critic_quality
from research.sandbox_a_resident.training_model import B8Actor, ClockedPopArtCritic, masked_logits
from research.sandbox_a_resident.training_rosters import (
    CHAMPION_SHA256,
    FRP3_S12_SHA256,
    MIXES,
    DeviceRosterRuntime,
)
from research.sandbox_a_resident.training_task import ResidentTrainingTask, TrainingTaskConfig
from research.sandbox_a_resident.units import digest, publish, verified_descriptor
from src.model.inference_agent import InferenceAgent

NAMESPACE = "a100-dev/sandbox-a-train01"
LEDGER_SCHEMA = "critic-rescore-ledger/v1"
B8_SHA = "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"


def fingerprint(control: dict) -> str:
    """Bind inputs, roots and collection recipe independently of attempt paths."""
    return hashlib.sha256(json.dumps(control, sort_keys=True).encode()).hexdigest()


def empty_ledger(control: dict) -> dict:
    """First-attempt ledger; supplied as a hash-bound input even when empty."""
    return dict(
        schema=LEDGER_SCHEMA, namespace=NAMESPACE, allocation=fingerprint(control), units=[]
    )


def specs(control: dict) -> list[tuple[str, int, int]]:
    """One baseline wave then a fixed maximum of fresh shared-holdout waves."""
    e = control["worlds"]
    return [("baseline", 0, control["baseline_offset"])] + [
        ("holdout", i, control["holdout_offset"] + i * e) for i in range(control["max_waves"])
    ]


def identity(phase: str, wave: int, offset: int) -> str:
    return f"{NAMESPACE}/critic-rescore/{phase}/{wave:04d}/{offset}"


def unit_filename(control: dict, phase: str, wave: int) -> str:
    """Bind namespace/recipe to the filename for outcome-blind Mac admission."""
    return f"{fingerprint(control)}-{phase}-{wave:04d}.json.gz"


def load_ledger(path: Path, control: dict) -> dict[str, dict]:
    """Refuse foreign or duplicated verified descriptors before any collection."""
    ledger = json.loads(path.read_text())
    if {k: ledger.get(k) for k in ("schema", "namespace", "allocation")} != {
        k: empty_ledger(control)[k] for k in ("schema", "namespace", "allocation")
    }:
        raise ValueError("foreign namespace/allocation")
    expected = {identity(*s) for s in specs(control)}
    found = {}
    for row in ledger["units"]:
        if row["id"] not in expected or row["id"] in found:
            raise ValueError("foreign/duplicate unit")
        if row["size"] < 1 or len(row["sha256"]) != 64:
            raise ValueError("malformed unit")
        found[row["id"]] = row
    return found


def write_json_unit(path: Path, value: dict) -> dict:
    """Deterministic compressed immutable payload with marker published last."""
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return publish(path, lambda p: p.write_bytes(gzip.compress(raw, mtime=0)))


def read_unit(path: Path) -> dict:
    return json.loads(gzip.decompress(path.read_bytes()))


def build_ledger(control: dict, paths: list[Path]) -> dict:
    """Outcome-blind Mac verification: descriptor path/size/hash, never decode data."""
    ledger = empty_ledger(control)
    expected = {unit_filename(control, p, w): identity(p, w, o) for p, w, o in specs(control)}
    for path in paths:
        if path.name not in expected:
            raise ValueError("foreign unit path")
        ledger["units"].append(verified_descriptor(path, expected[path.name]))
    # Enforce duplicates and allocation with the same loader as runtime.
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "ledger.json"
        p.write_text(json.dumps(ledger))
        load_ledger(p, control)
    return ledger


def load_critics(inputs: Path, actor: B8Actor, device: str) -> list[ClockedPopArtCritic]:
    """Decode byte-verified original snapshots and retain only frozen critics."""
    critics = []
    for seed in (1, 2):
        compressed = inputs / f"seed{seed}.pt.xz"
        meta = json.loads((inputs / f"seed{seed}.manifest").read_text())
        if digest(compressed) != meta["derivative"]["sha256"]:
            raise ValueError("critic source derivative mismatch")
        with tempfile.TemporaryDirectory(prefix="critic-decode-") as tmp:
            original = Path(tmp) / "iteration-0000.pt"
            with original.open("xb") as stream:
                decode_stream(
                    compressed,
                    stream.write,
                    expected_size=meta["original"]["size"],
                    expected_sha256=meta["original"]["sha256"],
                )
            state = torch.load(original, map_location="cpu", weights_only=False, mmap=True)
            if state["schema"] != "sandbox-a-train-state/v1" or state["iteration"] != 0:
                raise ValueError("expected prepared iteration-zero critic")
            critic = ClockedPopArtCritic(actor.reference).to(device).eval()
            critic.load_state_dict(state["critic"])
            critic.requires_grad_(False)
            critics.append(critic)
            del state
    return critics


def emit(output: Path, started: float, **fields: Any) -> None:
    """Operational support/progress only: no numerical quality look on runtime."""
    value = dict(elapsed_seconds=time.monotonic() - started, **fields)
    (output / "progress.json").write_text(json.dumps(value, sort_keys=True))
    print("CRITIC_RESCORE_PROGRESS " + json.dumps(value, sort_keys=True), flush=True)


@torch.inference_mode()
def collect_wave(
    control: dict,
    phase: str,
    wave: int,
    offset: int,
    actor: B8Actor,
    critics: list,
    frozen: dict,
    device: str,
    output: Path,
    started: float,
) -> dict:
    """Full natural lives, death/H5000; no resetting or right-censored targets."""
    e, horizon = control["worlds"], control["horizon"]
    task = ResidentTrainingTask(
        TrainingTaskConfig(
            worlds=e,
            namespace=NAMESPACE,
            allocation_offset=offset,
            allocation_phase_offset=offset,
            allocation_count=e,
            max_episodes=1,
            auto_reset=False,
            tiny=control["test_only"],
        ),
        device=device,
    )
    task.attach_opponents(
        DeviceRosterRuntime(
            task.sim, task.rosters, b8_network=actor.reference, checkpoint_networks=frozen
        )
    )
    # Tiny preflight shortens the horizon explicitly, production remains H5000.
    lengths = np.zeros((horizon, e), dtype=np.int32)
    rewards = np.zeros((horizon, e), dtype=np.float64)
    predictions = np.zeros((2, horizon, e), dtype=np.float32)
    active = torch.ones(e, dtype=torch.bool, device=device)
    ends = np.zeros(e, dtype=np.int32)
    reasons = [None] * e
    last_progress = time.monotonic()
    for tick in range(horizon):
        local, glob, scalars, legal = task.observe()
        mass = task.hero_length.clone()
        clock = (5000 - task.age).float() / 5000.0
        action = masked_logits(actor(local, glob, scalars), legal).argmax(-1)
        late = active & mass.ge(control["late_length"])
        if phase == "holdout" and bool(late.any()):
            for i, critic in enumerate(critics):
                value = critic(local[late], glob[late], scalars[late], clock[late])
                if not bool(torch.isfinite(value).all()):
                    raise ValueError("nonfinite critic predictions")
                predictions[i, tick, late.cpu().numpy()] = value.cpu().numpy()
        step = task.step(action)
        lengths[tick] = mass.cpu().numpy()
        rewards[tick] = step.reward.cpu().numpy()
        done = active & (step.terminal | step.cut | (tick + 1 == horizon))
        completed_slots = done.cpu().numpy()
        terminals = step.terminal.cpu().numpy()
        ends[completed_slots] = tick + 1
        for slot in np.flatnonzero(completed_slots):
            reasons[slot] = (
                "death" if terminals[slot] else ("fixture-cut" if control["test_only"] else "H5000")
            )
        active &= ~done
        if tick == 0 or time.monotonic() - last_progress >= 60 or not bool(active.any()):
            emit(
                output,
                started,
                phase=phase,
                wave=wave,
                tick=tick + 1,
                total_ticks=horizon,
                completed_lives=int((ends > 0).sum()),
                total_lives=e,
            )
            last_progress = time.monotonic()
        if not bool(active.any()):
            break
    lives = []
    for slot, end in enumerate(ends):
        if end <= 0:
            raise ValueError("censored life cannot enter shared holdout")
        reward = rewards[:end, slot]
        target = np.cumsum(reward[::-1])[::-1]
        life = dict(
            id=offset + slot,
            mix=task.rosters[slot].mix,
            length=lengths[:end, slot].tolist(),
            target=target.tolist(),
            rewards=reward.tolist(),
            end_reason=reasons[slot],
            clock=((5000 - np.arange(end)) / 5000).tolist(),
            predictions=predictions[:, :end, slot].tolist(),
        )
        lives.append(life)
    return dict(
        schema="critic-rescore-wave/v1",
        allocation=fingerprint(control),
        id=identity(phase, wave, offset),
        phase=phase,
        wave=wave,
        lives=lives,
    )


def validate_unit(payload: dict, control: dict, phase: str, wave: int, offset: int) -> None:
    """Verify structure and complete-life identities before merging any outputs."""
    if (
        payload.get("schema") != "critic-rescore-wave/v1"
        or payload.get("allocation") != fingerprint(control)
        or payload.get("id") != identity(phase, wave, offset)
        or payload.get("phase") != phase
        or payload.get("wave") != wave
    ):
        raise ValueError("foreign payload identity")
    lives = payload["lives"]
    if [life["id"] for life in lives] != list(range(offset, offset + control["worlds"])):
        raise ValueError("foreign or missing life allocation")
    for slot, life in enumerate(lives):
        n = len(life["length"])
        if not 0 < n <= control["horizon"] or life["mix"] != MIXES[slot % 3]:
            raise ValueError("invalid life support/mix")
        vectors = [life[k] for k in ("length", "clock", "target", "rewards")]
        vectors += life["predictions"]
        if len(life["predictions"]) != 2 or any(len(v) != n for v in vectors):
            raise ValueError("invalid row shapes")
        if any(not np.isfinite(v).all() for v in vectors):
            raise ValueError("nonfinite row payload")
        if not np.array_equal(life["target"], np.cumsum(np.array(life["rewards"])[::-1])[::-1]):
            raise ValueError("non-complete-life suffix targets")
        expected_clock = (5000 - np.arange(n)) / 5000
        if not np.array_equal(life["clock"], expected_clock):
            raise ValueError("foreign clock")
        reason = life["end_reason"]
        if reason == "death":
            if life["rewards"][-1] != 0:
                raise ValueError("death target includes terminal mass")
        elif reason == "H5000":
            if n != 5000 or control["test_only"]:
                raise ValueError("censored horizon")
        elif reason != "fixture-cut" or not control["test_only"] or n != control["horizon"]:
            raise ValueError("censored life")


def selected_late(lives: list[dict], control: dict) -> tuple[list[dict], dict]:
    """First K completed lives with at least one L>=800 pre-action row, per mix."""
    counts = {mix: 0 for mix in MIXES}
    selected = []
    for life in lives:
        if max(life["length"]) >= control["late_length"]:
            mix = life["mix"]
            if counts[mix] < control["late_lives_per_mix"]:
                selected.append(life)
                counts[mix] += 1
    return selected, counts


def collect(inputs: Path, output: Path, device: str, control_path: Path | None = None) -> None:
    control = json.loads((control_path or inputs / "control.json").read_text())
    if not control["test_only"] and (
        control["horizon"] != 5000
        or control["late_length"] != 800
        or control["late_lives_per_mix"] < 8
    ):
        raise ValueError("production endpoint requires H5000 and >=24 late lives")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA required")
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    emit(output, started, phase="loading", done=0, total=control["max_waves"] + 1)
    verified = load_ledger(inputs / "verified-ledger.json", control)
    for name, sha in control["input_hashes"].items():
        if digest(inputs / name) != sha:
            raise ValueError("hash-bound runtime input mismatch: " + name)
    if digest(inputs / "b8-s3.pth") != B8_SHA:
        raise ValueError("released B8 identity mismatch")
    for name, expected in (("frp3-s12.pth", FRP3_S12_SHA256), ("champion.pth", CHAMPION_SHA256)):
        if digest(inputs / name) != expected:
            raise ValueError("released opponent identity mismatch")
    actor = B8Actor(inputs / "b8-s3.pth").to(device).eval().requires_grad_(False)
    critics = load_critics(inputs, actor, device)
    frozen = {}
    for name, sha in (("frp3-s12.pth", FRP3_S12_SHA256), ("champion.pth", CHAMPION_SHA256)):
        frozen[sha] = InferenceAgent.from_checkpoint(
            str(inputs / name), device=torch.device(device)
        ).network
    heldout = []
    units = []
    for phase, wave, offset in specs(control):
        unit_id = identity(phase, wave, offset)
        filename = unit_filename(control, phase, wave)
        if unit_id in verified:
            descriptor = verified[unit_id]
            path = inputs / "prior" / filename
            if digest(path) != descriptor["sha256"] or path.stat().st_size != descriptor["size"]:
                raise ValueError("verified unit payload mismatch")
            payload = read_unit(path)
            action = "skip_verified"
        else:
            payload = collect_wave(
                control, phase, wave, offset, actor, critics, frozen, device, output, started
            )
            write_json_unit(output / filename, payload)
            action = "collected"
        validate_unit(payload, control, phase, wave, offset)
        units.append(unit_id)
        if phase == "holdout":
            heldout.extend(payload["lives"])
        _, counts = selected_late(heldout, control)
        emit(
            output,
            started,
            phase=phase,
            wave=wave,
            action=action,
            late_lives=counts,
            done=len(units),
            total=control["max_waves"] + 1,
        )
        if min(counts.values()) >= control["late_lives_per_mix"]:
            break
    _, counts = selected_late(heldout, control)
    complete = min(counts.values()) >= control["late_lives_per_mix"]
    summary = dict(
        schema="critic-rescore-collection/v1",
        complete=complete,
        allocation=fingerprint(control),
        units=units,
        late_lives=counts,
        actor_updates=0,
        critic_updates=0,
        numerical_analysis="MAC_ONLY",
        elapsed_seconds=time.monotonic() - started,
        test_only=control["test_only"],
    )
    write_json_unit(output / "collection.json.gz", summary)
    if not complete:
        raise RuntimeError("INCOMPLETE: finite collection exhausted before 8 late lives per mix")


def metric(y: np.ndarray, p: np.ndarray) -> dict:
    variance = float(np.var(y))
    return dict(
        rows=len(y),
        explained_variance=1 - float(np.var(p - y)) / max(variance, 1e-8),
        mse=float(np.mean((p - y) ** 2)),
        late_row_mean_baseline_mse=variance,
        mse_ratio_to_late_mean=float(np.mean((p - y) ** 2)) / max(variance, 1e-8),
    )


def analyse(lives: list[dict], seed: int, late_length: int, repeats: int) -> dict:
    """Stratified life-cluster bootstrap, preserving every row of sampled lives."""
    clusters = []
    per_life = []
    for life in lives:
        mask = np.array(life["length"]) >= late_length
        y, p = np.array(life["target"])[mask], np.array(life["predictions"])[seed][mask]
        if not len(y):
            continue
        result = metric(y, p)
        defined = len(y) >= 2 and float(np.var(y)) > 1e-8
        per_life.append(dict(id=life["id"], mix=life["mix"], ev_defined=defined, **result))
        r = p - y
        clusters.append(
            (life["mix"], np.array([len(y), y.sum(), (y * y).sum(), r.sum(), (r * r).sum()]))
        )
    if not clusters:
        return dict(status="NO_LATE_LIVES")

    def ev(s: np.ndarray) -> float:
        n, sy, sy2, sr, sr2 = s
        return 1 - (sr2 / n - (sr / n) ** 2) / max(sy2 / n - (sy / n) ** 2, 1e-8)

    totals = sum((s for _, s in clusters))
    rng = np.random.default_rng(20261029)
    samples, life_samples = [], []
    for _ in range(repeats):
        indices = []
        for mix in MIXES:
            group = [i for i, (m, _) in enumerate(clusters) if m == mix]
            if group:
                indices.extend(rng.choice(group, len(group), replace=True).tolist())
        samples.append(ev(sum((clusters[i][1] for i in indices))))
        values = [per_life[i]["explained_variance"] for i in indices if per_life[i]["ev_defined"]]
        if values:
            life_samples.append(float(np.mean(values)))
    n, sy, sy2, _, sr2 = totals
    variance = max(float(sy2 / n - (sy / n) ** 2), 0)
    return dict(
        late_lives=len(clusters),
        rows=int(n),
        explained_variance=ev(totals),
        life_clustered_ev_ci95=np.quantile(samples, [0.025, 0.975]).tolist(),
        mean_per_life_ev=(
            float(np.mean([p["explained_variance"] for p in per_life if p["ev_defined"]]))
            if any(p["ev_defined"] for p in per_life)
            else None
        ),
        mean_per_life_ev_ci95=(
            np.quantile(life_samples, [0.025, 0.975]).tolist() if life_samples else None
        ),
        mse=float(sr2 / n),
        late_row_mean_baseline_mse=variance,
        mse_ratio_to_late_mean=float(sr2 / n) / max(variance, 1e-8),
        per_life=per_life,
    )


def scarcity(lives: list[dict], late_length: int, repeats: int) -> dict:
    """Empirical finite-bank 160-row resampling; both row interpretations explicit."""
    all_ids, late_ids = [], []
    for life in lives:
        ids = np.where(np.array(life["length"]) >= late_length, life["id"], -1)
        all_ids.extend(ids.tolist())
        late_ids.extend(ids[ids >= 0].tolist())
    rng = np.random.default_rng(160)
    result = {}
    for name, values in (("160_all_rows", all_ids), ("160_late_rows", late_ids)):
        a = np.array(values)
        if len(a) < 160:
            result[name] = dict(status="INSUFFICIENT_ROWS", available=len(a))
            continue
        counts = []
        for _ in range(repeats):
            chosen = rng.choice(a, 160, replace=False)
            counts.append(len(set(chosen[chosen >= 0].tolist())))
        k = sum(c < 3 for c in counts)
        result[name] = dict(
            replicates=repeats,
            fewer_than_3_late_lives=k,
            frequency=k / repeats,
            monte_carlo_se=(k / repeats * (1 - k / repeats) / repeats) ** 0.5,
            distinct_late_lives_mean=float(np.mean(counts)),
        )
    # Reproduce the old collection's complete-life, 160-late-row stopping mechanism.
    failures = 0
    eligible = 0
    late_counts = [sum(v >= late_length for v in life["length"]) for life in lives]
    for _ in range(repeats):
        late_rows, late_lives, done = 0, 0, {m: 0 for m in MIXES}
        for i in rng.permutation(len(lives)):
            life = lives[i]
            count = late_counts[i]
            late_rows += count
            late_lives += count > 0
            done[life["mix"]] += 1
            if late_rows >= 160 and sum(done.values()) >= 40 and min(done.values()) >= 8:
                break
        if late_rows >= 160 and sum(done.values()) >= 40 and min(done.values()) >= 8:
            eligible += 1
            failures += late_lives < 3
    result["old_complete_life_stopping_permutation"] = dict(
        status="COMPLETE" if eligible == repeats else "INSUFFICIENT_SUPPORT",
        replicates=repeats,
        eligible_replicates=eligible,
        frequency_fewer_than_3_late_lives=failures / eligible if eligible else None,
    )
    return result


def aggregate(control: dict, paths: list[Path], repeats: int = 2000) -> dict:
    """Mac's single numerical look over the union of verified exported units."""
    build_ledger(control, paths)
    units = sorted(
        [read_unit(p) for p in paths], key=lambda u: (u["phase"] != "baseline", u["wave"])
    )
    seen, all_lives, baseline = set(), [], []
    for unit in units:
        match = next((s for s in specs(control) if identity(*s) == unit["id"]), None)
        if match is None:
            raise ValueError("foreign unit identity")
        validate_unit(unit, control, *match)
        if unit["allocation"] != fingerprint(control) or unit["id"] not in {
            identity(*s) for s in specs(control)
        }:
            raise ValueError("foreign unit allocation")
        for life in unit["lives"]:
            if life["id"] in seen:
                raise ValueError("duplicate life")
            seen.add(life["id"])
            (baseline if unit["phase"] == "baseline" else all_lives).append(life)
    if len(baseline) != control["worlds"]:
        raise ValueError("missing independent baseline bank")
    selected, counts = selected_late(all_lives, control)
    if min(counts.values()) < control["late_lives_per_mix"]:
        raise ValueError("INCOMPLETE late-life support")
    x = np.concatenate(
        [
            np.stack([np.ones(len(life["length"])), life["clock"], life["length"]], axis=1)
            for life in baseline
        ]
    )
    yfit = np.concatenate([life["target"] for life in baseline])
    coef = np.linalg.lstsq(x, yfit, rcond=None)[0]
    result = dict(
        schema="critic-rescore-result/v1",
        tier="simulator-D development diagnostic",
        allocation=fingerprint(control),
        actor_updates=0,
        critic_updates=0,
        late_lives=counts,
        selected_life_ids=[life["id"] for life in selected],
        completed_holdout_lives=len(all_lives),
        baseline_lives=len(baseline),
        fresh_fit_time_length_baseline_coefficients=coef.tolist(),
        bootstrap_replicates=repeats,
        bootstrap="stratified by mix; whole lives with replacement",
        scarcity=scarcity(all_lives, control["late_length"], repeats),
        critics={},
    )
    # Old-gate diagnostic: original coefficients were not saved; refit only the
    # independent fresh baseline bank and disclose this comparator explicitly.
    for seed in range(2):
        entry = analyse(selected, seed, control["late_length"], repeats)
        entry["per_mix"] = {
            m: analyse(
                [life for life in selected if life["mix"] == m],
                seed,
                control["late_length"],
                repeats,
            )
            for m in MIXES
        }
        vectors = [
            torch.tensor(np.concatenate([np.array(life[k]) for life in all_lives]))
            for k in ("target", "length", "clock")
        ]
        pred = torch.tensor(np.concatenate([life["predictions"][seed] for life in all_lives]))
        # Recompute the original row-pooled gate on the entire collected bank.
        # The primary analysis above retains exactly K late lives per mix.
        support = torch.tensor([sum(life["mix"] == m for life in all_lives) for m in MIXES])
        try:
            entry["old_gate"] = critic_quality(
                pred,
                vectors[0],
                vectors[1],
                vectors[2],
                torch.tensor(coef),
                support,
                late_length=control["late_length"],
            )
        except ValueError as error:
            entry["old_gate"] = dict(status="INSUFFICIENT_SUPPORT", reason=str(error))
        entry["old_gate_comparator"] = (
            "independent fresh-bank time+length OLS; original coefficients not saved"
        )
        result["critics"][f"seed{seed+1}"] = entry
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["collect", "ledger", "analyse"])
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control", type=Path)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--units", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    if args.command == "collect":
        collect(args.inputs, args.output, args.device, args.control)
    else:
        control = json.loads((args.inputs / "control.json").read_text())
        value = (
            build_ledger(control, args.units)
            if args.command == "ledger"
            else aggregate(control, args.units)
        )
        with args.output.open("x") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")


if __name__ == "__main__":
    main()
