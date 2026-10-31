"""CPU-only saved-data experiment; no assessment predictions or gameplay.

The original torch containers mix partitions and suffix families. Deserialization
necessarily reads those container bytes; numerical views immediately select train
roots, and scoring fields are accessed only after the OOF seal is published.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from research.doom_auc_smoke.data_mac import B8_SHA, BANK_SHA, LABEL_SHAS
from research.doom_auc_smoke.evidence import atomic, digest, json_once

ARTIFACTS = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts")
BANK = ARTIFACTS / (
    "t34-ct-h5000-1029/revision-02/attempt-02/" "output-a100-fallback/exports/ct/stage-0001.pt"
)
LABELS = ARTIFACTS / "t34-ct-h5000-stage2-1030/attempt-01/output-a100-fallback/exports/ct"
B8 = Path("/Users/josenunez/Projects/ml/snake-dqn/saved_snakes/b8_s3_t7_20261016.pth")
UPDATES, BATCH = 3000, 32


def save_tensor(path: Path, value: dict) -> str:
    """Publish create-only evidence with a SHA sidecar."""
    buffer = io.BytesIO()
    torch.save(value, buffer)
    atomic(path, buffer.getvalue())
    sha = digest(path)
    atomic(Path(str(path) + ".sha256"), (sha + "\n").encode())
    return sha


def choose(
    values: torch.Tensor, legal: torch.Tensor, b8: torch.Tensor, positive: bool = True
) -> torch.Tensor:
    """Largest legal advantage, B8 on exact ties or no positive prediction."""
    if values.shape != legal.shape or values.shape != (len(b8), 6):
        raise ValueError("unaligned action table")
    if not legal.any(1).all() or not legal.gather(1, b8[:, None]).all():
        raise ValueError("illegal B8 action or empty legal mask")
    if not torch.isfinite(values[legal]).all():
        raise ValueError("nonfinite legal values")
    masked = values.masked_fill(~legal, -torch.inf)
    maximum, action = masked.max(1)
    keep = masked.gather(1, b8[:, None]).flatten() == maximum
    if positive:
        keep |= maximum <= 0
    return torch.where(keep, b8, action)


def folds_for(worlds: torch.Tensor) -> torch.Tensor:
    """Five balanced folds of sorted parent worlds, shuffled with a fixed seed."""
    unique = torch.unique(worlds, sorted=True)
    order = unique[torch.randperm(len(unique), generator=torch.Generator().manual_seed(343013))]
    mapping = {int(w): i % 5 for i, w in enumerate(order)}
    return torch.tensor([mapping[int(w)] for w in worlds])


def checked_load(path: Path, sha: str, *, weights_only: bool = True) -> dict:
    if digest(path) != sha:
        raise ValueError(f"input SHA mismatch: {path}")
    return torch.load(path, map_location="cpu", weights_only=weights_only)


def training_bank(bank: Path, checkpoint: Path) -> tuple[dict, dict]:
    """Extract B8 features only for the original 2,048 training roots."""
    from research.sandbox_a_resident.ct_topology import decompress_u8_tensor_to_array
    from src.model.ego2s_network import load_ego2s_checkpoint

    raw = checked_load(bank, BANK_SHA, weights_only=False)
    data = raw["dataset"]
    if raw["schema"] != "ct-stage-0001/v1" or data["control"]["branch_horizon"] != 300:
        raise ValueError("foreign bank schema/horizon")
    metadata = data["store"]["metadata"]
    parts = data["world_partitions"]
    if len(metadata) != 2560 or set(parts) != {"train", "assessment"}:
        raise ValueError("foreign partition inventory")
    if any(
        m["root_index"] != i or m["partition"] != parts[m["world_id"]]
        for i, m in enumerate(metadata)
    ):
        raise ValueError("partition/root alignment differs")
    if len({m["root_uid"] for m in metadata}) != 2560:
        raise ValueError("duplicate root UID")
    ids = torch.tensor([i for i, m in enumerate(metadata) if m["partition"] == "train"])
    meta = [metadata[i] for i in ids.tolist()]
    worlds = torch.tensor([m["world_id"] for m in meta])
    if len(ids) != 2048 or len(torch.unique(worlds)) != 524:
        raise ValueError("expected 2048 roots / 524 training worlds")
    # No snapshot reconstruction, assessment model inference, or labels here.
    obs = {}
    for name, block in data["store"]["observations_compressed"].items():
        obs[name] = torch.from_numpy(decompress_u8_tensor_to_array(block)[ids.numpy()].copy())
    del raw, data, metadata
    legal = obs["legal_mask"].bool()
    b8 = torch.tensor([m["b8_action"] for m in meta])
    choose(obs["q_b8"], legal, b8, positive=False)
    if digest(checkpoint) != B8_SHA:
        raise ValueError("foreign B8")
    net = load_ego2s_checkpoint(str(checkpoint), "cpu").eval()
    net.requires_grad_(False)
    features, errors = [], []
    with torch.inference_mode():
        for start in range(0, len(ids), 32):
            sl = slice(start, start + 32)
            phi = net.fuse(
                torch.cat(
                    (
                        net.local(obs["local"][sl].float() / 255),
                        net.glob(obs["glob"][sl].float() / 255),
                        obs["scalars"][sl],
                    ),
                    1,
                )
            )
            adv = net.adv(phi)
            q = net.value(phi) + adv - adv.mean(1, keepdim=True)
            errors.append(float((q - obs["q_b8"][sl]).abs().max()))
            features.append(phi.clone())
    error = max(errors)
    if error > 1e-3:
        raise ValueError("B8 feature/Q parity failed")
    result = dict(
        ids=ids,
        world=worlds,
        phi=torch.cat(features).clone(),
        legal=legal,
        b8=b8,
        length=torch.tensor([m["hero_length"] for m in meta]),
        ordinary=torch.tensor([m["kind"] == "ordinary" for m in meta]),
        root_uids=[m["root_uid"] for m in meta],
    )
    return result, dict(
        bank_sha256=BANK_SHA,
        b8_sha256=B8_SHA,
        label_sha256=list(LABEL_SHAS),
        train_roots=2048,
        train_worlds=524,
        assessment_roots_used=0,
        q_max_abs_error=error,
    )


def outcomes(data: dict, labels: Path, phase: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Join only requested train roots, validate raw legal suffix outcomes."""
    if phase not in ("sel", "score"):
        raise ValueError("unknown suffix family")
    n = len(data["ids"])
    lookup = {int(root): i for i, root in enumerate(data["ids"])}
    mass, survival = torch.full((n, 6, 4), torch.nan), torch.full((n, 6, 4), torch.nan)
    seen = set()
    for index, sha in enumerate(LABEL_SHAS):
        row = checked_load(labels / f"stage-{index + 1:04d}.pt", sha)
        ids = row["root_indices"].tolist()
        if (
            row["schema"] != "ct-scored-root-chunk/v1"
            or row["chunk_index"] != index
            or row["telemetry"]["horizon"] != 300
            or len(ids) != 320
            or ids != list(range(index * 320, (index + 1) * 320))
        ):
            raise ValueError("foreign or incomplete chunk")
        source = torch.tensor([i for i, root in enumerate(ids) if root in lookup])
        dest = torch.tensor([lookup[ids[i]] for i in source.tolist()])
        if seen.intersection(dest.tolist()):
            raise ValueError("duplicate training roots")
        seen.update(dest.tolist())
        if not torch.equal(row["legal_mask"][source].bool(), data["legal"][dest]):
            raise ValueError("legal mask mismatch")
        if not torch.equal(row["b8_action"][source], data["b8"][dest]):
            raise ValueError("B8 action mismatch")
        mass[dest] = row[f"mass_{phase}_raw"][source]
        survival[dest] = row[f"survive_{phase}_raw"][source]
    if seen != set(range(n)):
        raise ValueError("missing training outcomes")
    mask = data["legal"]
    if (
        not torch.isfinite(mass[mask]).all()
        or (mass[mask] < 0).any()
        or not ((survival[mask] == 0) | (survival[mask] == 1)).all()
    ):
        raise ValueError("invalid legal H300 mass/survival")
    return mass, survival


def fit_fold(
    phi: torch.Tensor,
    target: torch.Tensor,
    legal: torch.Tensor,
    train: torch.Tensor,
    test: torch.Tensor,
    updates: int = UPDATES,
    deadline: float = float("inf"),
) -> tuple[torch.Tensor, dict]:
    """Fresh head, train-fold feature normalization, final checkpoint only."""
    torch.manual_seed(343010)
    model = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 6))
    mean = phi[train].mean(0)
    std = phi[train].std(0, unbiased=False).clamp_min(1e-6)
    x = (phi[train] - mean) / std
    y, mask = target[train], legal[train]
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sampler = torch.Generator().manual_seed(343011)
    started = time.monotonic()
    for step in range(updates):
        if time.monotonic() >= deadline:
            raise TimeoutError("one-hour experiment timeout")
        ids = torch.randint(len(train), (BATCH,), generator=sampler)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x[ids])
        loss = (prediction[mask[ids]] - y[ids][mask[ids]]).square().mean()
        if not torch.isfinite(loss):
            raise ValueError("nonfinite regression loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
        optimizer.step()
        if (step + 1) % 1000 == 0:
            print(
                json.dumps(
                    dict(
                        stage="fit",
                        step=step + 1,
                        loss=float(loss.detach()),
                        seconds=time.monotonic() - started,
                    )
                ),
                flush=True,
            )
    model.eval()
    with torch.no_grad():
        prediction = model((phi[test] - mean) / std)
    if not torch.isfinite(prediction).all():
        raise ValueError("nonfinite final predictions")
    state = dict(
        model=model.state_dict(),
        mean=mean,
        std=std,
        updates=updates,
        optimizer=optimizer.state_dict(),
        train=train,
        test=test,
        sampler=sampler.get_state(),
        rng=torch.get_rng_state(),
        final_loss=float(loss.detach()),
        seconds=time.monotonic() - started,
    )
    return prediction, state


def interval(point: float, draws: np.ndarray) -> dict:
    draws = draws[np.isfinite(draws)]
    if len(draws) < 1900:
        raise ValueError("insufficient defined bootstrap draws")
    return dict(
        point=float(point),
        lb90=float(np.quantile(draws, 0.10)),
        ci95=[float(v) for v in np.quantile(draws, [0.025, 0.975])],
    )


def summarize(
    world: np.ndarray,
    mass: np.ndarray,
    survival: np.ndarray,
    b8: np.ndarray,
    chosen: np.ndarray,
    teacher: np.ndarray,
) -> dict:
    """Root-weighted means; resample whole parent worlds with all roots/suffixes."""
    n = len(world)
    idx = np.arange(n)
    bm, cm, tm = (mass[idx, a].mean(1) for a in (b8, chosen, teacher))
    bs, cs, ts = (survival[idx, a].mean(1) for a in (b8, chosen, teacher))
    death = survival[idx, b8].sum(1) <= 1  # B8 dies in >=3/4 suffixes.
    rescued = (survival[idx, chosen].sum(1) >= 3) & death
    changed = chosen != b8
    # Each suffix is paired: B8 dies, chosen survives in that SAME suffix.
    events = 1 - survival[idx, b8]
    event_rescues = events * survival[idx, chosen]
    table = np.column_stack(
        (
            cm - bm,
            tm - bm,
            cs - bs,
            ts - bs,
            bm,
            rescued,
            death,
            changed & ~death,
            ~death,
            event_rescues.sum(1),
            events.sum(1),
            changed,
            np.ones(n),
        )
    )
    unique, inverse = np.unique(world, return_inverse=True)
    if len(unique) < 32:
        raise ValueError("fewer than 32 parent worlds in reported stratum")
    totals = np.zeros((len(unique), table.shape[1]))
    np.add.at(totals, inverse, table)
    rng = np.random.default_rng(343012)
    bootstrap = np.stack(
        [totals[rng.integers(len(unique), size=len(unique))].sum(0) for _ in range(2000)]
    )
    total = totals.sum(0)

    def ratio(num: int, den: int) -> dict:
        with np.errstate(divide="ignore", invalid="ignore"):
            samples = bootstrap[:, num] / bootstrap[:, den]
        if total[den] == 0:
            return dict(point=None, lb90=None, ci95=None)
        return interval(total[num] / total[den], samples)

    return dict(
        roots=n,
        worlds=len(unique),
        b8_mass=float(bm.mean()),
        b8_survival=float(bs.mean()),
        chosen_mass=ratio(0, 12),
        teacher_mass=ratio(1, 12),
        chosen_survival=ratio(2, 12),
        teacher_survival=ratio(3, 12),
        relative_mass=ratio(0, 4),
        teacher_retention=ratio(0, 1) if total[1] > 0 else None,
        rescue=ratio(5, 6),
        b8_death_roots=int(death.sum()),
        rescued_roots=int(rescued.sum()),
        nondeath_change=ratio(7, 8),
        nondeath_roots=int((~death).sum()),
        changed_nondeath=int((changed & ~death).sum()),
        paired_suffix_rescue=ratio(9, 10),
        change_fraction=ratio(11, 12),
    )


def decide(stats: dict) -> tuple[str, dict]:
    late, ordinary = stats["late"], stats["ordinary"]
    rescue = late["rescue"]["point"]
    clauses = dict(
        late_lb90_positive=late["chosen_mass"]["lb90"] > 0,
        retains_half_teacher=(
            late["teacher_mass"]["point"] > 0
            and late["chosen_mass"]["point"] >= 0.5 * late["teacher_mass"]["point"]
        ),
        ordinary_relative_lb90=ordinary["relative_mass"]["lb90"] >= -0.05,
        ordinary_survival_loss=ordinary["chosen_survival"]["point"] >= -0.01,
    )
    if late["teacher_mass"]["point"] <= 0 or (rescue is not None and rescue < 0.20):
        return "KILL", clauses
    if rescue is None:
        return "INCOMPLETE_SUPPORT", clauses
    return ("GO" if all(clauses.values()) else "GRAY"), clauses


def report(result: dict) -> str:
    lines = [
        "# B8 head test — exploratory saved-data result",
        "",
        f"**Verdict: {result['verdict']}**. No serving or promotion claim.",
        "",
        "2,048 original training roots / 524 parent worlds; five world-disjoint folds. "
        "No assessment features, predictions, targets, or metrics were used. "
        "Late = hero length >=600; ordinary = non-lookback roots (all lengths).",
        "",
        "H300 full-horizon mean mass (dead frames zero), forced first action then B8 "
        "continuation; four selection suffixes for fitting/teacher, four independent "
        "scoring suffixes after the OOF action seal. Survival = alive at H300.",
        "",
        "| Stratum | Roots / worlds | Chosen−B8 mass: point; LB90; CI95 | "
        "Teacher−B8 mass: point; LB90; CI95 | Chosen−B8 survival (pp): point; CI95 |",
        "|---|---:|---|---|---|",
    ]
    for name, row in result["stats"].items():

        def fmt(key: str, scale: float = 1) -> str:
            v = row[key]
            return (
                f"{v['point']*scale:.3f}; {v['lb90']*scale:.3f}; "
                f"[{v['ci95'][0]*scale:.3f}, {v['ci95'][1]*scale:.3f}]"
            )

        lines.append(
            f"| {name} | {row['roots']} / {row['worlds']} | {fmt('chosen_mass')} | "
            f"{fmt('teacher_mass')} | {fmt('chosen_survival', 100)} |"
        )
    lines += [
        "",
        "Parent-world percentile bootstrap: 2,000 draws, seed 343012; LB90 = "
        "10th percentile (one-sided 90% bound); CI95 = 2.5th–97.5th percentiles. "
        "Means weight roots equally, retaining all roots/suffixes when a world is drawn. "
        "Intervals condition on this fixed fitted OOF ensemble; they do not refit heads.",
        "",
        "## Decision clauses",
        "",
    ]
    for name, passed in result["clauses"].items():
        lines.append(f"- {name}: {'PASS' if passed else 'FAIL'}")
    l, o = result["stats"]["late"], result["stats"]["ordinary"]
    retention = l["teacher_retention"]
    lines += [
        "",
        (
            f"Late teacher gain retained: {retention['point']*100:.2f}% "
            f"(CI95 [{retention['ci95'][0]*100:.2f}, {retention['ci95'][1]*100:.2f}]%)."
            if retention
            else "Late teacher gain is nonpositive; retention undefined."
        ),
        f"Ordinary mass relative delta: {o['relative_mass']['point']*100:.3f}%; "
        f"LB90 {o['relative_mass']['lb90']*100:.3f}%; "
        f"CI95 {[round(x*100, 3) for x in o['relative_mass']['ci95']]}%.",
        "",
        "## Fable diagnostics",
        "",
        "B8-death root = B8 dies in >=3/4 scoring suffixes; rescue = chosen survives "
        "in >=3/4. Non-death roots are the complement, including 2/4 mixed roots. "
        "No Q-margin sweep or second fit. Paired-suffix rescue is also reported separately.",
        "",
        "| Stratum | Rescue % [CI95] | Non-death changed % [CI95] | "
        "Paired-suffix rescue % [CI95] |",
        "|---|---|---|---|",
    ]
    for name, row in result["stats"].items():

        def pct(key: str) -> str:
            v = row[key]
            if v["point"] is None:
                return "undefined"
            return f"{v['point']*100:.2f} [{v['ci95'][0]*100:.2f}, {v['ci95'][1]*100:.2f}]"

        lines.append(
            f"| {name} | {pct('rescue')} | {pct('nondeath_change')} | "
            f"{pct('paired_suffix_rescue')} |"
        )
    lines += [
        "",
        f"Fable's requested late bar (rescue >=35%, non-death changes <=15%): "
        f"{'PASS' if result['fable_bar'] else 'FAIL'}.",
        "",
        "## Fit and provenance",
        "",
        "Fresh 256→128→6 ReLU head per fold; frozen B8 encoder. Train-fold mean/std "
        "only. Legal-action MSE in raw mass-advantage units, including B8 zero targets. "
        "3,000 AdamW updates, batch 32, lr 3e-4, weight decay 1e-4, gradient clip 5; "
        "initialization seed 343010, sampler seed 343011; final checkpoint only. "
        "Fold permutation seed 343013, sorted parent worlds assigned round-robin. "
        "Policy: largest positive predicted legal advantage, otherwise B8; B8 wins exact "
        "ties. Teacher: selection-mean-mass argmax, B8 wins exact ties. "
        "No target scaling, encoder update, sweep, or model selection.",
        "",
        f"Experiment wall time: {result['wall_seconds']:.3f} s. "
        f"CPU process time: {result['cpu_seconds']:.3f} s. "
        f"Fit total: {result['fit_seconds']:.3f} s. Mac CPU, nice 15, one math thread.",
        "",
        f"Source base: `{result['source_commit']}`; run-source SHA256: "
        f"`{result['source_sha256']}`. OOF seal SHA256: `{result['oof_sha256']}`.",
        "",
        "All original bank/B8/eight chunk hashes are checked against the doom smoke pins; "
        "see inputs.json. Checkpoints, folds, raw OOF predictions/actions, scoring tables "
        "and machine-readable intervals are retained beside this report.",
        "",
        "Container boundary: the original torch archives physically mix train/assessment "
        "rows and selection/scoring fields. Their bytes are hash-read and deserialized; "
        "only training rows enter numerical views, B8 extraction, fits, or scoring. "
        "Scoring fields are not accessed until after the complete OOF seal. "
        "Original assessment artifacts are never opened.",
        "",
        "This is exploratory reuse of an enriched training bank, not a fresh gameplay "
        "test. Ordinary and late strata overlap; ordinary_late is their intersection. "
        "Cluster CIs describe this bank and fixed OOF fits, not repeated deployment. "
        "A GO would authorize only considering a fresh closed-loop development screen.",
        "",
        "Independent review: NOT RUN. Argon Gemini quota is 0% (live /usage check); "
        "no substitute reviewer or author self-approval. Code/tests/result are available "
        "for the required later Argon review.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bank", type=Path, default=BANK)
    parser.add_argument("--labels", type=Path, default=LABELS)
    parser.add_argument("--b8", type=Path, default=B8)
    args = parser.parse_args()
    started, cpu_started = time.monotonic(), time.process_time()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if os.getpriority(os.PRIO_PROCESS, 0) < 10:
        raise ValueError("launch with nice -n 15")
    args.output.mkdir(parents=True, exist_ok=False)
    data, audit = training_bank(args.bank, args.b8)
    fold = folds_for(data["world"])
    json_once(args.output / "inputs.json", audit)
    json_once(
        args.output / "folds.json",
        dict(
            root_ids=data["ids"].tolist(),
            world=data["world"].tolist(),
            fold=fold.tolist(),
            root_uids=data["root_uids"],
        ),
    )
    print(json.dumps(dict(stage="features-ready", **audit)), flush=True)
    mass, _ = outcomes(data, args.labels, "sel")
    mean_mass = mass.double().mean(-1)
    target = (mean_mass - mean_mass.gather(1, data["b8"][:, None])).float()
    teacher = choose(mean_mass, data["legal"], data["b8"], positive=False)
    prediction = torch.full((2048, 6), torch.nan)
    fits = []
    for f in range(5):
        train, test = (fold != f).nonzero().flatten(), (fold == f).nonzero().flatten()
        if set(data["world"][train].tolist()) & set(data["world"][test].tolist()):
            raise ValueError("world leakage")
        print(
            json.dumps(dict(stage="fold-start", fold=f, train=len(train), test=len(test))),
            flush=True,
        )
        values, state = fit_fold(
            data["phi"], target, data["legal"], train, test, deadline=started + 3600
        )
        prediction[test] = values
        save_tensor(args.output / f"fold-{f}-final.pt", state)
        fits.append(
            dict(
                fold=f,
                train=len(train),
                test=len(test),
                seconds=state["seconds"],
                final_loss=state["final_loss"],
            )
        )
    chosen = choose(prediction, data["legal"], data["b8"])
    seal = dict(
        schema="b8-head-oof/v1",
        ids=data["ids"],
        root_uids=data["root_uids"],
        fold=fold,
        world=data["world"],
        length=data["length"],
        ordinary=data["ordinary"],
        prediction=prediction,
        legal=data["legal"],
        b8=data["b8"],
        chosen=chosen,
        teacher=teacher,
        fits=fits,
    )
    oof_sha = save_tensor(args.output / "OOF-SEALED.pt", seal)
    json_once(
        args.output / "OOF-SEALED.json",
        dict(sha256=oof_sha, roots=2048, scoring_fields_accessed=False, unix_time=time.time()),
    )
    print(json.dumps(dict(stage="oof-sealed", sha256=oof_sha)), flush=True)
    # Independent scoring fields enter only after the immutable complete OOF seal.
    mass, survival = outcomes(data, args.labels, "score")
    if digest(args.output / "OOF-SEALED.pt") != oof_sha:
        raise ValueError("OOF seal changed")
    save_tensor(
        args.output / "training-score.pt", dict(ids=data["ids"], mass=mass, survival=survival)
    )
    strata = dict(
        all=torch.ones(2048, dtype=torch.bool),
        late=data["length"] >= 600,
        ordinary=data["ordinary"],
        ordinary_late=data["ordinary"] & (data["length"] >= 600),
        death_lookback=~data["ordinary"],
    )
    stats = {
        name: summarize(
            data["world"][mask].numpy(),
            mass[mask].numpy(),
            survival[mask].numpy(),
            data["b8"][mask].numpy(),
            chosen[mask].numpy(),
            teacher[mask].numpy(),
        )
        for name, mask in strata.items()
    }
    verdict, clauses = decide(stats)
    result = dict(
        verdict=verdict,
        clauses=clauses,
        stats=stats,
        fits=fits,
        wall_seconds=time.monotonic() - started,
        cpu_seconds=time.process_time() - cpu_started,
        fit_seconds=sum(f["seconds"] for f in fits),
        oof_sha256=oof_sha,
        fable_bar=(
            stats["late"]["rescue"]["point"] >= 0.35
            and stats["late"]["nondeath_change"]["point"] <= 0.15
        ),
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_sha256=digest(Path(__file__)),
    )
    json_once(args.output / "result.json", result)
    atomic(args.output / "RESULT.md", report(result).encode())
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
