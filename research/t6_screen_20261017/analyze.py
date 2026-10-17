"""Decision and registered readouts for the T6 RESTART screen.

This module deliberately keeps the screen decision independent of diagnostic
readouts.  The same functions are used by the watcher while rows arrive and by
the final, CPU-only report pass.
"""

from __future__ import annotations

import hashlib
import json
import math
from fractions import Fraction
from numbers import Real
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from research.t2_screen_20261015.analyze_t2 import _mean, _number_map, _se, _survival_maps

ARM = "RESTART"
SEEDS = tuple(range(10, 15))
MIXES = ("scripted", "frozen", "mixed")
INITIAL_ROUND, EXTENSION_ROUND, FRAMES = 72, 73, 5000
ARTIFACTS = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts")
B8_ROOT = ARTIFACTS / "redesign-m4b-mac-20261013"
HOLDOUT_ROOT = ARTIFACTS / "t4e-return-bank-20261016/banks/holdout"
HOLDOUT_MANIFEST_SHA256 = "c0ae22541d20927d1a7996f1a5a91c2aceb9591de8eb743c514def86e2c95e6c"


def _group(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[str, int, str], Mapping[str, Any]]:
    grouped: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for row in rows:
        try:
            key = (str(row["name"]), int(row["round_index"]), str(row["mix"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("every T6 row needs name, round_index, and mix") from exc
        if key in grouped:
            raise ValueError("duplicate T6 probe row: %r" % (key,))
        grouped[key] = row
    return grouped


def _metric(row: Mapping[str, Any], name: str) -> Mapping[str, float]:
    if name == "episode_survival":
        return _survival_maps(row)[0]
    if name == "alive_survival":
        return _survival_maps(row)[1]
    return _number_map(row, name)


def _paired(
    grouped: Mapping[tuple[str, int, str], Mapping[str, Any]], mix: str, metric: str
) -> dict[str, Any]:
    per_seed: list[float] = []
    pooled: list[float] = []
    seeds: list[int] = []
    exact_survival: list[Fraction] = []
    rounds = (INITIAL_ROUND, EXTENSION_ROUND) if mix == "scripted" else (INITIAL_ROUND,)
    for seed in SEEDS:
        left_all: dict[str, float] = {}
        right_all: dict[str, float] = {}
        for round_index in rounds:
            left, right = (
                grouped.get((f"{ARM}_s{seed}", round_index, mix)),
                grouped.get((f"CTRL_s{seed}", round_index, mix)),
            )
            if left is None or right is None:
                continue
            a, b = _metric(left, metric), _metric(right, metric)
            if set(a) != set(b) or set(left_all) & set(a):
                raise ValueError("T6 paired worlds must match and not repeat")
            left_all.update(a)
            right_all.update(b)
        if left_all:
            delta = [left_all[key] - right_all[key] for key in sorted(left_all)]
            per_seed.append(_mean(delta))
            if metric == "episode_survival":
                if any(value not in (-1, 0, 1) for value in delta):
                    raise ValueError("episode survival differences must be integer counts")
                exact_survival.append(Fraction(sum(int(value) for value in delta), len(delta)))
            pooled.extend(delta)
            seeds.append(seed)
    exact_mean = sum(exact_survival, Fraction()) / len(exact_survival) if exact_survival else None
    return {
        "mean": float(exact_mean) if exact_mean is not None else _mean(per_seed),
        "exact_mean_numerator": exact_mean.numerator if exact_mean is not None else None,
        "exact_mean_denominator": exact_mean.denominator if exact_mean is not None else None,
        "seed_se": _se(per_seed),
        "pooled_world_se": _se(pooled),
        "seeds": seeds,
        "seed_n": len(per_seed),
        "world_n": len(pooled),
        "complete": seeds == list(SEEDS),
    }


def _incumbent(
    grouped: Mapping[tuple[str, int, str], Mapping[str, Any]], mix: str
) -> dict[str, Any]:
    baseline = grouped.get(("baseline", INITIAL_ROUND, mix))
    base = _number_map(baseline, "mass") if baseline else {}
    per_seed: list[float] = []
    pooled: list[float] = []
    for seed in SEEDS:
        row = grouped.get((f"{ARM}_s{seed}", INITIAL_ROUND, mix))
        candidate = _number_map(row, "mass") if row else {}
        if candidate and base and set(candidate) != set(base):
            raise ValueError("RESTART and incumbent worlds must match")
        if candidate and base:
            delta = [candidate[key] - base[key] for key in sorted(base)]
            per_seed.append(_mean(delta))
            pooled.extend(delta)
    return {
        "mean": _mean(per_seed),
        "seed_se": _se(per_seed),
        "pooled_world_se": _se(pooled),
        "seed_n": len(per_seed),
        "world_n": len(pooled),
    }


def _shared_lengths(grouped: Mapping[tuple[str, int, str], Mapping[str, Any]]) -> dict[str, Any]:
    per_mix: dict[str, Any] = {}
    for mix in MIXES:
        per_seed, pooled = [], []
        for seed in SEEDS:
            left, right = grouped.get((f"{ARM}_s{seed}", INITIAL_ROUND, mix)), grouped.get(
                (f"CTRL_s{seed}", INITIAL_ROUND, mix)
            )
            if left is None or right is None:
                continue
            dead_a, dead_b = _number_map(left, "deaths"), _number_map(right, "deaths")
            length_a, length_b = _number_map(left, "length_at_5000"), _number_map(
                right, "length_at_5000"
            )
            if (
                set(dead_a) != set(dead_b)
                or set(length_a) != set(dead_a)
                or set(length_b) != set(dead_a)
            ):
                raise ValueError("T6 length_at_5000 must align with paired worlds")
            shared = [key for key in sorted(dead_a) if not dead_a[key] and not dead_b[key]]
            values = [length_a[key] - length_b[key] for key in shared]
            per_seed.append(
                {"seed": seed, "shared_survivors": len(shared), "mean_delta": _mean(values)}
            )
            pooled.extend(values)
        seed_values = [row["mean_delta"] for row in per_seed if math.isfinite(row["mean_delta"])]
        per_mix[mix] = {
            "per_seed": per_seed,
            "shared_survivors": len(pooled),
            "pooled_mean_delta": _mean(pooled),
            "five_seed_mean_delta": _mean(seed_values),
            "five_seed_n": len(seed_values),
        }
    return {"frame": FRAMES, "by_mix": per_mix}


def _decision(data: Mapping[str, Any], statuses: Mapping[str, int], extended: bool) -> str:
    if any(code != 0 for code in statuses.values()):
        return "KILL"
    primary = data["vs_ctrl"]["scripted"]["episode_survival"]["mean"]
    if not math.isfinite(primary):
        return "INCOMPLETE"

    def survival_fraction(mix: str) -> Fraction:
        value = data["vs_ctrl"][mix]["episode_survival"]
        return Fraction(value["exact_mean_numerator"], value["exact_mean_denominator"])

    primary = survival_fraction("scripted")
    if primary <= 0:
        return "KILL"
    complete = all(
        data["vs_ctrl"][mix][metric]["complete"]
        for mix in MIXES
        for metric in ("episode_survival", "alive_survival", "mass")
    )
    if not complete:
        return "INCOMPLETE"
    if (
        primary >= Fraction(3, 50)
        and all(data["vs_ctrl"][mix]["mass"]["mean"] >= -30 for mix in MIXES)
        and all(survival_fraction(mix) >= Fraction(-1, 20) for mix in ("frozen", "mixed"))
    ):
        return "GO"
    return "INCONCLUSIVE" if extended else "EXTEND"


def _validate(
    grouped: Mapping[tuple[str, int, str], Mapping[str, Any]], statuses: Mapping[str, int]
) -> None:
    expected = {f"{ARM}_s{seed}" for seed in SEEDS}
    if not set(statuses).issubset(expected) or any(
        type(code) is not int for code in statuses.values()
    ):
        raise ValueError("T6 requires integer RESTART statuses")
    if any(code != 0 for code in statuses.values()):
        return
    if set(statuses) != expected:
        raise ValueError("T6 requires all five RESTART statuses unless a run halted")
    required = {
        (f"{name}_s{seed}", INITIAL_ROUND, mix)
        for name in (ARM, "CTRL")
        for seed in SEEDS
        for mix in MIXES
    }
    required |= {("baseline", INITIAL_ROUND, mix) for mix in MIXES}
    if not required.issubset(grouped):
        raise ValueError("round 72 requires RESTART, CTRL, and incumbent rows")
    names = {"baseline"} | {f"{arm}_s{seed}" for arm in (ARM, "CTRL") for seed in SEEDS}
    for (name, round_index, mix), row in grouped.items():
        valid = (name == "baseline" and round_index == INITIAL_ROUND and mix in MIXES) or (
            name in names
            and (
                (round_index == INITIAL_ROUND and mix in MIXES)
                or (round_index == EXTENSION_ROUND and mix == "scripted")
            )
        )
        if not valid:
            raise ValueError("T6 rejects unregistered probe arms, rounds, or mixes")
        for field in ("mass", "deaths", "alive_frames", "length_at_5000"):
            raw = row.get(field)
            if not isinstance(raw, Mapping) or any(
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                for value in raw.values()
            ):
                raise ValueError("T6 metrics must be finite numeric maps")
        maps = [
            _number_map(row, key) for key in ("mass", "deaths", "alive_frames", "length_at_5000")
        ]
        if (
            row.get("frames") != FRAMES
            or not maps[0]
            or any(set(item) != set(maps[0]) for item in maps[1:])
        ):
            raise ValueError("T6 requires complete H5000 metrics including length_at_5000")
        if len(maps[0]) != (96 if mix == "scripted" else 32):
            raise ValueError("T6 probe row has wrong registered world count")


def analyze(
    rows: Sequence[Mapping[str, Any]],
    statuses: Mapping[str, int],
    telemetry: Mapping[str, Any] | None = None,
    q_gaps: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply the preregistered decision rule; readouts cannot alter it."""
    grouped = _group(rows)
    _validate(grouped, statuses)
    extended = any(round_index == EXTENSION_ROUND for _, round_index, _ in grouped)
    initial = {key: value for key, value in grouped.items() if key[1] == INITIAL_ROUND}
    initial_data = {
        "vs_ctrl": {
            mix: {
                metric: _paired(initial, mix, metric)
                for metric in ("episode_survival", "alive_survival", "mass")
            }
            for mix in MIXES
        }
    }
    initial_decision = _decision(initial_data, statuses, False)
    if extended:
        required = {
            (f"{name}_s{seed}", EXTENSION_ROUND, "scripted")
            for name in (ARM, "CTRL")
            for seed in SEEDS
        }
        if initial_decision != "EXTEND" or not required.issubset(grouped):
            raise ValueError("T6 extension requires initial EXTEND and all ten round-73 rows")
    data: dict[str, Any] = {
        "failed_runs": [
            {"name": name, "exit": code} for name, code in sorted(statuses.items()) if code != 0
        ],
        "unstarted_runs": [f"{ARM}_s{seed}" for seed in SEEDS if f"{ARM}_s{seed}" not in statuses],
        "vs_ctrl": {
            mix: {
                metric: _paired(grouped, mix, metric)
                for metric in ("episode_survival", "alive_survival", "mass")
            }
            for mix in MIXES
        },
        "mass_vs_incumbent": {mix: _incumbent(grouped, mix) for mix in MIXES},
        "shared_survivor_length_h5000": _shared_lengths(grouped),
    }
    data["decision"] = _decision(data, statuses, extended)
    data["initial_decision"], data["extension_started"], data["extension_exhausted"] = (
        initial_decision,
        extended,
        extended,
    )
    if telemetry is not None:
        data["telemetry"] = telemetry
    if q_gaps is not None:
        data["heldout_decisive_q_gaps"] = q_gaps
    return _json_safe({"schema": "t6-screen-analysis/v1", "arms": {ARM: data}})


def extension_requirements(
    result: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Return the single registered 96-world scripted extension, or no work."""
    if result["arms"][ARM]["decision"] != "EXTEND" or any(
        int(row["round_index"]) == EXTENSION_ROUND for row in rows
    ):
        return []
    return [
        {
            "name": f"{name}_s{seed}",
            "round_index": EXTENSION_ROUND,
            "mix": "scripted",
            "frames": FRAMES,
            "worlds": 96,
        }
        for seed in SEEDS
        for name in (ARM, "CTRL")
    ]


def read_training_telemetry(roots: Mapping[str, Path]) -> dict[str, Any]:
    """Read registered restart/fresh death counts and every 100k boost point."""
    per_seed: dict[str, Any] = {}
    curves: list[dict[str, Any]] = []
    for seed in SEEDS:
        root = Path(roots[f"{ARM}_s{seed}"])
        summary = json.loads((root / "summary.json").read_text())
        totals = summary.get("restart_totals", {})
        if not isinstance(totals, Mapping):
            raise ValueError("T6 summary lacks restart_totals")
        sides = {}
        for side in ("restart", "fresh"):
            decisions, deaths = totals.get(f"{side}_decisions_ge800"), totals.get(
                f"{side}_deaths_ge800"
            )
            if (
                not isinstance(decisions, int)
                or not isinstance(deaths, int)
                or decisions < 0
                or deaths < 0
            ):
                raise ValueError("T6 restart telemetry must contain nonnegative integer counts")
            sides[side] = {
                "decisions_ge800": decisions,
                "deaths_ge800": deaths,
                "deaths_per_1k_ge800": 1000.0 * deaths / decisions if decisions else None,
                "decisions": totals.get(f"{side}_decisions"),
                "deaths": totals.get(f"{side}_deaths"),
                "mean_decision_length": totals.get(f"{side}_length_mean"),
                "mean_predeath_length": totals.get(f"{side}_death_length_mean"),
            }
        records = [
            json.loads(line)
            for line in (root / "log.jsonl").read_text().splitlines()
            if line.strip()
        ]
        per_seed[str(seed)] = {"sides": sides, "summary": str(root / "summary.json")}
        for point in records:
            if isinstance(point.get("transitions"), (int, float)):
                curves.append(
                    {
                        "seed": seed,
                        "transitions": point["transitions"],
                        "greedy_boost_rate": point.get("greedy_boost_rate"),
                        "greedy_boost_rate_lt_700": point.get("greedy_boost_rate_lt_700"),
                        "greedy_boost_rate_by_length": point.get("greedy_boost_rate_by_length"),
                        "greedy_boost_length_decisions": point.get("greedy_boost_length_decisions"),
                    }
                )
    pooled = {}
    for side in ("restart", "fresh"):
        denominator = sum(per_seed[str(seed)]["sides"][side]["decisions_ge800"] for seed in SEEDS)
        numerator = sum(per_seed[str(seed)]["sides"][side]["deaths_ge800"] for seed in SEEDS)
        all_decisions = sum(per_seed[str(seed)]["sides"][side]["decisions"] for seed in SEEDS)
        all_deaths = sum(per_seed[str(seed)]["sides"][side]["deaths"] for seed in SEEDS)
        decision_length_numerator = sum(
            per_seed[str(seed)]["sides"][side]["decisions"]
            * per_seed[str(seed)]["sides"][side]["mean_decision_length"]
            for seed in SEEDS
        )
        death_length_numerator = sum(
            per_seed[str(seed)]["sides"][side]["deaths"]
            * per_seed[str(seed)]["sides"][side]["mean_predeath_length"]
            for seed in SEEDS
        )
        pooled[side] = {
            "decisions_ge800": denominator,
            "deaths_ge800": numerator,
            "deaths_per_1k_ge800": 1000.0 * numerator / denominator if denominator else None,
            "decisions": all_decisions,
            "deaths": all_deaths,
            "mean_decision_length": (
                decision_length_numerator / all_decisions if all_decisions else None
            ),
            "mean_predeath_length": death_length_numerator / all_deaths if all_deaths else None,
        }
    curve_summary: list[dict[str, Any]] = []
    nominal_points = {int(round(float(row["transitions"]) / 100000.0)) * 100000 for row in curves}
    for nominal in sorted(nominal_points):
        points = [
            row
            for row in curves
            if int(round(float(row["transitions"]) / 100000.0)) * 100000 == nominal
        ]
        rates = [
            float(row["greedy_boost_rate"])
            for row in points
            if isinstance(row.get("greedy_boost_rate"), Real)
        ]
        low = [
            float(row["greedy_boost_rate_lt_700"])
            for row in points
            if isinstance(row.get("greedy_boost_rate_lt_700"), Real)
        ]
        strata = {}
        for bucket in ("lt_400", "400_700", "700_1000", "ge_1000"):
            values = [
                float(row["greedy_boost_rate_by_length"][bucket])
                for row in points
                if isinstance(row.get("greedy_boost_rate_by_length"), Mapping)
                and isinstance(row["greedy_boost_rate_by_length"].get(bucket), Real)
            ]
            strata[bucket] = _mean(values) if values else None
        curve_summary.append(
            {
                "transitions": nominal,
                "n": len(points),
                "greedy_boost_rate_mean": _mean(rates) if rates else None,
                "greedy_boost_rate_lt_700_mean": _mean(low) if low else None,
                "by_length_mean": strata,
            }
        )
    return {
        "schema": "t6-training-telemetry/v1",
        "per_seed": per_seed,
        "pooled": pooled,
        "boost_curve": curves,
        "boost_curve_summary": curve_summary,
        "tripwires": {"greedy_boost_rate": 0.02, "greedy_boost_rate_lt_700": 0.01},
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _decisive_gaps(
    q: Any,
    actions: Any,
    died: Any,
    groups: Mapping[tuple[int, int], Sequence[int]],
    b8_actions: Mapping[bytes, int],
    state_keys: Sequence[bytes],
) -> Any:
    """Return one fatal-versus-best-survivor gap per decisive source state."""
    import numpy as np

    gaps = []
    for q_index, (_, indices) in enumerate(sorted(groups.items())):
        index = indices[0]
        b8_action = b8_actions.get(state_keys[index])
        survivors = [int(actions[item]) for item in indices if not bool(died[item])]
        fatal = [item for item in indices if int(actions[item]) == b8_action and bool(died[item])]
        if b8_action is None or len(fatal) != 1 or not survivors:
            raise ValueError("each decisive holdout state needs one B8-fatal action and a survivor")
        gaps.append(float(q[q_index, b8_action] - np.max(q[q_index, survivors])))
    return np.asarray(gaps, dtype=float)


def _phase_a_decisive_spools(
    store: Path, bank_root: Path, shard_hashes: Mapping[str, str]
) -> tuple[dict[str, str], dict[str, str]]:
    """Bind Phase-A spools and their published shards to the holdout manifest."""
    phase_root = store / "staging/phase-a"
    result: dict[str, str] = {}
    holdout_receipts: dict[str, str] = {}
    for receipt in sorted(phase_root.glob("*.json")):
        value = json.loads(receipt.read_text())
        relative, digest = value.get("decisive_spool"), value.get("decisive_spool_sha256")
        if value.get("schema") != "t4e-phase-a-world/v1" or not isinstance(relative, str):
            raise ValueError("malformed Phase-A decisive spool receipt")
        if not relative.startswith("decisive/") or not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("Phase-A receipt lacks a bound decisive spool hash")
        path = store / "staging" / relative
        if not path.is_file() or _sha256(path) != digest:
            raise ValueError("Phase-A decisive spool hash mismatch")
        previous = result.setdefault(relative, digest)
        if previous != digest:
            raise ValueError("conflicting Phase-A decisive spool hashes")
        published = value.get("published", {})
        shard = published.get("shard") if isinstance(published, Mapping) else None
        if shard in shard_hashes:
            if published.get("sha256") != shard_hashes[shard]:
                raise ValueError("Phase-A published shard does not match holdout manifest")
            receipt = bank_root / "worlds" / (Path(shard).stem + ".receipt.json")
            if not receipt.is_file() or _sha256(receipt) != json.loads(
                (bank_root / "manifest.json").read_text()
            )["publication"]["world_receipts"].get(receipt.name):
                raise ValueError("holdout world receipt is not bound by the manifest")
            receipt_value = json.loads(receipt.read_text())
            if receipt_value.get("sha256") != shard_hashes[shard]:
                raise ValueError("holdout world receipt does not bind Phase-A shard")
            holdout_receipts[receipt.name] = _sha256(receipt)
    if not result:
        raise FileNotFoundError("Phase-A decisive spool receipts are required")
    found = {"decisive/" + path.name for path in (store / "staging/decisive").glob("*.npz")}
    if found != set(result):
        raise ValueError("Phase-A receipt/spool inventory mismatch")
    if set(shard_hashes) != {
        json.loads((bank_root / "worlds" / name).read_text())["shard"] for name in holdout_receipts
    }:
        raise ValueError("not every immutable holdout shard has a Phase-A receipt")
    return result, holdout_receipts


def heldout_decisive_q_gaps(
    roots: Mapping[str, Path], bank_root: Path = HOLDOUT_ROOT
) -> dict[str, Any]:
    """Measure the immutable T4e holdout gap on CPU, with no fuse classifier.

    The branch shards provide legal surviving actions.  The original B8 fatal
    action is recovered from the collection spool by exact packed-observation
    bytes; it is never inferred from a model evaluated in this pass.
    """
    import numpy as np
    import torch

    from research.redesign_m3c_20261008.obs_pack import train_layout, unpack_ref
    from src.model.ego2s_network import load_ego2s_checkpoint
    from src.simd_env.ego2s_policy import forward_q

    bank_root = Path(bank_root)
    manifest_path = bank_root / "manifest.json"
    if _sha256(manifest_path) != HOLDOUT_MANIFEST_SHA256:
        raise ValueError("immutable T4e holdout manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    store = bank_root.parents[1]
    spool_hashes, holdout_receipts = _phase_a_decisive_spools(store, bank_root, manifest["shards"])
    pieces: dict[str, list[Any]] = {}
    for name, digest in manifest["shards"].items():
        shard = bank_root / name
        if _sha256(shard) != digest:
            raise ValueError("immutable T4e holdout shard hash mismatch")
        with np.load(shard, allow_pickle=False) as archive:
            for key in archive.files:
                if key != "manifest_json":
                    pieces.setdefault(key, []).append(archive[key])
    rows = {key: np.concatenate(value) for key, value in pieces.items()}
    if len(rows.get("action", ())) != 5418:
        raise ValueError("T4e holdout must contain exactly 5,418 branches")

    def key_for(packed: Any, scalars: Any) -> bytes:
        return (
            np.asarray(packed, dtype=np.uint8).tobytes()
            + np.asarray(scalars, dtype=np.float32).tobytes()
        )

    b8_action: dict[bytes, int] = {}
    for relative, digest in sorted(spool_hashes.items()):
        spool = store / "staging" / relative
        if _sha256(spool) != digest:
            raise ValueError("Phase-A decisive spool hash mismatch")
        with np.load(spool, allow_pickle=False) as archive:
            for index in range(len(archive["b8_action"])):
                state = key_for(archive["state_packed"][index], archive["state_scalars"][index])
                action = int(archive["b8_action"][index])
                previous = b8_action.setdefault(state, action)
                if previous != action:
                    raise ValueError("conflicting B8 actions for one immutable holdout observation")
    if not b8_action:
        raise FileNotFoundError("T4e decisive spool is required to recover B8 actions")
    state_keys = [
        key_for(packed, scalars)
        for packed, scalars in zip(rows["state_packed"], rows["state_scalars"])
    ]
    groups: dict[tuple[int, int], list[int]] = {}
    for index, (seed, decision) in enumerate(zip(rows["source_seed"], rows["source_decision"])):
        groups.setdefault((int(seed), int(decision)), []).append(index)
    take = [indices[0] for _, indices in sorted(groups.items())]

    def q_values(checkpoint: Path) -> Any:
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        net = load_ego2s_checkpoint(str(checkpoint), "cpu").eval()
        result = []
        for start in range(0, len(take), 256):
            indices = take[start : start + 256]
            local, global_ = unpack_ref(np.asarray(rows["state_packed"])[indices], train_layout())
            obs = {
                "local": local,
                "global": global_,
                "scalars": np.asarray(rows["state_scalars"])[indices].astype(np.float32),
            }
            result.append(forward_q(net, obs, "rowwise").detach().cpu().numpy())
        return np.concatenate(result).astype(np.float64)

    per_seed: dict[str, Any] = {}
    paired = []
    for seed in SEEDS:
        restart = Path(roots[f"{ARM}_s{seed}"]) / "final.pth"
        b8 = B8_ROOT / f"B8_s{seed - 10}" / "final.pth"
        if not restart.is_file() or not b8.is_file():
            raise FileNotFoundError("T6 and corresponding B8 final checkpoints are required")
        values = {}
        for label, path in (("B8", b8), (ARM, restart)):
            q = q_values(path)
            gap = _decisive_gaps(q, rows["action"], rows["died"], groups, b8_action, state_keys)
            values[label] = {
                "n": int(len(gap)),
                "median": float(np.median(gap)),
                "mean": float(gap.mean()),
                "p10": float(np.percentile(gap, 10)),
                "p90": float(np.percentile(gap, 90)),
                "values": gap.tolist(),
            }
        change = np.asarray(values[ARM]["values"]) - np.asarray(values["B8"]["values"])
        values["paired_change"] = {
            "n": int(len(change)),
            "median": float(np.median(change)),
            "mean": float(change.mean()),
            "p10": float(np.percentile(change, 10)),
            "p90": float(np.percentile(change, 90)),
            "values": change.tolist(),
        }
        paired.append(values["paired_change"]["mean"])
        per_seed[str(seed)] = values
    if len(take) != 1052:
        raise ValueError("T4e holdout must contain exactly 1,052 decisive states")
    return _json_safe(
        {
            "schema": "t6-heldout-decisive-q-gap/v1",
            "holdout_manifest_sha256": HOLDOUT_MANIFEST_SHA256,
            "phase_a_decisive_spool_sha256": spool_hashes,
            "holdout_phase_a_receipt_sha256": holdout_receipts,
            "states": len(take),
            "branches": int(len(rows["action"])),
            "per_seed": per_seed,
            "five_seed_paired_change": {
                "mean": _mean(paired),
                "seed_se": _se(paired),
                "n": len(paired),
            },
            "summary": (
                "Fuse-free CPU rowwise Q(original B8 fatal action) minus max Q(recorded "
                "surviving legal actions), with paired RESTART-minus-B8 change."
            ),
        }
    )


def _json_safe(value: Any) -> Any:
    """Encode undefined descriptive statistics as JSON null, never NaN."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _show(value: Any, digits: int = 4) -> str:
    return (
        "n/a"
        if not isinstance(value, (int, float)) or not math.isfinite(value)
        else f"{value:.{digits}f}"
    )


def markdown(result: Mapping[str, Any]) -> str:
    """Render the registered result, preserving diagnostic/decision separation."""
    data = result["arms"][ARM]
    lines = [
        "# T6 RESTART screen result (2026-10-17)",
        "",
        f"Decision: {data['decision']}",
        "",
        "## Registered gameplay result",
        "",
        "| mix | survival delta vs CTRL (seed SE) | mass delta vs CTRL (seed SE) | "
        "mass vs incumbent (seed SE) |",
        "|---|---:|---:|---:|",
    ]
    for mix in MIXES:
        survival, mass, incumbent = (
            data["vs_ctrl"][mix]["episode_survival"],
            data["vs_ctrl"][mix]["mass"],
            data["mass_vs_incumbent"][mix],
        )
        lines.append(
            "| %s | %s (%s) | %s (%s) | %s (%s) |"
            % (
                mix,
                _show(survival["mean"]),
                _show(survival["seed_se"]),
                _show(mass["mean"]),
                _show(mass["seed_se"]),
                _show(incumbent["mean"]),
                _show(incumbent["seed_se"]),
            )
        )
    primary = data["vs_ctrl"]["scripted"]["episode_survival"]
    lines += [
        "",
        "Primary five-seed scripted delta / SE: %s / %s."
        % (_show(primary["mean"]), _show(primary["seed_se"])),
        "",
        "## Registered descriptive readouts",
        "",
    ]
    for mix, value in data["shared_survivor_length_h5000"]["by_mix"].items():
        lines.append(
            "Shared-survivor H5000 length (%s): five-seed delta %s; pooled delta %s; "
            "denominator %s."
            % (
                mix,
                _show(value["five_seed_mean_delta"]),
                _show(value["pooled_mean_delta"]),
                value["shared_survivors"],
            )
        )
    telemetry = data.get("telemetry")
    if telemetry:
        lines += ["", "### Restart/fresh L>=800 deaths", ""]
        for side, value in telemetry.get("pooled", {}).items():
            lines.append(
                "%s: L>=800 %s/%s = %s deaths per 1,000; all lengths %s/%s; "
                "mean decision/pre-death length %s / %s."
                % (
                    side,
                    value["deaths_ge800"],
                    value["decisions_ge800"],
                    _show(value["deaths_per_1k_ge800"]),
                    value["deaths"],
                    value["decisions"],
                    _show(value["mean_decision_length"]),
                    _show(value["mean_predeath_length"]),
                )
            )
        lines += ["", "Per-seed L>=800 deaths (restart / fresh):", ""]
        for seed, point in telemetry.get("per_seed", {}).items():
            restart, fresh = point["sides"]["restart"], point["sides"]["fresh"]
            lines.append(
                "- %s: %s/%s (%s) / %s/%s (%s)."
                % (
                    seed,
                    restart["deaths_ge800"],
                    restart["decisions_ge800"],
                    _show(restart["deaths_per_1k_ge800"]),
                    fresh["deaths_ge800"],
                    fresh["decisions_ge800"],
                    _show(fresh["deaths_per_1k_ge800"]),
                )
            )
        lines.append(
            "Boost curve rows: %s; per-100k aggregate rows: %s; tripwires: %s."
            % (
                len(telemetry.get("boost_curve", [])),
                len(telemetry.get("boost_curve_summary", [])),
                telemetry.get("tripwires", {}),
            )
        )
    gap = data.get("heldout_decisive_q_gaps")
    if gap:
        lines += [
            "",
            "### Immutable T4e holdout decisive Q gap",
            "",
            str(gap.get("summary", gap)),
            "",
        ]
        lines += [
            "| seed | B8 median | RESTART median | paired mean change |",
            "|---|---:|---:|---:|",
        ]
        for seed, point in gap.get("per_seed", {}).items():
            lines.append(
                "| %s | %s | %s | %s |"
                % (
                    seed,
                    _show(point["B8"]["median"]),
                    _show(point[ARM]["median"]),
                    _show(point["paired_change"]["mean"]),
                )
            )
        paired = gap.get("five_seed_paired_change", {})
        lines += [
            "",
            "Five-seed paired change mean / SE: %s / %s."
            % (_show(paired.get("mean")), _show(paired.get("seed_se"))),
        ]
    return "\n".join(lines) + "\n"
