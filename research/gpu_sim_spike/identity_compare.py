"""Pre-registered comparison and reducer for the closed-loop identity falsifier.

Tier L (decides the outcome; JUDGMENT_POST_I0S_DIRECTION step 3): at every seam
``t`` the device and the independently regenerated CPU tape must agree exactly,
in this order, on

1. replica self-consistency of the device batch (every replica == replica 0);
2. the mapped seam state: every ``SNAKE_FIELDS`` value (head cell masked for
   dead snakes), the live logical body digest of every alive snake, ``frame``,
   and the ordered food list with corpse flags;
3. the hero's raw advisory mask;
4. the hero action;
5. each opponent action, for opponents alive at the seam, slot ascending.

Any Tier-L difference, including an exact float tie broken differently, is a
divergence and makes the outcome B.  Outcome A requires all 12 worlds to agree
through the end of both tapes (H1000 or the same hero death).  Dead opponents'
actions are inert and are recorded but not compared.

Tier D (recorded, never decides): hero Q bytes, observation digests, opponent
Q bytes, masks and vector61 state digests, and ``_last_reward`` (CPU float64
versus device float32, the inherited REWARD_PRECISION class).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from research.gpu_sim_spike import identity_tape as tape

SCHEMA = "gpu-identity-report/v1"
CLASSES = {
    "START_STATE": "start",
    "DEVICE_NONDETERMINISM": "device",
    "RNG": "rng",
    "EVENT_ORDER": "event_order",
    "MECHANICS": "mechanics",
    "FEATURE": "feature",
    "FALLBACK_RULE": "feature",
    "SCRIPTED_ANCHOR": "feature",
    "V8": "feature",
    "SELECTION": "feature",
    "FLOAT_TIE": "float_tie",
    "FLOAT_NUMERIC": "float_numeric",
}
# Explained classes: documented counter RNG, inherited ordered-food class, or
# CPU/GPU float arithmetic.  Every other class is an unexplained device defect.
EXPLAINED = {"RNG", "EVENT_ORDER", "FLOAT_TIE", "FLOAT_NUMERIC"}
_ALIVE = tape.SNAKE_FIELDS.index("alive")
_HEAD = (tape.SNAKE_FIELDS.index("head_x"), tape.SNAKE_FIELDS.index("head_y"))


def _food(arrays: Mapping[str, np.ndarray], t: int) -> np.ndarray:
    count = int(arrays["state_food_count"][t])
    return np.asarray(arrays["state_food"][t, :count], dtype=np.int64)


def state_differences(
    dev: Mapping[str, np.ndarray], cpu: Mapping[str, np.ndarray], t: int
) -> list[str]:
    """Names of every differing Tier-L state element at seam ``t``."""
    out: list[str] = []
    a = np.asarray(dev["state_snake"][t], dtype=np.int64)
    b = np.asarray(cpu["state_snake"][t], dtype=np.int64)
    for slot, field in zip(*np.nonzero(a != b)):
        out.append(f"snake.{tape.SNAKE_FIELDS[int(field)]}[{int(slot)}]")
    for slot in np.flatnonzero((dev["state_body"][t] != cpu["state_body"][t]).any(axis=1)):
        out.append(f"body[{int(slot)}]")
    if int(dev["state_frame"][t]) != int(cpu["state_frame"][t]):
        out.append("frame")
    if not np.array_equal(_food(dev, t), _food(cpu, t)):
        out.append("food")
    return out


def _rng_food(prior: np.ndarray, dev: np.ndarray, cpu: np.ndarray) -> bool:
    """Lists agree up to a tail made only of fresh ambient (non-corpse) cells."""
    old = {(int(x), int(y)) for x, y, _ in prior}
    n = 0
    while n < min(len(dev), len(cpu)) and np.array_equal(dev[n], cpu[n]):
        n += 1
    for tail in (dev[n:], cpu[n:]):
        if any(int(c) != 0 or (int(x), int(y)) in old for x, y, c in tail):
            return False
    return True


def classify_state(
    dev: Mapping[str, np.ndarray], cpu: Mapping[str, np.ndarray], t: int, diffs: list[str]
) -> str:
    """Classify a first state divergence at seam ``t`` (seam ``t - 1`` was equal)."""
    if t == 0:
        return "START_STATE"
    respawned = {
        slot
        for slot in range(dev["state_snake"].shape[1])
        if dev["state_snake"][t - 1, slot, _ALIVE] == 0
        and dev["state_snake"][t, slot, _ALIVE] == 1
        and cpu["state_snake"][t, slot, _ALIVE] == 1
    }
    allowed = {f"snake.head_x[{s}]" for s in respawned}
    allowed |= {f"snake.head_y[{s}]" for s in respawned} | {f"body[{s}]" for s in respawned}
    other = [name for name in diffs if name not in allowed and name != "food"]
    food_ok = "food" not in diffs or _rng_food(_food(dev, t - 1), _food(dev, t), _food(cpu, t))
    if not other and food_ok:
        return "RNG"
    if not other and sorted(map(tuple, _food(dev, t))) == sorted(map(tuple, _food(cpu, t))):
        return "EVENT_ORDER"
    return "MECHANICS"


def _selectable(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool).copy()
    if not mask.any():
        mask[:3] = True
    return mask


def _tie(q: np.ndarray, mask: np.ndarray) -> bool:
    values = np.asarray(q, dtype=np.float32)[_selectable(mask)]
    return values.size > 1 and int((values == values.max()).sum()) > 1


def _margin(q: np.ndarray, mask: np.ndarray) -> float | None:
    values = np.sort(np.asarray(q, dtype=np.float64)[_selectable(mask)])[::-1]
    return float(values[0] - values[1]) if values.size > 1 else None


def _bits_equal(a: np.ndarray, b: np.ndarray) -> bool:
    a32, b32 = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    return a32.shape == b32.shape and np.array_equal(a32.view(np.int32), b32.view(np.int32))


def _masked_argmax(q: np.ndarray, mask: np.ndarray) -> int:
    return int(np.where(_selectable(mask), np.asarray(q, dtype=np.float32), -np.inf).argmax())


def decision_difference(
    dev: Mapping[str, np.ndarray], cpu: Mapping[str, np.ndarray], t: int, kinds: list[str]
) -> dict[str, Any] | None:
    """First Tier-L decision difference at decision ``t`` (state ``t`` was equal)."""
    if not np.array_equal(dev["hero_mask"][t], cpu["hero_mask"][t]):
        return {"check": "hero_mask", "class": "FEATURE", "slot": 0}
    if int(dev["actions"][t, 0]) != int(cpu["actions"][t, 0]):
        q_dev, q_cpu = dev["hero_q"][t], cpu["hero_q"][t]
        mask = dev["hero_mask"][t]
        if not np.array_equal(dev["obs_digest"][t], cpu["obs_digest"][t]):
            label = "FEATURE"
        elif _bits_equal(q_dev, q_cpu):
            label = "SELECTION"
        elif _tie(q_dev, mask) or _tie(q_cpu, mask):
            label = "FLOAT_TIE"
        else:
            label = "FLOAT_NUMERIC"
        return {"check": "hero_action", "class": label, "slot": 0}
    alive = np.asarray(dev["state_snake"][t, :, _ALIVE], dtype=bool)
    for slot in range(1, dev["actions"].shape[1]):
        if not alive[slot] or int(dev["actions"][t, slot]) == int(cpu["actions"][t, slot]):
            continue
        if kinds[slot] != "network":
            return {"check": "opponent_action", "class": "SCRIPTED_ANCHOR", "slot": slot}
        recorded = bool(dev["opp_recorded"][t, slot]) and bool(cpu["opp_recorded"][t, slot])
        if not recorded or not (
            np.array_equal(dev["opp_mask"][t, slot], cpu["opp_mask"][t, slot])
            and np.array_equal(dev["opp_state_digest"][t, slot], cpu["opp_state_digest"][t, slot])
        ):
            label = "FEATURE"
        elif _bits_equal(dev["opp_q"][t, slot], cpu["opp_q"][t, slot]):
            label = "V8"
        else:
            mask = dev["opp_mask"][t, slot]
            q_dev, q_cpu = dev["opp_q"][t, slot], cpu["opp_q"][t, slot]
            if _tie(q_dev, mask) or _tie(q_cpu, mask):
                label = "FLOAT_TIE"
            else:
                label = "FLOAT_NUMERIC"
        return {"check": "opponent_action", "class": label, "slot": slot}
    return None


def seam_divergence(
    dev: Mapping[str, np.ndarray], cpu: Mapping[str, np.ndarray], t: int, kinds: list[str]
) -> dict[str, Any] | None:
    """Tier-L checks for seam ``t`` and, when both tapes have it, decision ``t``."""
    if not bool(dev["replica_state_agree"][t]):
        return {"tick": t, "check": "replicas", "class": "DEVICE_NONDETERMINISM"}
    diffs = state_differences(dev, cpu, t)
    if diffs:
        return {
            "tick": t,
            "check": "state",
            "class": classify_state(dev, cpu, t, diffs),
            "fields": diffs,
        }
    if t < len(dev["actions"]) and t < len(cpu["actions"]):
        if not bool(dev["replica_decision_agree"][t]):
            return {"tick": t, "check": "replicas", "class": "DEVICE_NONDETERMINISM"}
        found = decision_difference(dev, cpu, t, kinds)
        if found is not None:
            return {"tick": t, **found}
    return None


def opponent_kinds(row: Mapping[str, Any]) -> list[str]:
    return ["hero"] + [
        "network" if kind == "checkpoint" else "scripted" for kind, _ in row["opponents"]
    ]


def evidence(
    dev: Mapping[str, np.ndarray], cpu: Mapping[str, np.ndarray], t: int
) -> dict[str, Any]:
    """Q margins, masks and digests at a first divergence (both sides)."""
    out: dict[str, Any] = {"tick": t}
    if t < len(dev["actions"]) and t < len(cpu["actions"]):
        for side, arrays in (("device", dev), ("cpu", cpu)):
            out[side] = {
                "actions": arrays["actions"][t].tolist(),
                "hero_q_hex": [float(x).hex() for x in np.asarray(arrays["hero_q"][t], np.float32)],
                "hero_mask": np.asarray(arrays["hero_mask"][t], bool).tolist(),
                "hero_margin": _margin(arrays["hero_q"][t], arrays["hero_mask"][t]),
                "hero_tie": _tie(arrays["hero_q"][t], arrays["hero_mask"][t]),
                "opp_q_hex": [
                    [float(x).hex() for x in row] if recorded else None
                    for row, recorded in zip(
                        np.asarray(arrays["opp_q"][t], np.float32), arrays["opp_recorded"][t]
                    )
                ],
                "opp_mask": np.asarray(arrays["opp_mask"][t], bool).tolist(),
                "obs_digest": [bytes(row).hex() for row in arrays["obs_digest"][t]],
            }
    if t > 0:
        out["prior_state_equal"] = not state_differences(dev, cpu, t - 1)
    return out


def tier_d(
    dev: Mapping[str, np.ndarray], cpu: Mapping[str, np.ndarray], upto: int
) -> dict[str, Any]:
    """Recorded-only numeric diagnostics over decisions/seams ``[0, upto)``."""
    steps = min(upto, len(dev["actions"]), len(cpu["actions"]))
    q_diff = [t for t in range(steps) if not _bits_equal(dev["hero_q"][t], cpu["hero_q"][t])]
    obs_diff = [
        t for t in range(steps) if not np.array_equal(dev["obs_digest"][t], cpu["obs_digest"][t])
    ]
    max_abs = max(
        (
            float(
                np.max(
                    np.abs(
                        dev["hero_q"][t].astype(np.float64) - cpu["hero_q"][t].astype(np.float64)
                    )
                )
            )
            for t in range(steps)
        ),
        default=0.0,
    )
    opp_q, opp_mask, opp_state = 0, 0, 0
    for t in range(steps):
        both = np.asarray(dev["opp_recorded"][t], bool) & np.asarray(cpu["opp_recorded"][t], bool)
        for slot in np.flatnonzero(both):
            opp_q += int(not _bits_equal(dev["opp_q"][t, slot], cpu["opp_q"][t, slot]))
            opp_mask += int(not np.array_equal(dev["opp_mask"][t, slot], cpu["opp_mask"][t, slot]))
            opp_state += int(
                not np.array_equal(
                    dev["opp_state_digest"][t, slot], cpu["opp_state_digest"][t, slot]
                )
            )
    seams = min(upto + 1, len(dev["state_reward"]), len(cpu["state_reward"]))
    reward_diff = [
        t
        for t in range(seams)
        if not np.array_equal(
            np.asarray(dev["state_reward"][t]).astype(np.float64),
            np.asarray(cpu["state_reward"][t]).astype(np.float64),
        )
    ]
    return {
        "decisions_compared": steps,
        "hero_q_bit_differences": len(q_diff),
        "first_hero_q_bit_difference": q_diff[0] if q_diff else None,
        "hero_q_max_abs_difference": max_abs,
        "observation_digest_differences": len(obs_diff),
        "opponent_q_bit_differences": opp_q,
        "opponent_mask_differences": opp_mask,
        "opponent_state_digest_differences": opp_state,
        "reward_value_differences": len(reward_diff),
        "first_reward_value_difference": reward_diff[0] if reward_diff else None,
        "reward_dtypes": {
            "device": sorted(set(np.asarray(dev["state_reward_dtype"]).tolist())),
            "cpu": sorted(set(np.asarray(cpu["state_reward_dtype"]).tolist())),
        },
    }


def compare_world(
    dev: Mapping[str, np.ndarray],
    cpu: Mapping[str, np.ndarray],
    row: Mapping[str, Any],
    horizon: int,
) -> dict[str, Any]:
    """First Tier-L divergence, or identity through the end of both tapes."""
    kinds = opponent_kinds(row)
    seams = min(len(dev["state_frame"]), len(cpu["state_frame"]))
    for t in range(seams):
        found = seam_divergence(dev, cpu, t, kinds)
        if found is not None:
            label = found["class"]
            return {
                "status": "INVALID" if label == "START_STATE" else "DIVERGED",
                "first_divergence": {
                    **found,
                    "ruling_class": CLASSES[label],
                    "device_correctness_defect": label not in EXPLAINED,
                },
                "evidence": evidence(dev, cpu, int(found["tick"])),
                "tier_d": tier_d(dev, cpu, int(found["tick"])),
            }
    dev_steps, cpu_steps = len(dev["actions"]), len(cpu["actions"])
    hero_dead = not bool(dev["state_snake"][-1, 0, _ALIVE])
    complete = (
        dev_steps == cpu_steps
        and len(dev["state_frame"]) == len(cpu["state_frame"])
        and (dev_steps == horizon or hero_dead)
    )
    return {
        "status": "IDENTICAL" if complete else "INCOMPLETE",
        "first_divergence": None,
        "decisions": {"device": dev_steps, "cpu": cpu_steps},
        "tier_d": tier_d(dev, cpu, min(dev_steps, cpu_steps)),
    }


def verdict(worlds: Mapping[str, Mapping[str, Any]], allocation: tuple[str, ...]) -> str:
    """INVALID > B (any divergence) > INCOMPLETE > A (all identical)."""
    statuses = [worlds.get(unit, {}).get("status", "MISSING") for unit in allocation]
    if "INVALID" in statuses:
        return "INVALID"
    if "DIVERGED" in statuses:
        return "OUTCOME_B_IDENTITY_FAILS"
    if all(status == "IDENTICAL" for status in statuses):
        return "OUTCOME_A_IDENTICAL"
    return "INCOMPLETE"


def reduce(
    device_root: Path,
    cpu_root: Path,
    *,
    mode: str,
    commit: str,
    preregistration: Path | None = None,
) -> dict[str, Any]:
    """Verify every device and CPU tape, then compare (one registered look)."""
    from research.gpu_sim_spike import identity_cpu

    rows = tape.world_rows(mode)
    allocation = tape.allocation(mode)
    ns = tape.namespace(mode, commit)
    horizon = int(tape.MODES[mode]["horizon"])
    worlds: dict[str, Any] = {}
    bridged: dict[str, Any] = {}
    for index, unit in enumerate(allocation):
        row = rows[index]
        try:
            record, dev, dev_meta = tape.load_unit(device_root, ns, allocation, unit)
        except FileNotFoundError:
            worlds[unit] = {"status": "MISSING", "reason": "device tape absent"}
            continue
        except ValueError as exc:
            worlds[unit] = {"status": "INVALID", "reason": f"device tape: {exc}"}
            continue
        if not dev_meta.get("start_matches_cpu_constructor") or (
            dev_meta.get("device") == "cuda"
            and not (
                dev_meta["native"]["move_extension_loaded"]
                and dev_meta["native"]["geometry_extension_loaded"]
            )
        ):
            worlds[unit] = {"status": "INVALID", "reason": "device start or native route"}
            continue
        try:
            cpu, cpu_meta = identity_cpu.load_cpu_tape(cpu_root, ns, unit, "natural", record)
        except FileNotFoundError:
            worlds[unit] = {"status": "MISSING", "reason": "CPU regeneration absent"}
            continue
        except ValueError as exc:
            worlds[unit] = {"status": "INVALID", "reason": f"CPU tape: {exc}"}
            continue
        result = compare_world(dev, cpu, row, horizon)
        if result["status"] == "INCOMPLETE" and cpu_meta.get("stopped") == "first_divergence":
            result = {
                "status": "INVALID",
                "reason": "CPU stopped without a reproducible divergence",
            }
        result.update(
            mix=row["mix"],
            world_id=row["world_id"],
            device_terminal=dev_meta["terminal"],
            cpu_terminal=cpu_meta.get("terminal"),
        )
        worlds[unit] = result
        try:
            bridged_tape, bridged_meta = identity_cpu.load_cpu_tape(
                cpu_root, ns, unit, "bridged", record
            )
        except (FileNotFoundError, ValueError) as exc:
            bridged[unit] = {"status": "NOT_RUN", "reason": str(exc)[:200]}
            continue
        bridged[unit] = {
            "rng_bridges": bridged_meta.get("rng_bridges"),
            "first_non_rng_divergence": bridged_meta.get("first_non_rng_divergence"),
            "stopped": bridged_meta.get("stopped"),
            "decisions": int(len(bridged_tape["actions"])),
        }
    label = verdict(worlds, allocation)
    report = {
        "schema": SCHEMA,
        "mode": mode,
        "commit": commit,
        "namespace": ns,
        "verdict": label,
        "worlds": worlds,
        "class_counts": _counts(worlds),
        "device_correctness_defects": sorted(
            unit
            for unit, row in worlds.items()
            if (row.get("first_divergence") or {}).get("device_correctness_defect")
        ),
        "rng_bridged_diagnostic": {
            "decides_outcome": False,
            "note": "CPU regenerated with device food/respawn draws supplied; not natural identity",
            "worlds": bridged,
        },
        "tier": "Tier L decides; Tier D recorded only (PREREGISTRATION.json)",
    }
    if preregistration is not None:
        report["preregistration"] = {
            "path": str(preregistration),
            "sha256": hashlib.sha256(Path(preregistration).read_bytes()).hexdigest(),
        }
    return report


def _counts(worlds: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in worlds.values():
        divergence = row.get("first_divergence") or {}
        if divergence:
            counts[divergence["class"]] = counts.get(divergence["class"], 0) + 1
    return dict(sorted(counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-exports", type=Path, required=True)
    parser.add_argument("--cpu-root", type=Path, required=True)
    parser.add_argument("--mode", choices=sorted(tape.MODES), required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--preregistration", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite the single registered look: {args.output}")
    report = reduce(
        args.device_exports,
        args.cpu_root,
        mode=args.mode,
        commit=args.commit,
        preregistration=args.preregistration,
    )
    tape.publish_json(args.output, report)
    print(json.dumps({"verdict": report["verdict"], "class_counts": report["class_counts"]}))


if __name__ == "__main__":
    main()
