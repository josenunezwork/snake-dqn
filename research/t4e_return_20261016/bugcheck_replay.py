"""Single-process replay evidence for the T4e RETURN boost investigation.

This is deliberately an audit tool: it reconstructs one persisted decisive
world, never starts training, and writes a create-only SHA-bound receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ[_name] = "1"
os.environ["SNAKE_DQN_DEVICE"] = "cpu"

import numpy as np  # noqa: E402

ARTIFACT_ROOT = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts")
BANK = ARTIFACT_ROOT / "t4e-return-bank-20261016"
SHARD = BANK / "banks/train/world-00003-B8_s0-scripted-s2151847346-decisive.npz"
PLAN = BANK / "plans/cycle-0000.json"
OUT = ARTIFACT_ROOT / "t4e-bugcheck-20261016/replay-v2.json"
DECISION = 3960
DEPTH = 300


def sha256(path: Path) -> str:
    """Return the content hash used to bind this receipt's inputs."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _state(sim: Any) -> dict[str, Any]:
    """Capture transition-relevant hero state without touching simulator internals."""
    return {
        "alive": bool(sim.alive[0, 0]),
        "direction": int(sim.direction[0, 0]),
        "head": [int(value) for value in sim.heads()[0, 0]],
        "length": int(sim.length[0, 0]) if bool(sim.alive[0, 0]) else 0,
        "boost_frames": int(sim.boost_frames[0, 0]),
        "food": len(sim.food_cells[0]),
        "state_sha256": _full_digest(sim),
    }


def _full_digest(sim: Any) -> str:
    """Hash arrays, food/corpse sets, and RNG state of a public replay world."""
    pieces: list[bytes] = []
    for name, value in sorted(vars(sim).items()):
        if isinstance(value, np.ndarray) and name != "_last_reward":
            pieces.extend((name.encode(), str(value.dtype).encode(), repr(value.shape).encode()))
            if value.dtype == object:
                pieces.append(repr(value.tolist()).encode())
            else:
                pieces.append(np.ascontiguousarray(value).tobytes())
        elif name in {"food_cells", "food_set", "corpse_cells"}:
            pieces.extend((name.encode(), repr(value).encode()))
        elif name == "_rngs":
            for rng in value:
                pieces.append(repr(rng._rng.getstate()).encode())
    return hashlib.sha256(b"".join(pieces)).hexdigest()


def _load_rows() -> dict[str, np.ndarray]:
    """Load exactly the six persisted all-action rows at the registered state."""
    with np.load(SHARD, allow_pickle=False) as archive:
        rows = {name: archive[name].copy() for name in archive.files}
    selected = rows["source_decision"] == DECISION
    rows = {name: value[selected] for name, value in rows.items() if value.ndim}
    if len(rows["action"]) != 6 or set(rows["action"].tolist()) != set(range(6)):
        raise RuntimeError("expected the persisted all-action D300 rows 0..5")
    return rows


def _job() -> dict[str, Any]:
    jobs = json.loads(PLAN.read_text())["jobs"]
    return next(dict(job) for job in jobs if int(job["plan_index"]) == 3)


def _capture_world(job: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Replay the prefix and capture the real prepared selector state at D3960."""
    from research.t3_rollout_labels_20261015.bank_engine import build_world, choose_and_step
    from research.t4d_return_regression_20261016.collector import _snapshot

    world = build_world("B8_s0", job, hero_checkpoint=Path(job["student"]))
    captured: dict[str, Any] = {}
    for decision in range(DECISION + 1):
        original = world.runtime.prepare

        def prepare(sim: Any, *, current: int = decision) -> None:
            original(sim)
            if current == DECISION:
                captured["snapshot"] = _snapshot(world, current)

        world.runtime.prepare = prepare
        try:
            choose_and_step(world, capture_obs=False)
        finally:
            world.runtime.prepare = original
        if decision == DECISION:
            break
    snapshot = captured.get("snapshot")
    if snapshot is None:
        raise RuntimeError("registered state did not reconstruct as an eligible selector snapshot")
    return world, snapshot


def _packed_matches(snapshot: dict[str, Any], rows: dict[str, np.ndarray]) -> bool:
    from research.redesign_m3c_20261008.obs_pack import pack_ref, train_layout

    packed = pack_ref(snapshot["obs"]["local"], snapshot["obs"]["global"], train_layout())[0]
    return bool(
        all(np.array_equal(packed, row) for row in rows["state_packed"])
        and all(np.array_equal(snapshot["obs"]["scalars"][0], row) for row in rows["state_scalars"])
    )


def _roll(
    snapshot: dict[str, Any], action: int, *, reward_cfg: bool = False
) -> tuple[dict[str, Any], dict[str, Any], list[float], list[tuple[int, bool]], list[str]]:
    """Roll one forced branch, retaining reward trace and state identity trace."""
    from research.redesign_m3c_20261008.train_m3c import mass_term
    from research.t3_rollout_labels_20261015.bank_engine import choose_and_step, step_prepared

    branch = snapshot["branch"].clone()
    if reward_cfg:
        branch.sim.cfg = replace(branch.sim.cfg, gamma=0.998, kill_scale=0.0, death_value=-3.0)
    rewards, steps, states = [], [], []
    step_prepared(branch, action)
    rewards.append(
        float(branch.sim.get_reward()[0, 0])
        + float(mass_term(branch.sim.get_lengths(), branch.sim.get_alive(), 0.998, 1.0)[0, 0])
    )
    steps.append((int(branch.sim.get_lengths()[0, 0]), bool(branch.sim.get_alive()[0, 0])))
    states.append(_state(branch.sim)["state_sha256"])
    first = _state(branch.sim)
    while len(rewards) < DEPTH and bool(branch.sim.alive[0, 0]):
        choose_and_step(branch, capture_obs=False)
        rewards.append(
            float(branch.sim.get_reward()[0, 0])
            + float(mass_term(branch.sim.get_lengths(), branch.sim.get_alive(), 0.998, 1.0)[0, 0])
        )
        steps.append((int(branch.sim.get_lengths()[0, 0]), bool(branch.sim.get_alive()[0, 0])))
        states.append(_state(branch.sim)["state_sha256"])
    return first, _state(branch.sim), rewards, steps, states


def _independent_first_frame(snapshot: dict[str, Any], action: int) -> dict[str, Any]:
    """Apply the forced action through the simulator's public step lifecycle."""
    from src.simd_env.eval_engine import _dispatch_actions

    branch = snapshot["branch"].clone()
    actions = np.ones((1, branch.sim.S), dtype=np.int64)
    _dispatch_actions(
        branch.sim,
        branch.sim.get_resolved_action_mask(),
        actions,
        [branch.hero],
        [branch.opponents],
    )
    actions[0, 0] = action
    branch.sim.step(actions)
    branch.runtime.observe_step(branch.sim)
    return _state(branch.sim)


def _terminal_matches(branch: dict[str, Any], rows: dict[str, np.ndarray], index: int) -> bool:
    """Byte-check terminal inputs and mask from collector's actual selector lifecycle."""
    from research.redesign_m3c_20261008.obs_pack import pack_ref, train_layout

    if branch["terminal"] is None:
        return bool(
            not rows["terminal_mask"][index].any()
            and not rows["terminal_packed"][index].any()
            and not rows["terminal_scalars"][index].any()
        )
    terminal = branch["terminal"]
    packed = pack_ref(terminal["local"], terminal["global"], train_layout())[0]
    return bool(
        np.array_equal(packed, rows["terminal_packed"][index])
        and np.array_equal(terminal["scalars"][0], rows["terminal_scalars"][index])
        and np.array_equal(branch["terminal_mask"], rows["terminal_mask"][index])
    )


def audit() -> dict[str, Any]:
    """Run the bounded replay and return a JSON-safe, evidence-first receipt."""
    rows, job = _load_rows(), _job()
    _world, snapshot = _capture_world(job)
    legal = sorted(int(value) for value in snapshot["legal"])
    initial = _state(snapshot["branch"].sim)
    if legal != list(range(6)):
        raise RuntimeError(f"registered state lost legal actions: {legal}")
    if not _packed_matches(snapshot, rows):
        raise RuntimeError("reconstructed selector observation differs from persisted bank state")
    outcomes = {}
    for action in legal:
        from research.t4d_return_regression_20261016.collector import rollout_branch

        first, state, rewards, steps, hashes = _roll(snapshot, action)
        independent_first = _independent_first_frame(snapshot, action)
        index = int(np.flatnonzero(rows["action"] == action)[0])
        expected = {name: rows[name][index].item() for name in ("T", "died", "S", "L0", "L_T")}
        actual_s = sum(
            0.998**step * (length / 10.0) for step, (length, alive) in enumerate(steps) if alive
        )
        collector_branch = rollout_branch(snapshot, action, DEPTH)
        first_equal = first["state_sha256"] == independent_first["state_sha256"]
        turn = (-1, 0, 1)[action % 3]
        expected_direction = (initial["direction"] + turn) % 4
        distance = sum(abs(first["head"][i] - initial["head"][i]) for i in range(2))
        action_geometry_matches = (
            first["direction"] == expected_direction
            and distance == (2 if action >= 3 else 1)
            and first["boost_frames"] == (1 if action >= 3 else 0)
        )
        matches = (
            len(rewards) == expected["T"]
            and bool(not state["alive"]) == bool(expected["died"])
            and state["length"] == int(expected["L_T"])
            and int(snapshot["length"]) == int(expected["L0"])
            and abs(actual_s - float(expected["S"])) < 1e-9
            and _terminal_matches(collector_branch, rows, index)
            and first_equal
            and action_geometry_matches
        )
        if not matches:
            raise RuntimeError(f"persisted branch action {action} did not exactly replay")
        outcomes[str(action)] = {
            "boost": action >= 3,
            "first_frame_reconstructed": first,
            "T": len(rewards),
            "died": not state["alive"],
            "L_T": state["length"],
            "S": actual_s,
            "row_fields_match": True,
            "first_frame_public_step_matches_collector": first_equal,
            "relative_action_geometry_matches": action_geometry_matches,
            "terminal_packed_scalars_mask_match": True,
            "transition_trace_sha256": hashlib.sha256("".join(hashes).encode()).hexdigest(),
        }
    boost = next(action for action in legal if action >= 3)
    _first, _normal_state, normal_rewards, _steps, normal_hashes = _roll(
        snapshot, boost, reward_cfg=False
    )
    _first, _reward_state, reward_rewards, _steps, reward_hashes = _roll(
        snapshot, boost, reward_cfg=True
    )
    index = int(np.flatnonzero(rows["action"] == boost)[0])
    gamma = 0.998
    discounted = sum(gamma**step * reward for step, reward in enumerate(reward_rewards))
    closed = (1.0 - gamma) * float(rows["S"][index]) - float(rows["L0"][index]) / 10.0
    if bool(rows["died"][index]):
        closed -= 3.0 * gamma ** (int(rows["T"][index]) - 1)
    else:
        closed += gamma ** int(rows["T"][index]) * float(rows["L_T"][index]) / 10.0
    if normal_hashes != reward_hashes or abs(discounted - closed) >= 1e-8:
        raise RuntimeError("reward-only configuration changed transitions or disagreed with G")
    checkpoint = Path(job["student"])
    return {
        "schema": "t4e-bugcheck-replay/v2",
        "processes": 1,
        "device": "cpu",
        "inputs": {
            str(SHARD): sha256(SHARD),
            str(PLAN): sha256(PLAN),
            str(checkpoint): sha256(checkpoint),
            str(Path(__file__).resolve()): sha256(Path(__file__).resolve()),
        },
        "state": {
            "plan_index": 3,
            "seed": 2151847346,
            "source_decision": DECISION,
            "L0": int(rows["L0"][0]),
            "initial": initial,
            "legal": legal,
            "packed_and_scalars_match": True,
        },
        "action_contract": {
            "relative_actions": {"0-2": "left,straight,right", "3-5": "same turns with boost"},
            "bank_actions": rows["action"].tolist(),
            "mask": legal,
        },
        "branches": outcomes,
        "reward_verification": {
            "action": boost,
            "transition_hashes_equal_after_reward_only_config": normal_hashes == reward_hashes,
            "discounted_trainer_reward": discounted,
            "closed_form_G": closed,
            "absolute_error": abs(discounted - closed),
            "passes": abs(discounted - closed) < 1e-8,
            "note": (
                "Reward is trainer mass (0.002 times post-step living Phi(length)=length/10) "
                "plus simulator PBRS. Collector gamma=.99 and kill=.3 are unused; replay "
                "overrides only reward fields to gamma=.998, kill=0, death=-3."
            ),
        },
        "limitation": (
            "No persisted first-frame archive exists; the first frame is reconstructed twice, "
            "once by collector step_prepared and once by direct public simulator "
            "dispatch/step/observe. All six D300 rows were byte-bound to the published "
            "shard's state and terminal inputs."
        ),
    }


def main() -> None:
    """Create the required artifact without overwriting an existing receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    out = parser.parse_args().out
    if out.exists():
        raise FileExistsError(f"create-only receipt already exists: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = audit()
    payload["supersedes"] = [str(OUT.with_name("replay-v1.json"))]
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(out), "sha256": sha256(out)}, sort_keys=True))


if __name__ == "__main__":
    main()
