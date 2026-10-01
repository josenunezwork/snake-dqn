"""End-to-end parity: opt-in SIMD vector61 (Apex) policy vs the live AISnake path.

The reference is the real live evaluator, ``tournament_eval.rollout`` under the
promotion-v2-watch-rect profile (``GameState`` + ``AISnake`` + carry-forward
selection + optional ``FreeSpaceVeto``). The candidate is
``run_simd_eval(..., vector61=True)`` on the same world seeds and rosters.
Every vector-policy decision (hero and checkpoint opponents) is compared per
``(seed, frame, slot)``; the profiled records must be identical too.

Two families of worlds:

* a tiny self-contained world with random 61-D networks (always runs), which
  forces deaths, opponent respawns (carry invalidation) and veto overrides;
* the real deployment world (``research/apex_safety_20260926/deployment.yaml``)
  with the real champion pool (skipped when the checkpoints are absent).

No test here calls a screen ``main``; horizons are at most a few hundred frames.
"""

from __future__ import annotations

import random
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pytest
import torch

from src.simd_env.vector61_policy import (
    VETO_KEPT,
    VETO_NO_SPACIOUS,
    VETO_VETOED,
    batched_veto_choice,
    free_space_counts,
    veto_threshold,
)

REPO = Path(__file__).resolve().parents[1]
DEPLOYMENT_YAML = REPO / "research" / "apex_safety_20260926" / "deployment.yaml"
CHECKPOINT_DIR = Path("/Users/josenunez/Projects/ml/snake-dqn/saved_snakes")
POOL_NAMES = (
    "champion_a5_freespace_20260621.pth",
    "best_apex_fs.pth",
    "best_apex_stage1_fs.pth",
    "best_apex_pre_fs.pth",
)
# Record keys that only one engine emits (SIMD binds its storage runtime).
SIMD_ONLY_KEYS = {"world_runtime_spec", "world_runtime_spec_digest"}

Key = Tuple[int, int, int]  # (seed, frame, slot)


# ---------------------------------------------------------------------------
# Pure veto rule
# ---------------------------------------------------------------------------
class TestBatchedVetoRule:
    def test_matches_scalar_rule_on_random_and_tied_inputs(self):
        from src.evaluation.safety_veto import veto_choice

        gen = np.random.default_rng(0)
        n = 20000
        q = gen.normal(size=(n, 6)).astype(np.float32)
        q[: n // 4] = np.round(q[: n // 4])  # many exact ties
        mask = gen.random((n, 6)) < 0.6
        spacious = gen.random((n, 3)) < 0.5
        base = gen.integers(0, 6, size=n)
        masked = np.where(mask, q, np.float32(-1.0e9))
        actions, outcomes = batched_veto_choice(masked, mask, spacious, base)
        labels = {"kept": VETO_KEPT, "vetoed": VETO_VETOED, "no_spacious": VETO_NO_SPACIOUS}
        for i in range(n):
            ref_action, ref_outcome = veto_choice(
                masked[i].tolist(), mask[i].tolist(), spacious[i].tolist(), int(base[i])
            )
            assert (int(actions[i]), int(outcomes[i])) == (ref_action, labels[ref_outcome]), i
        assert {VETO_KEPT, VETO_VETOED, VETO_NO_SPACIOUS} <= set(outcomes.tolist())

    def test_veto_threshold_matches_live_formula(self):
        from src.evaluation.safety_veto import free_space_threshold

        lengths = np.arange(0, 400)
        cap, need = veto_threshold(lengths)
        live = [free_space_threshold(int(n), max(1, int(n))) for n in lengths]
        assert list(zip(cap.tolist(), need.tolist())) == live

    def test_free_space_counts_round_trip_through_float32(self):
        for length in range(1, 120):
            cap = min(160, max(32, 2 * length))
            counts = np.arange(cap + 1)
            feats = np.minimum(counts / cap, 1.0).astype(np.float32)
            states = np.zeros((len(counts), 61), dtype=np.float32)
            states[:, 58:61] = feats[:, None]
            got, got_cap = free_space_counts(states, np.full(len(counts), length))
            assert np.all(got_cap == cap)
            assert np.array_equal(got, np.repeat(counts[:, None], 3, axis=1))
            # The live rule: round(feature * cap) on the float64 feature.
            assert [round(float(c / cap) * cap) for c in counts] == counts.tolist()


# ---------------------------------------------------------------------------
# Live vs SIMD harness
# ---------------------------------------------------------------------------
def _profile(frames: int):
    from src.evaluation.protocol import promotion_v2_watch_rect
    from src.scripts import tournament_eval as te

    return replace(
        promotion_v2_watch_rect(te._evaluation_world_from_config()), scored_horizon=frames
    )


def _run_live(
    monkeypatch: pytest.MonkeyPatch,
    hero: Tuple[str, str],
    rosters: Dict[int, List[Tuple[str, str]]],
    frames: int,
    seeds: Sequence[int],
    mix_id: str,
    veto: bool,
) -> Tuple[List[dict], Dict[Key, int], Dict[Key, np.ndarray], int]:
    """Real ``tournament_eval.rollout`` per seed, recording every AISnake decision.

    Also returns each decision's selection state and the number of ambient
    pellets the profile loop's ``trim_ambient`` removed (the one live step the
    SIMD engine has no equivalent of).
    """
    from src.game.ai_snake import AISnake
    from src.game.food_manager import FoodManager
    from src.scripts import tournament_eval as te

    actions: Dict[Key, int] = {}
    states: Dict[Key, np.ndarray] = {}
    trimmed = {"pellets": 0}
    current = {"seed": -1}
    original = AISnake.update
    original_trim = FoodManager.trim_ambient

    def recording_update(self, other_snakes, food, **kwargs):
        original(self, other_snakes, food, **kwargs)
        key = (current["seed"], int(self._get_frame()), int(self.id))
        assert key not in actions
        actions[key] = int(self._pre_collision_action)
        states[key] = self._pre_collision_state.detach().cpu().numpy().copy()

    def counting_trim(self, target):
        removed = original_trim(self, target)
        trimmed["pellets"] += int(removed)
        return removed

    monkeypatch.setattr(AISnake, "update", recording_update)
    monkeypatch.setattr(FoodManager, "trim_ambient", counting_trim)
    records = []
    profile = _profile(frames)
    for seed in seeds:
        current["seed"] = int(seed)
        records.append(
            te.rollout(
                hero,
                rosters[seed],
                frames,
                int(seed),
                profile,
                mix_id=mix_id,
                hero_safety_veto=veto,
            )
        )
    monkeypatch.setattr(AISnake, "update", original)
    monkeypatch.setattr(FoodManager, "trim_ambient", original_trim)
    return records, actions, states, trimmed["pellets"]


def _run_simd(
    monkeypatch: pytest.MonkeyPatch,
    hero: Tuple[str, str],
    rosters: Dict[int, List[Tuple[str, str]]],
    frames: int,
    seeds: Sequence[int],
    mix_id: str,
    veto: bool,
    forward: str = "rowwise",
) -> Tuple[List[dict], Dict[Key, int], Dict[str, int], Dict[Key, np.ndarray]]:
    """``run_simd_eval(vector61=True)`` over all seeds as one batch, recording decisions."""
    from src.simd_env import eval_engine as ee
    from src.simd_env.vector61_policy import Vector61SimdPolicy

    actions: Dict[Key, int] = {}
    states: Dict[Key, np.ndarray] = {}
    stats = {"fresh_after_first_frame": 0, "rows": 0}
    original = Vector61SimdPolicy.actions

    def recording_actions(self, masks, sim, slots):
        out = original(self, masks, sim, slots)
        for (env, slot), action in zip(slots, out):
            key = (int(seeds[int(env)]), int(sim.frame[int(env)]), int(slot))
            assert key not in actions
            actions[key] = int(action)
        return out

    def trace(frame: dict) -> None:
        stats["rows"] += int(frame["ready"].sum())
        late = frame["frame"] > 1
        stats["fresh_after_first_frame"] += int(frame["fresh"][late].sum())
        for env, slot in np.argwhere(frame["ready"]):
            key = (int(seeds[env]), int(frame["frame"][env]), int(slot))
            states[key] = frame["state"][env, slot].copy()

    monkeypatch.setattr(Vector61SimdPolicy, "actions", recording_actions)
    records = ee.run_simd_eval(
        hero,
        rosters[seeds[0]],
        frames,
        list(seeds),
        profile=_profile(frames),
        opponent_specs_by_world=rosters,
        mix_id=mix_id,
        vector61=True,
        hero_safety_veto=veto,
        vector61_forward=forward,
        vector61_trace=trace,
    )
    monkeypatch.setattr(Vector61SimdPolicy, "actions", original)
    return records, actions, stats, states


def _first_state_divergence(
    live: Dict[Key, np.ndarray], simd: Dict[Key, np.ndarray]
) -> Tuple[int, object]:
    """Count bit-identical float32 selection states; return the first mismatch."""
    same, first = 0, None
    for key in sorted(set(live) | set(simd)):
        a, b = live.get(key), simd.get(key)
        if a is not None and b is not None and np.array_equal(a.view(np.uint32), b.view(np.uint32)):
            same += 1
        elif first is None:
            cols = None if a is None or b is None else np.flatnonzero(a != b).tolist()
            first = {"key": key, "columns": cols}
    return same, first


def _compare(live: Dict[Key, int], simd: Dict[Key, int]) -> Dict[str, object]:
    keys = sorted(set(live) | set(simd))
    matches = sum(live.get(k) == simd.get(k) for k in keys)
    first = next((k for k in keys if live.get(k) != simd.get(k)), None)
    return {
        "decisions": len(keys),
        "matches": matches,
        "first_divergence": (
            None
            if first is None
            else {"key": first, "live": live.get(first), "simd": simd.get(first)}
        ),
    }


def _summary(report: Dict[str, object]) -> Dict[str, object]:
    """Compact evidence line (printed with ``pytest -s``)."""
    records = report["records"]
    return {
        "decisions": report["decisions"],
        "matches": report["matches"],
        "bit_identical_states": report["bit_identical_states"],
        "hero_decisions": report["hero_decisions"],
        "veto_cramped_decisions": report.get("veto_cramped_decisions"),
        "veto_boundary_decisions": report.get("veto_boundary_decisions"),
        "fresh_after_first_frame": report["fresh_after_first_frame"],
        "deaths": [r["deaths"] for r in records],
        "kills": [r["kills"] for r in records],
        "mass": [round(r["mass_integral"], 4) for r in records],
        "vetoes": [
            r["probes"].get("safety_veto", {}).get("counters", {}).get("vetoes_applied")
            for r in records
        ],
    }


def _install_veto_spies(monkeypatch: pytest.MonkeyPatch, seeds: Sequence[int]):
    """Record every veto decision's raw counts, ``need``, base and final action.

    Live: one ``FreeSpaceVeto`` instance per rollout (seed order); the counts
    are recovered with the live helpers (``round(feature * cap)``) right before
    the real ``apply``. SIMD: the counts/need the batched path actually used.
    """
    from src.evaluation import safety_veto as sv
    from src.simd_env import vector61_policy as vp

    live_logs: Dict[int, list] = {}
    keep_alive: list = []  # pin instances so id() keys are never reused
    simd_logs: Dict[int, list] = {int(seed): [] for seed in seeds}
    original_apply = sv.FreeSpaceVeto.apply
    original_apply_veto = vp.Vector61SimdPolicy._apply_veto

    def live_apply(self, snake, other_snakes, q_values, action_mask, base_action):
        features = snake._get_free_space_features(list(other_snakes))
        cap, need = sv.free_space_threshold(snake.length, snake._logical_length())
        counts = tuple(round(float(f) * cap) for f in features)
        action = original_apply(self, snake, other_snakes, q_values, action_mask, base_action)
        if id(self) not in live_logs:
            keep_alive.append(self)
        live_logs.setdefault(id(self), []).append(
            (int(snake._get_frame()), counts, int(need), int(base_action), int(action))
        )
        return action

    def simd_apply_veto(self, sim, slots, masked_q, mask, actions):
        self.last_veto = None
        out = original_apply_veto(self, sim, slots, masked_q, mask, actions)
        last = self.last_veto
        if last is not None:
            for i, (env, _slot) in enumerate(last["rows"]):
                simd_logs[int(seeds[int(env)])].append(
                    (
                        int(last["frame"][i]),
                        tuple(int(c) for c in last["counts"][i]),
                        int(last["need"][i]),
                        int(last["base"][i]),
                        int(last["final"][i]),
                    )
                )
        return out

    monkeypatch.setattr(sv.FreeSpaceVeto, "apply", live_apply)
    monkeypatch.setattr(vp.Vector61SimdPolicy, "_apply_veto", simd_apply_veto)

    def collected() -> Tuple[Dict[int, list], Dict[int, list]]:
        ordered = list(live_logs.values())
        assert len(ordered) == len(seeds)
        return {int(seed): log for seed, log in zip(seeds, ordered)}, simd_logs

    return collected


def _assert_parity(
    monkeypatch: pytest.MonkeyPatch,
    hero: Tuple[str, str],
    rosters: Dict[int, List[Tuple[str, str]]],
    frames: int,
    seeds: Sequence[int],
    mix_id: str,
    veto: bool,
) -> Dict[str, object]:
    veto_logs = _install_veto_spies(monkeypatch, seeds) if veto else None
    live_records, live_actions, live_states, trimmed = _run_live(
        monkeypatch, hero, rosters, frames, seeds, mix_id, veto
    )
    simd_records, simd_actions, stats, simd_states = _run_simd(
        monkeypatch, hero, rosters, frames, seeds, mix_id, veto
    )
    # Known world gap: the live profile loop trims ambient food after every
    # update and BatchSim does not. It must stay a no-op for parity to hold.
    assert trimmed == 0, f"live trim_ambient removed {trimmed} pellets"
    report = _compare(live_actions, simd_actions)
    same_states, first_state = _first_state_divergence(live_states, simd_states)
    report["bit_identical_states"] = same_states
    assert first_state is None, (first_state, report)
    assert report["first_divergence"] is None, report
    assert report["decisions"] == stats["rows"]
    for live, simd in zip(live_records, simd_records):
        simd_view = {k: v for k, v in simd.items() if k not in SIMD_ONLY_KEYS}
        assert simd_view == live, (live, simd)
    hero_decisions = sum(1 for key in live_actions if key[2] == 0)
    if veto_logs is not None:
        live_veto, simd_veto = veto_logs()
        for seed in seeds:
            assert simd_veto[int(seed)] == live_veto[int(seed)], seed
        entries = [entry for log in live_veto.values() for entry in log]
        report["veto_decisions"] = len(entries)
        report["veto_cramped_decisions"] = sum(min(e[1]) < e[2] for e in entries)
        report["veto_boundary_decisions"] = sum(e[2] in e[1] for e in entries)
    report.update(stats)
    report["hero_decisions"] = hero_decisions
    report["records"] = live_records
    return report


# ---------------------------------------------------------------------------
# Tiny self-contained world (random 61-D networks)
# ---------------------------------------------------------------------------
TINY_YAML = """\
network:
  input_size: 61
  output_size: 6
  use_free_space: true
game:
  mechanics_version: 2
  width: 300
  height: 200
  num_snakes: 5
  initial_food: 40
  max_food: 40
  max_frames: 5000
  frame_rate: 1
  max_length: 150
pqn:
  max_capacity: 400
"""


@pytest.fixture
def tiny_world(setup_config, tmp_path):
    """Install the tiny v2 world; return a factory of random vector checkpoints."""
    from src.core.config_loader import load_and_initialize_config
    from src.model.apex_network import ApexNetwork

    cfg = tmp_path / "tiny_v61.yaml"
    cfg.write_text(TINY_YAML)
    load_and_initialize_config(str(cfg))

    def checkpoint(name: str, seed: int) -> Tuple[str, str]:
        torch.manual_seed(seed)
        net = ApexNetwork(input_size=61, hidden_size=32, output_size=6)
        with torch.no_grad():
            for param in net.parameters():
                param.mul_(3.0)  # sharper preferences: more varied, decisive actions
        path = tmp_path / f"{name}.pth"
        torch.save({"dqn_state_dict": net.state_dict(), "input_size": 61, "hidden_size": 32}, path)
        return ("checkpoint", str(path))

    yield checkpoint
    random.seed()


def _tiny_rosters(make, seeds: Sequence[int]) -> Tuple[tuple, Dict[int, list]]:
    hero = make("hero", 1)
    rival_a = make("rival_a", 2)
    rival_b = make("rival_b", 3)
    roster = [rival_a, ("scripted", "random_safe"), rival_b, hero]
    return hero, {int(seed): list(roster) for seed in seeds}


@pytest.mark.parametrize("veto", [False, True])
def test_tiny_world_live_and_simd_vector61_decisions_are_identical(tiny_world, monkeypatch, veto):
    """Every vector decision, death and record matches the live evaluator exactly."""
    seeds = (3, 4, 5, 6)
    hero, rosters = _tiny_rosters(tiny_world, seeds)
    report = _assert_parity(monkeypatch, hero, rosters, 300, seeds, "tiny-v61", veto)
    print(_summary(report))
    assert report["matches"] == report["decisions"] > 1000
    # Coverage: opponent respawns forced fresh rebuilds (carry invalidation),
    # and the hero both survived and died across worlds.
    assert report["fresh_after_first_frame"] > 0, report
    assert sum(r["deaths"] for r in report["records"]) >= 1, report
    if veto:
        counters = [r["probes"]["safety_veto"]["counters"] for r in report["records"]]
        assert sum(c["vetoes_applied"] for c in counters) > 0, counters
        assert sum(c["decisions"] for c in counters) == report["hero_decisions"]


def test_default_off_vector61_checkpoint_still_raises(tiny_world):
    from src.simd_env import eval_engine as ee

    hero, rosters = _tiny_rosters(tiny_world, (3,))
    with pytest.raises(ValueError, match="Use --engine live"):
        ee.run_simd_eval(hero, rosters[3], 5, [3], profile=_profile(5))
    with pytest.raises(ValueError, match="vector61 checkpoint hero"):
        ee.run_simd_eval(hero, rosters[3], 5, [3], profile=_profile(5), hero_safety_veto=True)
    with pytest.raises(ValueError, match="explicit evaluation profile"):
        ee.run_simd_eval(hero, rosters[3], 5, [3], vector61=True)
    scripted = ("scripted", "greedy_food")
    with pytest.raises(ValueError, match="vector61 checkpoint hero"):
        ee.run_simd_eval(
            scripted, rosters[3], 5, [3], profile=_profile(5), vector61=True, hero_safety_veto=True
        )


def test_hero_veto_never_reaches_a_same_checkpoint_opponent(tiny_world, monkeypatch):
    """The hero checkpoint also sits in slot 4; only slot 0 decisions are vetoed."""
    from src.simd_env.vector61_policy import Vector61SimdPolicy

    seen: Dict[bool, set] = {True: set(), False: set()}
    original = Vector61SimdPolicy.actions

    def spy(self, masks, sim, slots):
        seen[bool(self.veto_slots)].update(int(s) for s in np.asarray(slots)[:, 1])
        return original(self, masks, sim, slots)

    monkeypatch.setattr(Vector61SimdPolicy, "actions", spy)
    hero, rosters = _tiny_rosters(tiny_world, (3,))
    from src.simd_env import eval_engine as ee

    ee.run_simd_eval(
        hero, rosters[3], 20, [3], profile=_profile(20), vector61=True, hero_safety_veto=True
    )
    assert seen[True] == {0}
    assert 4 in seen[False] and 0 not in seen[False]


def test_batched_forward_selects_the_same_actions_as_rowwise(tiny_world, monkeypatch):
    """The faster batched forward is checked against the bit-exact rowwise path."""
    seeds = (3, 4)
    hero, rosters = _tiny_rosters(tiny_world, seeds)
    rowwise = _run_simd(monkeypatch, hero, rosters, 200, seeds, "tiny-v61", False, "rowwise")
    batched = _run_simd(monkeypatch, hero, rosters, 200, seeds, "tiny-v61", False, "batched")
    # Not bit-exact by construction (BLAS blocking changes low Q bits), so only
    # the selected actions and the resulting records are compared.
    assert _compare(rowwise[1], batched[1])["first_divergence"] is None
    assert rowwise[0] == batched[0]


# ---------------------------------------------------------------------------
# Real deployment world + real champion pool (skipped when absent)
# ---------------------------------------------------------------------------
_POOL_PATHS = [CHECKPOINT_DIR / name for name in POOL_NAMES]
needs_real_pool = pytest.mark.skipif(
    not DEPLOYMENT_YAML.exists() or not all(path.exists() for path in _POOL_PATHS),
    reason="deployment config or champion pool checkpoints not available",
)


@pytest.fixture
def deployment_world(setup_config):
    from src.core.config_loader import load_and_initialize_config

    load_and_initialize_config(str(DEPLOYMENT_YAML))
    yield [("checkpoint", str(path)) for path in _POOL_PATHS]
    random.seed()


@needs_real_pool
@pytest.mark.parametrize(
    "mix, veto, frames, seeds",
    [
        # 500 frames on these worlds include real veto overrides (seed 13) ...
        ("frozen", True, 500, (11, 12, 13)),
        # ... and a hero death (seed 13) plus scripted-opponent respawns.
        ("mixed", False, 500, (11, 12, 13)),
        ("scripted", True, 300, (11, 12)),
        ("frozen", False, 300, (11, 12)),
    ],
)
def test_deployment_world_champion_decisions_match_live(
    deployment_world, monkeypatch, mix, veto, frames, seeds
):
    """Champion hero vs the strict pilot pool mixes at the deployment profile."""
    from src.scripts import tournament_eval as te

    pool = deployment_world
    hero = pool[0]
    specs = te.build_mix_specs(mix, 5, pool)
    rosters = te.materialize_opponent_specs_by_world(specs, seeds)
    report = _assert_parity(monkeypatch, hero, rosters, frames, seeds, mix, veto)
    print(mix, veto, _summary(report))
    assert report["matches"] == report["decisions"] >= len(seeds) * 200
    if veto:
        counters = [r["probes"]["safety_veto"]["counters"] for r in report["records"]]
        assert sum(c["decisions"] for c in counters) == report["hero_decisions"]
        if frames >= 500:
            assert sum(c["vetoes_applied"] for c in counters) > 0, counters
    elif frames >= 500:
        assert sum(r["deaths"] for r in report["records"]) >= 1
