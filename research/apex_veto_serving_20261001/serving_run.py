"""Web serving run for the Apex champion + free-space veto (vector61 + wrapper).

Drives the REAL web backend headlessly: ``web.backend.session.GameSession`` is
stepped directly (``step()`` then ``snapshot()``, the order of the app's
``Hub.engine_loop``) and every control goes through ``web.backend.app._apply_control``
(the WebSocket dispatch the browser uses). No server, no network listener.

Episodes (pre-registered in ``protocol.md``):

* 25 Watch episodes as served: default served config, all-respawn Watch, the
  wrapper on the Watch hero only (``SNAKE_SERVE_VETO_WATCH_HERO`` semantics, the
  scope a release would turn on);
* 25 Play episodes: a scripted stand-in steers the human through
  ``human_input`` controls; every AI snake is wrapped (``SNAKE_SERVE_VETO_PLAY_AI``);
* 2 parity probes (not served episodes): the same session stepping code on the
  promotion-v2-watch-rect deployment config with a terminal hero, compared frame
  by frame with ``tournament_eval.rollout(..., hero_safety_veto=True)`` on the
  same seed.

Every record binds the checkpoint sha256, the wrapper descriptor and source
sha256, the served config, and the veto counters. ``serving_audit.py`` (stdlib)
checks them. Output is create-only under ``--out``.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _load_schema() -> Any:
    spec = importlib.util.spec_from_file_location("apex_veto_serving_schema", HERE / "schema.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


schema = _load_schema()

PROTOCOL = HERE / "protocol.md"
# sha256 of protocol.md as pre-registered for the real run (the audit pins it too).
# Any protocol edit must re-pin both, deliberately; a test asserts they match.
PROTOCOL_SHA256 = "9b65e9c568c9de76ca7aa3b3af1bb1e232a9ada5f39083062dc0b8b58d121809"
CHAMPION_PATH = Path(
    "/Users/josenunez/Projects/ml/snake-dqn/saved_snakes/champion_a5_freespace_20260621.pth"
)
CHAMPION_SHA256 = "43d4e2c53919dd59416c145cf0ba7c4faf1c7f298eebbb1723146807d747ac93"
WRAPPER_SOURCE_SHA256 = "1b62d15c48987584533efbefdf3fd84e1e3f2e22cfe70dbb82b43ec57169c428"
SERVED_CONFIG_SHA256 = "c2db7607915f70eaa46489a48598db65c4593cf039a166f37842423bb1479911"
PARITY_CONFIG = REPO / "research/apex_safety_20260926/deployment.yaml"
PARITY_CONFIG_SHA256 = "4146baa3a06102b8afd627b1fba8384e9a2f47aaac4a9bc96292c3eb71715aa5"
PROFILE_NAME = "promotion-v2-watch-rect"

HORIZON = 5000
COUNTS = {"watch": 25, "play": 25, "parity": 2}
SMOKE_HORIZON = 500
SMOKE_PARITY_HORIZON = 200
SMOKE_COUNTS = {"watch": 1, "play": 1, "parity": 1}
SEED_DOMAIN = "apex-veto-web-serving-v1"
SMOKE_SEED_DOMAIN = "apex-veto-web-serving-smoke-v1"
ARTIFACT_ROOT = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts")
CONN_ID = "apex-veto-serving-run"
STAND_IN_TURN_PROBABILITY = 0.08
V3_SCREEN_DOMAINS = (
    "apex-veto-v3-screen-v1",
    "apex-veto-v3-screen-v2",
    "apex-veto-v3-screen-smoke-v1",
)
V3_SCREEN_PREFIX_CHECKED = 1000


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def write_new_json(path: Path, value: Any) -> str:
    """Create-only canonical-ish JSON write; returns the file's sha256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    import json

    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return schema.sha256_file(path)


def world_seeds(domain: str, counts: Mapping[str, int]) -> Dict[str, List[int]]:
    """Ordered seeds per purpose (the dev_screen uint32 SHA-prefix recipe)."""
    from research.apex_safety_20260926 import dev_screen

    return {
        purpose: [dev_screen.uint32_seed(domain, purpose, i) for i in range(count)]
        for purpose, count in counts.items()
    }


def seed_report(seeds: Mapping[str, Sequence[int]]) -> Dict[str, Any]:
    """In-code freshness check against every strict-veto and screen bank (fail closed)."""
    from research.apex_safety_20260926 import dev_screen
    from research.apex_veto_strict_20260927 import strict_run

    earlier = set(dev_screen.screen_seeds(strict_run.SCREEN_PREFIX_CHECKED))
    for domain, count in {
        **strict_run.RUN_V1_NAMESPACES,
        **dict(strict_run.NAMESPACES.values()),
    }.items():
        earlier |= {
            dev_screen.uint32_seed(domain, strict_run.NAMESPACE_KEY, i) for i in range(count)
        }
    for values in dev_screen.challenger_namespaces().values():
        earlier |= set(values)
    # Concurrent v3 tail-aware screen (research/apex_veto_v3_screen_20261001); it
    # excludes this lane's domains in turn, so the two banks are disjoint both ways.
    for domain in V3_SCREEN_DOMAINS:
        earlier |= {
            dev_screen.uint32_seed(domain, "worlds", i) for i in range(V3_SCREEN_PREFIX_CHECKED)
        }
    flat = [seed for values in seeds.values() for seed in values]
    overlap = sorted(earlier & set(flat))
    return {
        "unique": len(set(flat)) == len(flat),
        "strict_and_screen_overlap": overlap,
        "checked_against": [
            "apex-veto-strict dev/final/serving v1-v3",
            "apex-safety-screen-v1[:1000]",
            "task-aligned-challenger-20260924 namespaces",
            "apex-veto-v3-screen-v1/worlds[:1000] (+ smoke)",
        ],
        "registry_checked": False,
        "registry_note": "governance namespace registry not implemented; in-code check only",
        "disjoint": len(set(flat)) == len(flat) and not overlap,
    }


def git_state() -> Dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True
        ).stdout

    return {"commit": run("rev-parse", "HEAD").strip(), "dirty": bool(run("status", "--porcelain"))}


@contextlib.contextmanager
def scrubbed_veto_env() -> Iterator[None]:
    """Temporarily remove the two serving flags from the environment."""
    from web.backend.safety_veto_serving import ENV_PLAY_AI, ENV_WATCH_HERO

    saved = {key: os.environ.pop(key) for key in (ENV_WATCH_HERO, ENV_PLAY_AI) if key in os.environ}
    try:
        yield
    finally:
        os.environ.update(saved)


def default_check(checkpoint: Path) -> Dict[str, Any]:
    """Criterion S6: with no flag set, nothing is wrapped and served defaults are unchanged."""
    from web.backend import session as web_session
    from web.backend.app import _apply_control
    from web.backend.safety_veto_serving import ServingVetoFlags

    out: Dict[str, Any] = {
        "flags_from_empty_env": ServingVetoFlags.from_env({}).to_dict(),
        "default_checkpoint_basename": os.path.basename(web_session.DEFAULT_CHECKPOINT),
        "served_config_basename": os.path.basename(web_session._config_for(61)),
        "served_config_sha256": schema.sha256_file(Path(web_session._config_for(61))),
    }
    with scrubbed_veto_env():
        sess = web_session.GameSession(checkpoint=str(checkpoint))
        modes = {}
        for mode in ("watch", "play"):
            if mode != sess.mode:
                _apply_control(sess, {"type": "control", "action": "set_mode", "value": mode})
            state = sess.safety_veto_state()
            modes[mode] = {
                "mode": sess.mode,
                "active": bool(state and state["active"]),
                "snakes_with_veto": [
                    int(s.id) for s in sess.game.snakes if getattr(s, "safety_veto", None)
                ],
            }
    out["modes"] = modes
    out["pass"] = (
        out["flags_from_empty_env"] == {"watch_hero": False, "play_ai": False}
        and out["default_checkpoint_basename"] == CHAMPION_PATH.name
        and out["served_config_basename"] == "mechanics_v2.yaml"
        and out["served_config_sha256"] == SERVED_CONFIG_SHA256
        and all(
            m["mode"] == k and not m["active"] and not m["snakes_with_veto"]
            for k, m in modes.items()
        )
    )
    return out


# ---------------------------------------------------------------- served identity


def served_identity(sess: Any) -> Dict[str, Any]:
    """The configuration the session is actually serving (read back, not assumed)."""
    from src.core.game_config import GameConfig

    config_path = Path(sess.config_path)
    return {
        "mode": sess.mode,
        "config_path": str(config_path),
        "config_basename": config_path.name,
        "config_sha256": schema.sha256_file(config_path) if config_path.is_file() else None,
        "obs_spec": str(sess.obs_spec),
        "policy_class": type(sess.policy).__name__,
        "policy_training": bool(getattr(sess.policy, "training", False)),
        "epsilon": float(getattr(sess.policy, "epsilon", 0.0)),
        "protocol_version": int(sess.protocol_version),
        "num_snakes": len(sess.game.snakes),
        "world": {
            "width": int(GameConfig.WIDTH),
            "height": int(GameConfig.HEIGHT),
            "mechanics_version": int(GameConfig.MECHANICS_VERSION),
            "initial_food": int(GameConfig.INITIAL_FOOD),
            "max_food": int(sess.game.food_manager.max_food),
            "input_size": int(GameConfig.INPUT_SIZE),
        },
        "hero_id": int(sess.hero_id),
        "human_id": None if sess.human_id is None else int(sess.human_id),
        "snake_ids": [int(s.id) for s in sess.game.snakes],
        "veto_flags": sess.safety_veto.flags.to_dict(),
    }


def counter_delta(after: Mapping[str, int], before: Mapping[str, int]) -> Dict[str, int]:
    return {key: int(after[key]) - int(before.get(key, 0)) for key in schema.COUNTER_KEYS}


class DecisionCounter:
    """Harness-side count of frames on which each wrapped snake selected an action.

    Source of truth: ``GameState.update`` respawns first, then calls ``update`` on
    every snake alive at that point, and resolves deaths after all moves. So a
    snake decided on an advanced frame iff it was alive before the step, is alive
    after it, or appears in that frame's ``frame_death_causes``.
    """

    def __init__(self, snake_ids: Sequence[int]) -> None:
        self.counts = {int(sid): 0 for sid in snake_ids}

    def step(self, sess: Any) -> None:
        game = sess.game
        by_id = {int(s.id): s for s in game.snakes}
        pre = {sid: bool(by_id[sid].is_alive) for sid in self.counts}
        frame = game.frame
        sess.step()
        sess.snapshot()
        if sess.game is not game or game.frame == frame:
            return
        causes = game.frame_death_causes
        for sid in self.counts:
            if pre[sid] or by_id[sid].is_alive or sid in causes:
                self.counts[sid] += 1


class ScriptedHuman:
    """Seeded stand-in for a browser player: wall/body-avoiding wanderer.

    It reads the served frame state and answers with the same named directions
    the browser's arrow keys send. It never boosts. It is a coverage driver for
    the serving path, not a model of human skill.
    """

    DIRECTIONS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}

    def __init__(self, seed: int) -> None:
        self.rng = random.Random(f"stand-in/{int(seed)}")

    def choose(self, human: Any, game: Any, must_turn: bool) -> Optional[str]:
        seg = int(human.segment_size)
        width, height = int(human.game_width), int(human.game_height)
        occupied = {(int(x), int(y)) for s in game.snakes if s.is_alive for (x, y) in s.segments}
        hx, hy = (int(v) for v in human.head)
        cur = tuple(int(v) for v in human.direction)

        def safe(vec: tuple) -> bool:
            for k in (1, 2):
                x, y = hx + vec[0] * seg * k, hy + vec[1] * seg * k
                if not (0 <= x < width and 0 <= y < height) or (x, y) in occupied:
                    return False
            return True

        names = sorted(n for n, v in self.DIRECTIONS.items() if v != (-cur[0], -cur[1]))
        current = next((n for n in names if self.DIRECTIONS[n] == cur), None)
        safe_names = [n for n in names if safe(self.DIRECTIONS[n])]
        if not must_turn and current in safe_names:
            if self.rng.random() >= STAND_IN_TURN_PROBABILITY:
                return None
        turns = [n for n in safe_names if n != current] or [n for n in names if n != current]
        return self.rng.choice(turns) if turns else None


def control(sess: Any, action: str, value: Any = None) -> None:
    """One control message through the browser's dispatch path; fail on any refusal.

    ``web.backend.app._apply_control`` reports a failed handler (and an unknown
    action, or a refused mode switch) on ``session.last_error`` and returns None,
    so a None reply alone is not success. ``last_error`` is cleared before the
    dispatch and must still be None after it. A failure raises, and the episode
    runner records it in the record's ``error`` (status ``error``), so S1 fails.
    """
    from web.backend.app import _apply_control

    sess.last_error = None
    reply = _apply_control(sess, {"type": "control", "action": action, "value": value}, CONN_ID)
    require(reply is None, f"control {action!r} was refused: {reply}")
    require(sess.last_error is None, f"control {action!r} failed: last_error={sess.last_error!r}")


def set_world_seed(seed: int) -> None:
    from src.scripts.eval_cli import set_seed

    set_seed(int(seed))


def veto_block(sess: Any, before: Mapping[str, Mapping[str, int]]) -> Dict[str, Any]:
    state = sess.safety_veto_state()
    per_snake = {
        sid: counter_delta(counters, before.get(sid, {}))
        for sid, counters in state["counters"].items()
    }
    total = {key: sum(c[key] for c in per_snake.values()) for key in schema.COUNTER_KEYS}
    return {
        "active": state["active"],
        "scope": state["scope"],
        "reason": state["reason"],
        "flags": state["flags"],
        "wrapped_snake_ids": state["wrapped_snake_ids"],
        "per_snake": per_snake,
        "total": total,
    }


# ---------------------------------------------------------------- served episodes


@contextlib.contextmanager
def veto_env(watch_hero: bool, play_ai: bool) -> Iterator[None]:
    """Set the two serving flags exactly as an operator's environment would."""
    from web.backend.safety_veto_serving import ENV_PLAY_AI, ENV_WATCH_HERO

    with scrubbed_veto_env():
        os.environ[ENV_WATCH_HERO] = "1" if watch_hero else "0"
        os.environ[ENV_PLAY_AI] = "1" if play_ai else "0"
        try:
            yield
        finally:
            os.environ.pop(ENV_WATCH_HERO, None)
            os.environ.pop(ENV_PLAY_AI, None)


def base_record(kind: str, index: int, seed: int, horizon: int, ctx: Mapping[str, Any]) -> Dict:
    return {
        "schema_version": schema.SCHEMA,
        "study_id": schema.STUDY_ID,
        "authority": schema.AUTHORITY,
        "intent_sha256": ctx["intent_sha256"],
        "episode_index": int(index),
        "episode_id": schema.episode_id(kind, index),
        "kind": kind,
        "world_seed": int(seed),
        "status": schema.STATUS_ERROR,
        "error": None,
        "horizon_frames": int(horizon),
        "frames_stepped": 0,
        "ended_by": None,
        "wall_seconds": None,
        "checkpoint": {"path": str(ctx["checkpoint"]), "sha256": ctx["checkpoint_sha256"]},
        "wrapper": None,
        "served": None,
        "veto": None,
        "decision_frames": None,
        "episode": None,
    }


def _finish(record: Dict, sess: Any, before: Mapping, counter: DecisionCounter) -> None:
    state = sess.safety_veto_state()
    record["wrapper"] = state["wrapper"]
    record["checkpoint"]["session_sha256"] = state["checkpoint_sha256"]
    record["served"] = served_identity(sess)
    record["veto"] = veto_block(sess, before)
    per_snake = {str(sid): n for sid, n in sorted(counter.counts.items())}
    record["decision_frames"] = {"per_snake": per_snake, "total": sum(per_snake.values())}


def run_watch_episode(index: int, seed: int, horizon: int, ctx: Mapping[str, Any]) -> Dict:
    """The served Watch configuration: legacy all-respawn Watch, wrapper on the hero."""
    from web.backend.session import GameSession

    record = base_record(schema.KIND_WATCH, index, seed, horizon, ctx)
    started = time.monotonic()
    try:
        with veto_env(watch_hero=True, play_ai=False):
            set_world_seed(seed)
            sess = GameSession(checkpoint=str(ctx["checkpoint"]))
        state = sess.safety_veto_state()
        before = state["counters"]
        counter = DecisionCounter(state["wrapped_snake_ids"])
        hero = next(s for s in sess.game.snakes if int(s.id) == int(sess.hero_id))
        deaths, alive_frames, prev = 0, 0, bool(hero.is_alive)
        for _ in range(horizon):
            counter.step(sess)
            record["frames_stepped"] += 1
            alive = bool(hero.is_alive)
            deaths += int(prev and not alive)
            alive_frames += int(alive)
            prev = alive
        record["ended_by"] = schema.ENDED_HORIZON
        record["episode"] = {
            "hero_id": int(hero.id),
            "hero_deaths": deaths,
            "hero_alive_frames": alive_frames,
            "hero_final_length": int(len(hero.segments)),
            "game_frame": int(sess.game.frame),
        }
        _finish(record, sess, before, counter)
        record["status"] = schema.STATUS_COMPLETE
    except Exception as exc:  # recorded, never swallowed silently
        record["error"] = f"{type(exc).__name__}: {exc}"
    record["wall_seconds"] = time.monotonic() - started
    return record


def play_session(ctx: Mapping[str, Any]) -> Any:
    """One served session switched into Play through the dispatch path."""
    from web.backend.session import GameSession

    with veto_env(watch_hero=False, play_ai=True):
        sess = GameSession(checkpoint=str(ctx["checkpoint"]))
        control(sess, "set_mode", "play")
    require(sess.mode == "play", f"session did not enter play mode: {sess.last_error}")
    return sess


def run_play_episode(sess: Any, index: int, seed: int, horizon: int, ctx: Mapping) -> Dict:
    """One scored Play run: new_game, scripted human input, until death or horizon."""
    record = base_record(schema.KIND_PLAY, index, seed, horizon, ctx)
    started = time.monotonic()
    try:
        set_world_seed(seed)
        control(sess, "new_game")
        state = sess.safety_veto_state()
        before = state["counters"]
        counter = DecisionCounter(state["wrapped_snake_ids"])
        human = sess._find_human()
        require(human is not None, "play session has no human snake")
        stand_in = ScriptedHuman(seed)
        inputs = 0
        while not sess.run_started:
            name = stand_in.choose(human, sess.game, must_turn=True)
            require(name is not None and inputs < 8, "the stand-in could not start the run")
            control(sess, "human_input", name)
            inputs += 1
        while record["frames_stepped"] < horizon and not sess.run_over:
            name = stand_in.choose(human, sess.game, must_turn=False)
            if name is not None:
                control(sess, "human_input", name)
                inputs += 1
            counter.step(sess)
            record["frames_stepped"] += 1
        record["ended_by"] = schema.ENDED_HUMAN_DEATH if sess.run_over else schema.ENDED_HORIZON
        record["episode"] = {
            "human_id": int(human.id),
            "inputs_sent": inputs,
            "run_frames": int(sess.run_frames),
            "run_over": bool(sess.run_over),
            "last_run": sess.last_run,
            "game_frame": int(sess.game.frame),
        }
        _finish(record, sess, before, counter)
        record["status"] = schema.STATUS_COMPLETE
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    record["wall_seconds"] = time.monotonic() - started
    return record


# ---------------------------------------------------------------- parity probes


def frame_signature(game: Any) -> List[Any]:
    """Everything observable about one frame that both paths must agree on."""
    return [
        int(game.frame),
        [
            [int(s.id), [int(v) for v in s.head], len(s.segments), bool(s.is_alive)]
            + [bool(s.is_boosting)]
            for s in game.snakes
        ],
        len(game.food_manager.food),
    ]


@contextlib.contextmanager
def parity_config() -> Iterator[None]:
    """Point the session's 61-D config at the gate's deployment config (probe only)."""
    from web.backend import session as web_session

    saved = web_session.CONFIG_MECHANICS_V2
    saved_env = os.environ.pop("SNAKE_MECHANICS_V2", None)
    web_session.CONFIG_MECHANICS_V2 = str(PARITY_CONFIG)
    try:
        yield
    finally:
        web_session.CONFIG_MECHANICS_V2 = saved
        if saved_env is not None:
            os.environ["SNAKE_MECHANICS_V2"] = saved_env


def _session_side(seed: int, horizon: int, ctx: Mapping[str, Any]) -> Dict[str, Any]:
    from web.backend.session import GameSession

    with parity_config(), veto_env(watch_hero=True, play_ai=False):
        set_world_seed(seed)
        sess = GameSession(checkpoint=str(ctx["checkpoint"]))
    hero = sess.game.snakes[0]
    hero.auto_respawn = False  # the profile's hero_terminal; legacy served Watch respawns
    state = sess.safety_veto_state()
    counter = DecisionCounter(state["wrapped_snake_ids"])
    trace = []
    for _ in range(horizon):
        counter.step(sess)
        trace.append(frame_signature(sess.game))
    state = sess.safety_veto_state()
    return {
        "trace": trace,
        "config_sha256": schema.sha256_file(Path(sess.config_path)),
        "wrapped_snake_ids": state["wrapped_snake_ids"],
        "counters": state["counters"][str(hero.id)],
        "decision_frames": counter.counts[int(hero.id)],
    }


def _rollout_side(seed: int, horizon: int, ctx: Mapping[str, Any]) -> Dict[str, Any]:
    import src.scripts.tournament_eval as te
    from src.core.config_loader import load_and_initialize_config
    from src.core.game_config import GameConfig

    load_and_initialize_config(str(PARITY_CONFIG))
    profile = te.evaluation_profile_for_name(PROFILE_NAME, HORIZON)
    trace: List[Any] = []
    vetoes: List[Any] = []
    probes_cls, install = te.BehaviorProbes, te._install_hero_safety_veto

    class TracingProbes(probes_cls):  # observes only; BehaviorProbes semantics unchanged
        def observe(self, gs: Any) -> Any:
            trace.append(frame_signature(gs))
            return super().observe(gs)

    def capture(hero: Any, spec: Any) -> Any:
        vetoes.append(install(hero, spec))
        return vetoes[-1]

    hero_spec = ("checkpoint", str(ctx["checkpoint"]))
    te.BehaviorProbes, te._install_hero_safety_veto = TracingProbes, capture
    record = None
    try:
        record = te.rollout(
            hero_spec,
            [hero_spec] * (int(GameConfig.NUM_SNAKES) - 1),
            horizon,
            seed,
            profile=profile,
            hero_safety_veto=True,
        )
    except ValueError as exc:  # a smoke horizon cannot complete the H5000 accumulator
        if horizon >= HORIZON or "incomplete evaluation" not in str(exc):
            raise
    finally:
        te.BehaviorProbes, te._install_hero_safety_veto = probes_cls, install
    summary = None
    if record is not None:
        keys = ("mass_integral", "max_mass", "deaths", "kills", "survival_fraction")
        summary = {key: record[key] for key in keys}
        summary["evaluation_profile_digest"] = record["evaluation_profile_digest"]
    return {
        "trace": trace,
        "counters": vetoes[0].counters.to_dict(),
        "record_complete": record is not None,
        "record": summary,
    }


def run_parity_probe(index: int, seed: int, horizon: int, ctx: Mapping[str, Any]) -> Dict:
    record = {
        "schema_version": schema.SCHEMA,
        "study_id": schema.STUDY_ID,
        "intent_sha256": ctx["intent_sha256"],
        "probe_index": int(index),
        "probe_id": schema.parity_id(index),
        "world_seed": int(seed),
        "status": schema.STATUS_ERROR,
        "error": None,
        "horizon_frames": int(horizon),
        "config": {"path": str(PARITY_CONFIG), "profile": PROFILE_NAME},
        "session": None,
        "rollout": None,
        "frames_compared": 0,
        "first_divergence_frame": None,
        "trace_sha256_equal": False,
        "veto_counters_equal": False,
        "wall_seconds": None,
    }
    started = time.monotonic()
    try:
        left = _session_side(seed, horizon, ctx)
        right = _rollout_side(seed, horizon, ctx)
        pairs = list(zip(left.pop("trace"), right.pop("trace")))
        left["trace_sha256"] = schema.canonical_sha256([a for a, _ in pairs])
        right["trace_sha256"] = schema.canonical_sha256([b for _, b in pairs])
        record["session"], record["rollout"] = left, right
        record["frames_compared"] = len(pairs)
        record["first_divergence_frame"] = next(
            (i for i, (a, b) in enumerate(pairs) if a != b), None
        )
        record["trace_sha256_equal"] = left["trace_sha256"] == right["trace_sha256"]
        record["veto_counters_equal"] = left["counters"] == right["counters"]
        record["status"] = schema.STATUS_COMPLETE
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    record["wall_seconds"] = time.monotonic() - started
    return record


# ---------------------------------------------------------------- run


def check_output_root(out: Path, smoke: bool) -> None:
    require(not out.exists(), f"--out {out} already exists (create-only)")
    if smoke:
        require(ARTIFACT_ROOT not in out.parents, "a smoke never writes under the artifact root")


def build_intent(args: argparse.Namespace, checkpoint_sha256: str) -> Dict[str, Any]:
    from web.backend.safety_veto_serving import wrapper_identity

    counts = SMOKE_COUNTS if args.smoke else COUNTS
    seeds = world_seeds(SMOKE_SEED_DOMAIN if args.smoke else SEED_DOMAIN, counts)
    report = seed_report(seeds)
    require(report["disjoint"], f"serving seeds overlap earlier banks: {report}")
    git = git_state()
    protocol_sha256 = schema.sha256_file(PROTOCOL)
    if not args.smoke:  # the audit enforces both again; fail before any compute
        require(not git["dirty"], "a non-smoke run needs a clean git tree")
        require(protocol_sha256 == PROTOCOL_SHA256, "protocol.md differs from its pinned sha256")
    return {
        "schema_version": schema.SCHEMA,
        "study_id": schema.STUDY_ID,
        "authority": schema.AUTHORITY,
        "smoke": bool(args.smoke),
        "git": git,
        "protocol_sha256": protocol_sha256,
        "checkpoint": {"path": str(args.checkpoint), "sha256": checkpoint_sha256},
        "expected": {
            "checkpoint_sha256": CHAMPION_SHA256,
            "wrapper_source_sha256": WRAPPER_SOURCE_SHA256,
            "served_config_basename": "mechanics_v2.yaml",
            "served_config_sha256": SERVED_CONFIG_SHA256,
            "parity_config_sha256": PARITY_CONFIG_SHA256,
        },
        "wrapper_identity": wrapper_identity(),
        "horizon_frames": SMOKE_HORIZON if args.smoke else HORIZON,
        "parity_horizon_frames": SMOKE_PARITY_HORIZON if args.smoke else HORIZON,
        "counts": dict(counts),
        "seed_recipe": {
            "domain": SMOKE_SEED_DOMAIN if args.smoke else SEED_DOMAIN,
            "rule": "uint32 big-endian of sha256(f'{domain}|{purpose}|{i}')[:4]",
        },
        "seeds": seeds,
        "seed_report": report,
        "criteria": ["S1", "S2", "S3", "S4", "S5", "S6"],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=CHAMPION_PATH)
    parser.add_argument("--smoke", action="store_true", help="1 watch + 1 play x 500 frames")
    args = parser.parse_args(argv)
    out = args.out.resolve()
    check_output_root(out, args.smoke)

    import torch

    torch.set_num_threads(1)  # as web/backend/app.py and web/serve.py pin it
    os.environ.setdefault("SNAKE_DQN_DEVICE", "cpu")
    checkpoint_sha256 = schema.sha256_file(args.checkpoint)
    require(checkpoint_sha256 == CHAMPION_SHA256, "checkpoint is not the pinned champion")
    started = time.monotonic()
    intent = build_intent(args, checkpoint_sha256)
    intent_sha = write_new_json(out / schema.INTENT, intent)
    ctx = {
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": checkpoint_sha256,
        "intent_sha256": intent_sha,
    }
    horizon, parity_horizon = intent["horizon_frames"], intent["parity_horizon_frames"]

    files: Dict[str, str] = {}
    files[schema.DEFAULT_CHECK] = write_new_json(
        out / schema.DEFAULT_CHECK, {"intent_sha256": intent_sha, **default_check(args.checkpoint)}
    )
    records: List[Dict[str, Any]] = []

    def keep(record: Dict[str, Any]) -> None:
        # Written as each episode ends, so a crash keeps every finished record.
        records.append(record)
        rel = f"{schema.RECORDS_DIR}/{record['episode_id']}.json"
        files[rel] = write_new_json(out / rel, record)

    for i, seed in enumerate(intent["seeds"]["watch"]):
        keep(run_watch_episode(i, seed, horizon, ctx))
    try:
        sess, play_error = play_session(ctx), None
    except Exception as exc:  # every Play record then carries the build error
        sess, play_error = None, f"{type(exc).__name__}: {exc}"
    for i, seed in enumerate(intent["seeds"]["play"]):
        if sess is None:
            record = base_record(schema.KIND_PLAY, i, seed, horizon, ctx)
            record["error"] = play_error
            keep(record)
        else:
            keep(run_play_episode(sess, i, seed, horizon, ctx))
    for i, seed in enumerate(intent["seeds"]["parity"]):
        probe = run_parity_probe(i, seed, parity_horizon, ctx)
        rel = f"{schema.PARITY_DIR}/{probe['probe_id']}.json"
        files[rel] = write_new_json(out / rel, probe)

    failures = [r["episode_id"] for r in records if r["status"] != schema.STATUS_COMPLETE]
    receipt = {
        "schema_version": schema.SCHEMA,
        "study_id": schema.STUDY_ID,
        "intent_sha256": intent_sha,
        "files": files,
        "episodes": len(records),
        "frames": sum(r["frames_stepped"] for r in records),
        "failures": failures,
        "wall_seconds": time.monotonic() - started,
        "serving_path_qualified": None,
        "note": "serving_path_qualified is set only by serving_audit.py's verdict",
    }
    write_new_json(out / schema.RECEIPT, receipt)
    print(f"wrote {out} ({len(records)} episodes, {len(failures)} failures)")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
