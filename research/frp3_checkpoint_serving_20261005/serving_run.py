"""Web serving run for the FRP-v3 s12 checkpoint swap (M3@60000) + released v8 veto on Watch.

Drives the REAL web backend headlessly, as the v2/v5/v7/v8 serving runs did (the v2 lane's
stepping, stand-in, decision counting, dispatch and served-identity helpers and the v8 lane's
veto block, diagnostics and seed-report helpers are imported unchanged): ``GameSession.step()``
then ``snapshot()`` (``Hub.engine_loop`` order), every control through
``web.backend.app._apply_control``. No server, no network listener.

What is new against the v8 lane: every session is built the way ``web/backend/app.py`` builds
it, ``GameSession()`` with NO explicit checkpoint, under the swap's release environment
``SNAKE_SERVE_CHECKPOINT=frp3-s12`` (the three veto variables unset: Watch hero on, released
variant v8, Play AI off). So the checkpoint comes from the pinned registry
(``web/backend/served_checkpoint.py``), which serves it only when its strict-receipt pin is
filled and its bytes hash to the pinned sha256, and the v8 wrapper binds it only through that
pin. Episodes (pre-registered in ``protocol.md``):

* Watch episodes as served: all-respawn Watch, the v8 wrapper on the slot-0 hero only;
* Play episodes: a scripted stand-in steers the human; no snake is wrapped;
* parity probes: session stepping on the gate's deployment config with a terminal hero,
  compared frame by frame (cell space) with a SIMD rollout of the same identity:
  ``run_simd_eval(frp3, 5 x frp3, ..., vector61=True, hero_safety_veto="v8",
  hero_safety_veto_lambda=8.0, vector61_forward="rowwise")`` on the same world seed.

``--smoke`` runs before the owner fills the strict pin: it serves the swap under an
in-process placeholder pin (recorded in the intent; a smoke never qualifies). A real run
refuses unless the repo's pin is filled. Output is create-only.
"""

from __future__ import annotations

import os

# One shared CPU slot = one thread: set before torch/numpy load anywhere in the process.
for _var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")
os.environ.setdefault("SNAKE_DQN_DEVICE", "cpu")

import argparse  # noqa: E402
import contextlib  # noqa: E402
import dataclasses  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.apex_veto_serving_20261001 import serving_run as v2run  # noqa: E402
from research.apex_veto_v8_serving_20261003 import serving_run as v8run  # noqa: E402


def _load_schema() -> Any:
    spec = importlib.util.spec_from_file_location(
        "frp3_checkpoint_serving_schema", HERE / "schema.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


schema = _load_schema()

PROTOCOL = HERE / "protocol.md"
# sha256 of protocol.md as pre-registered for the real run (the audit pins it too).
PROTOCOL_SHA256 = "9163461767343f94a52e9a64f83f201b68ed3b4b322400854284fffe868727c6"
CHECKPOINT_NAME = "frp3-s12"
CHECKPOINT_SHA256 = "eec144bf92509a42664e2d650b8741f7d7774b86d7b6601d012bdd9425dd3723"
CHECKPOINT_FILENAME = "frp3_m3_s12_u60000_20261005.pth"
CHECKPOINT_ARTIFACT_PATH = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/frp-v3-20261005/train/arm-M3/seed-12/"
    "checkpoints/apex_mark_u60000.pth"
)
CHAMPION_PATH = v2run.CHAMPION_PATH
CHAMPION_SHA256 = v2run.CHAMPION_SHA256
SERVED_CONFIG_SHA256 = v2run.SERVED_CONFIG_SHA256
PARITY_CONFIG = v2run.PARITY_CONFIG
PARITY_CONFIG_SHA256 = v2run.PARITY_CONFIG_SHA256
PROFILE_NAME = v2run.PROFILE_NAME
PROFILE_DIGEST = "d396d3ed93e3a264d050674887eb47e59de67d3c6696c2c41fa5f8b145ea0e8b"
V8_LAMBDA = v8run.V8_LAMBDA
V8_METHOD = v8run.V8_METHOD
V8_SOURCE_SHA256S = dict(v8run.V8_SOURCE_SHA256S)
V8_HARNESS = REPO / "research/apex_veto_v8_serving_20261003/serving_run.py"
V2_HARNESS = REPO / "research/apex_veto_serving_20261001/serving_run.py"
VECTOR61_FORWARD = "rowwise"  # bit-exact batch-1 forwards

HORIZON = 5000
COUNTS = {"watch": 25, "play": 25, "parity": 2}
SMOKE_HORIZON = 500
SMOKE_PARITY_HORIZON = 200
SMOKE_COUNTS = {"watch": 1, "play": 1, "parity": 1}
SEED_DOMAIN = "frp3-checkpoint-web-serving-v1"
SMOKE_SEED_DOMAIN = "frp3-checkpoint-web-serving-smoke-v1"
SMALL_INTEGER_SEEDS = 1000
ARTIFACT_ROOT = v2run.ARTIFACT_ROOT
# Every seed FRP-v3 had seen or banked when it trained (its namespace preflight's output).
FRP3_EXCLUSIONS = ARTIFACT_ROOT / "frp-v3-20261005/namespaces/training_exclusions.json"
FRP3_EXCLUSIONS_SHA256 = "4b0f33775544e36e2ccbe6a5d8ab35b81de40565c4328dc8abddd3cea4bc1edf"
# The v8 web lane's own domains (its report covers the earlier ones, not itself).
V8_SERVING_DOMAINS = ("apex-veto-v8-web-serving-v1", "apex-veto-v8-web-serving-smoke-v1")
WEB_PURPOSES = ("watch", "play", "parity")
SLOT_POOL = 3  # the Tier-1/dev default (compute policy); strict gates keep 2
MAX_UNSAFE_PAUSE_SECONDS = 3600.0

# The swap's release environment every served build reads (None = unset).
RELEASE_ENV = {
    "SNAKE_SERVE_CHECKPOINT": CHECKPOINT_NAME,
    "SNAKE_SERVE_VETO_WATCH_HERO": None,
    "SNAKE_SERVE_VETO_PLAY_AI": None,
    "SNAKE_SERVE_VETO_VARIANT": None,
}
# The in-process placeholder pin a smoke serves under (never written to the repo file).
SMOKE_PIN_ROOT = "/smoke-placeholder-not-a-strict-run"

require = v2run.require
write_new_json = v2run.write_new_json
world_seeds = v2run.world_seeds
git_state = v2run.git_state
DecisionCounter = v2run.DecisionCounter
ScriptedHuman = v2run.ScriptedHuman
control = v2run.control
set_world_seed = v2run.set_world_seed
served_identity = v2run.served_identity
snakes_with_veto = v8run.snakes_with_veto
int_diagnostics = v8run.int_diagnostics
veto_block = v8run.veto_block


# ---------------------------------------------------------------- pins and env


def pin_entry(pins: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    return (pins.get("checkpoints") or {}).get(CHECKPOINT_NAME)


def smoke_pins() -> Dict[str, Any]:
    """The repo pins document with the swap's strict receipt replaced by a placeholder."""
    from web.backend import served_checkpoint as registry

    doc = json.loads(json.dumps(registry.load_pins()))
    entry = doc["checkpoints"][CHECKPOINT_NAME]
    entry["strict_receipt"] = {
        "verdict": "STRICT_PASS",
        "root": SMOKE_PIN_ROOT,
        **{key: {"path": f"{key}.json", "sha256": "0" * 64} for key in registry.STRICT_FILES},
    }
    return doc


def unfilled_pins() -> Dict[str, Any]:
    """The repo pins document with the swap's strict receipt reset to the placeholder."""
    from web.backend import served_checkpoint as registry

    doc = json.loads(json.dumps(registry.load_pins()))
    doc["checkpoints"][CHECKPOINT_NAME]["strict_receipt"] = None
    return doc


@contextlib.contextmanager
def pins_override(doc: Optional[Mapping[str, Any]]) -> Iterator[None]:
    """Point the registry at ``doc`` (written to a temp file) for this block; None = repo."""
    if doc is None:
        yield
        return
    from web.backend import served_checkpoint as registry

    saved = registry.PINS_PATH
    with tempfile.TemporaryDirectory(prefix="frp3-serving-pins-") as tmp:
        path = Path(tmp) / "pins.json"
        path.write_text(json.dumps(doc, sort_keys=True), encoding="utf-8")
        registry.PINS_PATH = str(path)
        try:
            yield
        finally:
            registry.PINS_PATH = saved


@contextlib.contextmanager
def serving_env(values: Mapping[str, Optional[str]]) -> Iterator[None]:
    """Set the four serving env vars exactly as ``values`` says (None = unset)."""
    saved = {key: os.environ.pop(key) for key in schema.ENV_KEYS if key in os.environ}
    try:
        for key in schema.ENV_KEYS:
            if values.get(key) is not None:
                os.environ[key] = str(values[key])
        yield
    finally:
        for key in schema.ENV_KEYS:
            os.environ.pop(key, None)
        os.environ.update(saved)


def released_session() -> Any:
    """``GameSession()`` exactly as ``web/backend/app.py`` builds it, under the release env."""
    from web.backend.session import GameSession

    with serving_env(RELEASE_ENV):
        return GameSession()


# ---------------------------------------------------------------- seeds


def _ints_under(value: Any, out: set, key_ok: bool) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _ints_under(item, out, key_ok or "seed" in str(key).lower() or key == "banks")
    elif isinstance(value, list):
        for item in value:
            _ints_under(item, out, key_ok)
    elif key_ok and isinstance(value, int) and not isinstance(value, bool):
        out.add(int(value))


def strict_observed_seeds(pin: Mapping[str, Any]) -> Dict[str, List[int]]:
    """Every integer under a seed-named key (or ``banks``) in the swap's strict intent and in
    every ``rosters.json`` under its root (refuses when the intent is missing or has none)."""
    receipt = pin["strict_receipt"]
    root = Path(receipt["root"])
    intent = root / receipt["intent"]["path"]
    require(intent.is_file(), f"strict intent missing: {intent}")
    out: Dict[str, List[int]] = {}
    for path in [intent] + sorted(root.rglob("rosters.json")):
        seeds: set = set()
        _ints_under(json.loads(path.read_text(encoding="utf-8")), seeds, False)
        if path == intent:
            require(bool(seeds), f"strict intent has no seeds: {intent}")
        out[f"observed:{path}"] = sorted(seeds)
    return out


def strict_domains(pin: Mapping[str, Any]) -> List[str]:
    """Every string under a key containing ``domain`` or ``namespace`` in the strict intent
    (its bank domains; their first 1000 ``worlds`` seeds are excluded, as the v8 lane did)."""
    receipt = pin["strict_receipt"]
    doc = json.loads((Path(receipt["root"]) / receipt["intent"]["path"]).read_text("utf-8"))
    out: set = set()

    def walk(value: Any, key_ok: bool) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                k = str(key).lower()
                walk(item, key_ok or "domain" in k or "namespace" in k)
        elif isinstance(value, list):
            for item in value:
                walk(item, key_ok)
        elif key_ok and isinstance(value, str) and value and "/" not in value:
            out.add(value)

    walk(doc, False)
    return sorted(out)


def frp3_exclusions() -> List[int]:
    require(FRP3_EXCLUSIONS.is_file(), f"FRP-v3 training exclusions missing: {FRP3_EXCLUSIONS}")
    require(
        schema.sha256_file(FRP3_EXCLUSIONS) == FRP3_EXCLUSIONS_SHA256,
        "FRP-v3 training exclusions differ from the pinned sha256",
    )
    return [int(s) for s in json.loads(FRP3_EXCLUSIONS.read_text(encoding="utf-8"))]


def seed_report(
    seeds: Mapping[str, Sequence[int]], pin: Optional[Mapping[str, Any]] = None
) -> Dict[str, Any]:
    """Fail-closed freshness check (in-code; no registry).

    Union of: the v8 web lane's report (which contains the v7/v5/v2 lanes', the v8 strict
    gate's exclusions, banks and observed seeds); the first 1000 seeds of every web purpose
    of the v8 web lane's own domains; FRP-v3's written training exclusion file (every seed
    its namespace preflight found in every artifact root, plus its own banks); seeds 0..999;
    and, when ``pin`` carries a filled strict receipt, every seed its strict run saved.
    """
    from research.apex_safety_20260926 import dev_screen

    flat = [seed for values in seeds.values() for seed in values]
    v8_report = v8run.seed_report(seeds)
    banks: Dict[str, List[int]] = {}
    for domain in V8_SERVING_DOMAINS:
        for purpose in WEB_PURPOSES:
            banks[f"{domain}/{purpose}[0:{SMALL_INTEGER_SEEDS}]"] = [
                dev_screen.uint32_seed(domain, purpose, i) for i in range(SMALL_INTEGER_SEEDS)
            ]
    banks[f"frp3-training-exclusions:{FRP3_EXCLUSIONS}"] = frp3_exclusions()
    strict_checked = bool(pin and isinstance(pin.get("strict_receipt"), Mapping))
    if strict_checked and pin["strict_receipt"].get("root") != SMOKE_PIN_ROOT:
        banks.update(strict_observed_seeds(pin))
        for domain in strict_domains(pin):
            banks[f"{domain}/worlds[0:{SMALL_INTEGER_SEEDS}] (strict bank domain)"] = [
                dev_screen.uint32_seed(domain, "worlds", i) for i in range(SMALL_INTEGER_SEEDS)
            ]
    else:
        strict_checked = False
    earlier: set = set(range(SMALL_INTEGER_SEEDS))
    for values in banks.values():
        earlier |= set(values)
    overlap = sorted(earlier & set(flat))
    unique = len(set(flat)) == len(flat)
    return {
        "unique": unique,
        "v8_serving_report_disjoint": bool(v8_report["disjoint"]),
        "v8_serving_report_overlap": v8_report["v8_earlier_overlap"],
        "frp3_earlier_overlap": overlap,
        "strict_observed_checked": strict_checked,
        "checked_against": v8_report["checked_against"] + sorted(banks),
        "registry_checked": False,
        "registry_note": "governance namespace registry not implemented; in-code check only",
        "disjoint": unique and bool(v8_report["disjoint"]) and not overlap,
    }


# ---------------------------------------------------------------- S6 default check

V8_METHOD_SERVED = V8_METHOD
V7_METHOD = v8run.V7_METHOD
CHECKPOINT_ENV = "SNAKE_SERVE_CHECKPOINT"
# case -> (env, pins override kind or None)
DEFAULT_CASES: Dict[str, Dict[str, str]] = {
    "empty_env": {},
    "checkpoint_champion": {CHECKPOINT_ENV: "champion"},
    "checkpoint_unknown": {CHECKPOINT_ENV: "frp3-s99"},
    "frp3_pin_unfilled": {CHECKPOINT_ENV: CHECKPOINT_NAME},
    "frp3_rollback_variant_v7": {
        CHECKPOINT_ENV: CHECKPOINT_NAME,
        "SNAKE_SERVE_VETO_VARIANT": "v7",
    },
    "frp3_watch_off": {CHECKPOINT_ENV: CHECKPOINT_NAME, "SNAKE_SERVE_VETO_WATCH_HERO": "0"},
}
# Expected served checkpoint (name, sha) and Watch wrapper (variant, method) or None (bare).
DEFAULT_CASE_EXPECT = {
    "empty_env": ("champion", CHAMPION_SHA256, ("v8", V8_METHOD)),
    "checkpoint_champion": ("champion", CHAMPION_SHA256, ("v8", V8_METHOD)),
    "checkpoint_unknown": ("champion", CHAMPION_SHA256, ("v8", V8_METHOD)),
    "frp3_pin_unfilled": ("champion", CHAMPION_SHA256, ("v8", V8_METHOD)),
    "frp3_rollback_variant_v7": (CHECKPOINT_NAME, CHECKPOINT_SHA256, None),
    "frp3_watch_off": (CHECKPOINT_NAME, CHECKPOINT_SHA256, None),
}
# Cases whose served_checkpoint.reason must be set (the request was refused).
REFUSED_CASES = ("checkpoint_unknown", "frp3_pin_unfilled")


def default_check(run_pins: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """S6: the released defaults, the fail-closed refusals and the rollbacks, read back from
    real ``GameSession()`` builds. ``run_pins`` is the pins override the run serves under
    (None = the repo file; a smoke passes its placeholder)."""
    from web.backend import served_checkpoint as registry
    from web.backend import session as web_session
    from web.backend.safety_veto_serving import ServingVetoFlags

    flags = ServingVetoFlags.from_env({})
    out: Dict[str, Any] = {
        "flags_from_empty_env": flags.to_dict(),
        "variant_from_empty_env": flags.variant,
        "checkpoint_released_default": registry.CHECKPOINT_RELEASED_DEFAULT,
        "default_checkpoint_basename": os.path.basename(web_session.DEFAULT_CHECKPOINT),
        "served_config_basename": os.path.basename(web_session._config_for(61)),
        "served_config_sha256": schema.sha256_file(Path(web_session._config_for(61))),
    }
    builds: Dict[str, Any] = {}
    for name, env in DEFAULT_CASES.items():
        pins = unfilled_pins() if name == "frp3_pin_unfilled" else run_pins
        with pins_override(pins), serving_env(env):
            sess = web_session.GameSession()
            modes = {}
            for mode in ("watch", "play"):
                if mode != sess.mode:
                    control(sess, "set_mode", mode)
                state = sess.safety_veto_state()
                modes[mode] = {
                    "mode": sess.mode,
                    "active": bool(state["active"]),
                    "variant": state["variant"],
                    "method": (state["wrapper"] or {}).get("method"),
                    "reason": state["reason"],
                    "checkpoint_sha256": state["checkpoint_sha256"],
                    "snakes_with_veto": snakes_with_veto(sess),
                    "hero_id": int(sess.hero_id),
                }
        builds[name] = {
            "env": dict(env),
            "pins": "unfilled" if name == "frp3_pin_unfilled" else "run",
            "served_checkpoint": sess.served_checkpoint,
            "modes": modes,
        }
    out["builds"] = builds
    out["pass"] = bool(default_check_failures(out) == [])
    return out


def default_check_failures(doc: Mapping[str, Any]) -> List[str]:
    """Producer-side S6 rules (the audit re-implements them)."""
    out = []
    if doc["flags_from_empty_env"] != {"watch_hero": True, "play_ai": False}:
        out.append("empty env flags are not the released {watch_hero: on, play_ai: off}")
    if doc["variant_from_empty_env"] != "v8":
        out.append("empty env variant is not the released v8")
    if doc["checkpoint_released_default"] != "champion":
        out.append("released checkpoint default is not the champion")
    if doc["default_checkpoint_basename"] != CHAMPION_PATH.name:
        out.append("served default checkpoint path changed")
    if (doc["served_config_basename"], doc["served_config_sha256"]) != (
        "mechanics_v2.yaml",
        SERVED_CONFIG_SHA256,
    ):
        out.append("served default config changed")
    for name, (ckpt, sha, wrapper) in DEFAULT_CASE_EXPECT.items():
        build = doc["builds"][name]
        served = build["served_checkpoint"] or {}
        if served.get("name") != ckpt:
            out.append(f"{name}: served {served.get('name')!r}, expected {ckpt!r}")
        if (served.get("reason") is not None) != (name in REFUSED_CASES):
            out.append(f"{name}: served_checkpoint.reason {served.get('reason')!r}")
        watch, play = build["modes"]["watch"], build["modes"]["play"]
        if watch["checkpoint_sha256"] != sha or play["checkpoint_sha256"] != sha:
            out.append(f"{name}: session checkpoint sha is not {ckpt}'s")
        if wrapper is None:
            if watch["active"] or watch["snakes_with_veto"]:
                out.append(f"{name}: Watch is wrapped")
        elif not (
            watch["active"]
            and (watch["variant"], watch["method"]) == wrapper
            and watch["snakes_with_veto"] == [watch["hero_id"]]
        ):
            out.append(f"{name}: Watch is not the {wrapper[0]} hero wrapper")
        if play["active"] or play["snakes_with_veto"]:
            out.append(f"{name}: Play is wrapped")
    return out


# ---------------------------------------------------------------- episodes


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
        "env": dict(RELEASE_ENV),
        "checkpoint": {"path": str(ctx["checkpoint"]), "sha256": ctx["checkpoint_sha256"]},
        "served_checkpoint": None,
        "wrapper": None,
        "served": None,
        "veto": None,
        "snakes_with_veto": None,
        "decision_frames": None,
        "episode": None,
    }


def _finish(record: Dict, sess: Any, before: Mapping, counter: Any) -> None:
    state = sess.safety_veto_state()
    record["served_checkpoint"] = sess.served_checkpoint
    record["wrapper"] = state["wrapper"]
    record["checkpoint"]["session_sha256"] = state["checkpoint_sha256"]
    record["served"] = served_identity(sess)
    record["veto"] = veto_block(sess, before)
    record["snakes_with_veto"] = snakes_with_veto(sess)
    per_snake = {str(sid): n for sid, n in sorted(counter.counts.items())}
    record["decision_frames"] = {"per_snake": per_snake, "total": sum(per_snake.values())}


def run_watch_episode(index: int, seed: int, horizon: int, ctx: Mapping[str, Any]) -> Dict:
    """The served Watch configuration: all-respawn Watch, the v8 wrapper on the hero."""
    record = base_record(schema.KIND_WATCH, index, seed, horizon, ctx)
    started = time.monotonic()
    try:
        ctx["guard"]()
        set_world_seed(seed)
        sess = released_session()
        state = sess.safety_veto_state()
        before = state["counters"]
        require(
            all(d["decisions"] == 0 for d in state["diagnostics"].values()),
            "a fresh Watch build carried v8 diagnostics",
        )
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
    """One served session (release env) switched into Play through the dispatch path."""
    sess = released_session()
    with serving_env(RELEASE_ENV):
        control(sess, "set_mode", "play")
    require(sess.mode == "play", f"session did not enter play mode: {sess.last_error}")
    return sess


def run_play_episode(sess: Any, index: int, seed: int, horizon: int, ctx: Mapping) -> Dict:
    """One scored Play run (new_game, scripted human input) that must stay unwrapped."""
    record = base_record(schema.KIND_PLAY, index, seed, horizon, ctx)
    started = time.monotonic()
    try:
        ctx["guard"]()
        set_world_seed(seed)
        with serving_env(RELEASE_ENV):
            control(sess, "new_game")
        state = sess.safety_veto_state()
        before = state["counters"]
        require(snakes_with_veto(sess) == [], "a Play snake is wrapped at episode start")
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


def _cell(position: Sequence[Any], size: int) -> List[int]:
    from src.core.mechanics_constants import cell_index

    col, row = cell_index((int(position[0]), int(position[1])), int(size))
    return [int(col), int(row)]


def live_frame_signature(game: Any) -> List[Any]:
    """One live frame in cell space: frame, per slot [alive, head cell, logical length]
    (None for a dead slot), food count and the sha256 of the sorted food cells."""
    size = int(game.snakes[0].segment_size)
    snakes = []
    for slot, snake in enumerate(game.snakes):
        alive = bool(snake.is_alive)
        snakes.append(
            [slot, alive]
            + ([_cell(snake.head, size), int(snake.length)] if alive else [None, None])
        )
    food = sorted(_cell(f, size) for f in game.food_manager.food)
    return [int(game.frame), snakes, len(food), schema.canonical_sha256(food)]


def simd_frame_signature(sim: Any, env: int = 0) -> List[Any]:
    """The same signature read from a BatchSim env (cells already)."""
    heads, alive, lengths = sim.get_heads()[env], sim.get_alive()[env], sim.get_lengths()[env]
    snakes = []
    for slot in range(int(sim.S)):
        is_alive = bool(alive[slot])
        snakes.append(
            [slot, is_alive]
            + (
                [[int(heads[slot][0]), int(heads[slot][1])], int(lengths[slot])]
                if is_alive
                else [None, None]
            )
        )
    food = sorted([int(c), int(r)] for c, r in sim.get_food(env))
    return [int(sim.frame[env]), snakes, len(food), schema.canonical_sha256(food)]


def _session_side(seed: int, horizon: int, ctx: Mapping[str, Any]) -> Dict[str, Any]:
    with v2run.parity_config():
        set_world_seed(seed)
        sess = released_session()
    hero = sess.game.snakes[0]
    hero.auto_respawn = False  # the profile's hero_terminal; legacy served Watch respawns
    state = sess.safety_veto_state()
    counter = DecisionCounter(state["wrapped_snake_ids"])
    trace = []
    for _ in range(horizon):
        counter.step(sess)
        trace.append(live_frame_signature(sess.game))
    state = sess.safety_veto_state()
    return {
        "trace": trace,
        "served_checkpoint": sess.served_checkpoint,
        "checkpoint_sha256": state["checkpoint_sha256"],
        "config_sha256": schema.sha256_file(Path(sess.config_path)),
        "variant": state["variant"],
        "wrapper_method": (state["wrapper"] or {}).get("method"),
        "wrapped_snake_ids": state["wrapped_snake_ids"],
        "counters": state["counters"][str(hero.id)],
        "diagnostics": int_diagnostics(state["diagnostics"][str(hero.id)]),
        "decision_frames": counter.counts[int(hero.id)],
    }


def simd_profile(horizon: int) -> Any:
    """The gate's profile; a shorter (smoke/test) horizon changes only ``scored_horizon``
    (the mass-integral denominator), never the dynamics or the observation normalizer."""
    import src.scripts.tournament_eval as te

    profile = te.evaluation_profile_for_name(PROFILE_NAME, HORIZON)
    if horizon == HORIZON:
        return profile
    return dataclasses.replace(profile, scored_horizon=int(horizon))


def _simd_side(seed: int, horizon: int, ctx: Mapping[str, Any]) -> Dict[str, Any]:
    """The same identity on the SIMD engine: ``run_simd_eval`` with the frp3 hero (v8, lambda
    8, rowwise bit-exact forwards) against five frp3 opponents, one env on ``seed``. The
    per-frame trace is read from the BatchSim after every transition (observation only)."""
    import src.simd_env.eval_engine as ee
    from src.core.config_loader import load_and_initialize_config
    from src.core.game_config import GameConfig

    load_and_initialize_config(str(PARITY_CONFIG))
    profile = simd_profile(horizon)
    trace: List[Any] = []
    original = ee._hero_post_frame_snapshot

    def traced(sim: Any, events: Any) -> Any:
        trace.append(simd_frame_signature(sim, 0))
        return original(sim, events)

    spec = ("checkpoint", str(ctx["checkpoint"]))
    ee._hero_post_frame_snapshot = traced
    try:
        records = ee.run_simd_eval(
            spec,
            [spec] * (int(GameConfig.NUM_SNAKES) - 1),
            horizon,
            [int(seed)],
            profile=profile,
            frame_observer=lambda _frame: None,
            vector61=True,
            hero_safety_veto="v8",
            vector61_forward=VECTOR61_FORWARD,
            hero_safety_veto_lambda=V8_LAMBDA,
        )
    finally:
        ee._hero_post_frame_snapshot = original
    record = records[0]
    probe = record["probes"]["safety_veto"]
    keys = ("mass_integral", "max_mass", "deaths", "kills", "survival_fraction")
    return {
        "trace": trace,
        "profile_digest": profile.digest,
        "profile_scored_horizon": int(profile.scored_horizon),
        "provenance": record.get("vector61_policy"),
        "wrapper_method": probe.get("method"),
        "counters": {key: int(probe["counters"][key]) for key in schema.COUNTER_KEYS},
        "diagnostics": int_diagnostics(record["veto_diagnostics"]),
        "record_complete": True,
        "record": {key: record.get(key) for key in keys},
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
        "env": dict(RELEASE_ENV),
        "session": None,
        "simd": None,
        "frames_compared": 0,
        "first_divergence_frame": None,
        "divergence": None,
        "trace_sha256_equal": False,
        "veto_counters_equal": False,
        "diagnostics_equal": False,
        "wall_seconds": None,
    }
    started = time.monotonic()
    try:
        ctx["guard"]()
        left = _session_side(seed, horizon, ctx)
        right = _simd_side(seed, horizon, ctx)
        require(len(left["trace"]) == len(right["trace"]) == horizon, "trace length differs")
        pairs = list(zip(left.pop("trace"), right.pop("trace")))
        left["trace_sha256"] = schema.canonical_sha256([a for a, _ in pairs])
        right["trace_sha256"] = schema.canonical_sha256([b for _, b in pairs])
        record["session"], record["simd"] = left, right
        record["frames_compared"] = len(pairs)
        record["first_divergence_frame"] = next(
            (i for i, (a, b) in enumerate(pairs) if a != b), None
        )
        if record["first_divergence_frame"] is not None:
            i = record["first_divergence_frame"]
            record["divergence"] = {"session": pairs[i][0], "simd": pairs[i][1]}
        record["trace_sha256_equal"] = left["trace_sha256"] == right["trace_sha256"]
        record["veto_counters_equal"] = left["counters"] == right["counters"]
        record["diagnostics_equal"] = left["diagnostics"] == right["diagnostics"]
        record["status"] = schema.STATUS_COMPLETE
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    record["wall_seconds"] = time.monotonic() - started
    return record


# ---------------------------------------------------------------- launch guards


def on_ac_power() -> bool:
    from research.apex_safety_20260926 import dev_screen

    return bool(dev_screen.on_ac_power())


def parse_clamshell(text: str) -> Optional[bool]:
    """``True`` lid open, ``False`` closed, ``None`` unparseable (the census's parser)."""
    states = []
    for line in str(text or "").splitlines():
        if "AppleClamshellState" in line and "=" in line:
            value = line.split("=", 1)[1].strip()
            if value not in ("Yes", "No"):
                return None
            states.append(value == "No")
    return all(states) if states else None


def lid_open() -> bool:
    """The lid is open per ``ioreg`` (fails closed)."""
    try:
        out = subprocess.run(
            ["ioreg", "-r", "-k", "AppleClamshellState", "-d", "4"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return parse_clamshell(out) is True


def make_guard(
    ac: Callable[[], bool] = on_ac_power,
    lid: Callable[[], bool] = lid_open,
    sleep: Callable[[float], None] = time.sleep,
    max_pause: float = MAX_UNSAFE_PAUSE_SECONDS,
    poll: float = 30.0,
) -> Callable[[], None]:
    """Before each episode: wait while on battery or with the lid closed; past ``max_pause``
    raise (the episode then records the error and the run cannot pass S1)."""

    def guard() -> None:
        waited = 0.0
        while not (ac() and lid()):
            require(waited < max_pause, f"unsafe (battery or lid closed) for {waited:.0f} s")
            sleep(poll)
            waited += poll

    return guard


# ---------------------------------------------------------------- run


def check_output_root(out: Path, smoke: bool) -> None:
    require(not out.exists(), f"--out {out} already exists (create-only)")
    if smoke:
        require(ARTIFACT_ROOT not in out.parents, "a smoke never writes under the artifact root")


def harness_sources() -> Dict[str, str]:
    """sha256 of this harness's code, the v2/v8 harness code it imports, the registry and
    pins it serves through, and the SIMD engine files the parity probe runs."""
    paths = [HERE / "serving_run.py", HERE / "schema.py", HERE / "serving_audit.py"]
    paths += [HERE / "fill_strict_pin.py"]
    for harness in (V2_HARNESS, V8_HARNESS):
        paths += [harness, harness.parent / "schema.py", harness.parent / "serving_audit.py"]
    paths += [
        REPO / "web/backend/served_checkpoint.py",
        REPO / "web/backend/served_checkpoint_pins.json",
        REPO / "web/backend/safety_veto_serving.py",
        REPO / "web/backend/session.py",
        REPO / "src/simd_env/eval_engine.py",
        REPO / "src/simd_env/vector61_policy.py",
        REPO / "src/simd_env/batch_sim.py",
    ]
    return {str(p.relative_to(REPO)): schema.sha256_file(p) for p in paths}


def resolve_identity(smoke: bool) -> Dict[str, Any]:
    """The checkpoint the release env serves (via the registry) and the pins it serves under.

    A real run needs the repo's pin filled and valid; a smoke serves under the in-process
    placeholder pin (:func:`smoke_pins`). Refuses unless the registry resolves the release
    env to ``frp3-s12`` with the pinned bytes.
    """
    from web.backend import served_checkpoint as registry

    repo_pins = registry.load_pins()
    problems = registry.strict_pin_problems(CHECKPOINT_NAME, repo_pins)
    if smoke:
        run_pins: Optional[Dict[str, Any]] = smoke_pins()
    else:
        require(
            not problems,
            f"the {CHECKPOINT_NAME} strict-receipt pin is not filled/valid: {problems} "
            "(fill it with fill_strict_pin.py after the strict gate passes)",
        )
        run_pins = None
    with pins_override(run_pins), serving_env(RELEASE_ENV):
        choice = registry.resolve_served_checkpoint()
    require(
        choice.name == CHECKPOINT_NAME and choice.sha256 == CHECKPOINT_SHA256,
        f"the release env does not resolve to the pinned {CHECKPOINT_NAME}: {choice.to_dict()}",
    )
    return {
        "choice": choice.to_dict(),
        "run_pins": run_pins,
        "repo_pin_problems": problems,
        "pin": pin_entry(run_pins if run_pins is not None else repo_pins),
    }


def build_intent(args: argparse.Namespace, identity: Mapping[str, Any]) -> Dict[str, Any]:
    from web.backend import safety_veto_serving as serving
    from web.backend import served_checkpoint as registry
    from web.backend import session as web_session

    counts = SMOKE_COUNTS if args.smoke else COUNTS
    domain = SMOKE_SEED_DOMAIN if args.smoke else SEED_DOMAIN
    seeds = world_seeds(domain, counts)
    report = seed_report(seeds, identity["pin"])
    require(report["disjoint"], f"serving seeds overlap earlier banks: {report}")
    if not args.smoke:
        require(report["strict_observed_checked"], "the strict run's seeds were not checked")
    wrapper = serving.wrapper_identity_v8()
    require(wrapper["source_sha256s"] == V8_SOURCE_SHA256S, "v8 veto sources changed")
    require(wrapper["method"] == V8_METHOD, f"v8 method {wrapper['method']!r}")
    require(wrapper["descriptor"].get("head_avoidance") is True, "v8 head layer is off")
    require(
        serving.V8_STRICT_RECEIPT_SOURCE_SHA256S == V8_SOURCE_SHA256S
        and serving.V8_STRICT_RECEIPT_CHECKPOINT_SHA256 == CHAMPION_SHA256
        and serving.V8_STRICT_RECEIPT_METHOD == V8_METHOD
        and serving.V8_LAMBDA == V8_LAMBDA
        and serving.VARIANT_RELEASED_DEFAULT == "v8",
        "the web hook's v8 pins or released default differ from this harness's",
    )
    entry = registry.REGISTRY[CHECKPOINT_NAME]
    require(
        (entry.sha256, entry.filename, entry.artifact_path)
        == (CHECKPOINT_SHA256, CHECKPOINT_FILENAME, str(CHECKPOINT_ARTIFACT_PATH))
        and registry.CHECKPOINT_RELEASED_DEFAULT == "champion"
        and registry.V8_METHOD == V8_METHOD,
        "the registry's frp3-s12 entry or released default differs from this harness's",
    )
    require(
        Path(web_session.DEFAULT_CHECKPOINT).is_file()
        and schema.sha256_file(Path(web_session.DEFAULT_CHECKPOINT)) == CHAMPION_SHA256,
        f"the served default champion is missing or changed at {web_session.DEFAULT_CHECKPOINT} "
        "(copy saved_snakes/champion_a5_freespace_20260621.pth into this checkout)",
    )
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
        "harness_sources": harness_sources(),
        "checkpoint": {"path": identity["choice"]["path"], "sha256": CHECKPOINT_SHA256},
        "served_checkpoint": identity["choice"],
        "strict_pin": identity["pin"],
        "strict_pin_placeholder_for_smoke": identity["run_pins"] is not None,
        "release_env": dict(RELEASE_ENV),
        "expected": {
            "checkpoint_name": CHECKPOINT_NAME,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "checkpoint_filename": CHECKPOINT_FILENAME,
            "checkpoint_artifact_path": str(CHECKPOINT_ARTIFACT_PATH),
            "champion_sha256": CHAMPION_SHA256,
            "v8_strict_receipt_sha256": v8run.V8_STRICT_RECEIPT_SHA256,
            "v8_method": V8_METHOD,
            "v8_lambda": V8_LAMBDA,
            "v8_head_avoidance": True,
            "v8_reference_lambda": None,
            "v8_source_sha256s": dict(V8_SOURCE_SHA256S),
            "served_config_basename": "mechanics_v2.yaml",
            "served_config_sha256": SERVED_CONFIG_SHA256,
            "parity_config_sha256": PARITY_CONFIG_SHA256,
            "parity_profile_digest": PROFILE_DIGEST,
            "simd_forward": VECTOR61_FORWARD,
        },
        "wrapper_identity": wrapper,
        "horizon_frames": SMOKE_HORIZON if args.smoke else HORIZON,
        "parity_horizon_frames": SMOKE_PARITY_HORIZON if args.smoke else HORIZON,
        "counts": dict(counts),
        "seed_recipe": {
            "domain": domain,
            "rule": "uint32 big-endian of sha256(f'{domain}|{purpose}|{i}')[:4]",
        },
        "seeds": seeds,
        "seed_report": report,
        "criteria": ["S1", "S2", "S3", "S4", "S5", "S6"],
    }


def acquire_slot(timeout: float, pool: int) -> List[Any]:
    from research.apex_safety_20260926 import dev_screen

    return dev_screen.acquire_cpu_slots(dev_screen.DEFAULT_SLOT_LOCK_ROOT, 1, timeout, pool=pool)


def release_slot(handles: Sequence[Any]) -> None:
    from research.apex_safety_20260926 import dev_screen

    dev_screen.release_cpu_slots(handles)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="1 watch + 1 play + 1 parity, short")
    parser.add_argument("--slot-timeout", type=float, default=7200.0, help="wait for a CPU slot")
    parser.add_argument("--slot-pool", type=int, default=SLOT_POOL, choices=(2, 3))
    args = parser.parse_args(argv)
    out = args.out.resolve()
    check_output_root(out, args.smoke)

    import torch

    torch.set_num_threads(1)  # as web/backend/app.py and web/serve.py pin it
    os.environ.setdefault("SNAKE_DQN_DEVICE", "cpu")
    identity = resolve_identity(args.smoke)
    if not args.smoke:
        require(on_ac_power() and lid_open(), "a real run needs AC power and an open lid")
    guard = make_guard() if not args.smoke else (lambda: None)
    slots = acquire_slot(args.slot_timeout, args.slot_pool)
    try:
        with pins_override(identity["run_pins"]):
            return _run(args, out, identity, guard)
    finally:
        release_slot(slots)


def _run(
    args: argparse.Namespace, out: Path, identity: Mapping[str, Any], guard: Callable[[], None]
) -> int:
    started = time.monotonic()
    intent = build_intent(args, identity)
    intent_sha = write_new_json(out / schema.INTENT, intent)
    ctx = {
        "checkpoint": Path(identity["choice"]["path"]),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "intent_sha256": intent_sha,
        "guard": guard,
    }
    horizon, parity_horizon = intent["horizon_frames"], intent["parity_horizon_frames"]

    files: Dict[str, str] = {}
    files[schema.DEFAULT_CHECK] = write_new_json(
        out / schema.DEFAULT_CHECK,
        {"intent_sha256": intent_sha, **default_check(identity["run_pins"])},
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
    probes = []
    for i, seed in enumerate(intent["seeds"]["parity"]):
        probe = run_parity_probe(i, seed, parity_horizon, ctx)
        probes.append(probe)
        rel = f"{schema.PARITY_DIR}/{probe['probe_id']}.json"
        files[rel] = write_new_json(out / rel, probe)

    failures = [r["episode_id"] for r in records if r["status"] != schema.STATUS_COMPLETE]
    failures += [p["probe_id"] for p in probes if p["status"] != schema.STATUS_COMPLETE]
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
