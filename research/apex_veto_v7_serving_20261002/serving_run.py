"""Web serving run for the Apex champion + v7 space-preference veto on the Watch hero.

Drives the REAL web backend headlessly, as the v2 and v5 serving runs did
(``research/apex_veto_serving_20261001/serving_run.py``, whose stepping, stand-in,
decision counting, dispatch and rollout-tracing helpers are imported unchanged):
``GameSession.step()`` then ``snapshot()`` (``Hub.engine_loop`` order), every control
through ``web.backend.app._apply_control``. No server, no network listener.

Every served build uses the v7 release environment: ``SNAKE_SERVE_VETO_VARIANT=v7`` with
``SNAKE_SERVE_VETO_WATCH_HERO`` and ``SNAKE_SERVE_VETO_PLAY_AI`` unset (Watch hero on by
the released default, Play AI off). Episodes (pre-registered in ``protocol.md``):

* Watch episodes as served: all-respawn Watch, the v7 wrapper on the slot-0 hero only;
* Play episodes: a scripted stand-in steers the human through ``human_input`` controls;
  no snake is wrapped (Play AI wrapping is not part of the release);
* parity probes: session stepping on the gate's deployment config with a terminal hero,
  compared frame by frame with ``tournament_eval.rollout(..., hero_safety_veto=True)``
  under ``dev_screen.hero_veto_installer(strict_run.install_candidate_veto)``, the v7
  strict gate's candidate install (lambda 4.0).

Every record binds the checkpoint sha256, the v7 wrapper identity (four source sha256s),
the env the build read, the served config, and the veto counters and v7 diagnostics
(v5 nested). ``serving_audit.py`` (stdlib) checks them. Output is create-only.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.apex_veto_serving_20261001 import serving_run as v2run  # noqa: E402


def _load_schema() -> Any:
    spec = importlib.util.spec_from_file_location("apex_veto_v7_serving_schema", HERE / "schema.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


schema = _load_schema()

PROTOCOL = HERE / "protocol.md"
# sha256 of protocol.md as pre-registered for the real run (the audit pins it too).
PROTOCOL_SHA256 = "6028193ecda233cd84be7d63a0896362690e5e54f02fafc4198d27500124a6da"
CHAMPION_PATH = v2run.CHAMPION_PATH
CHAMPION_SHA256 = v2run.CHAMPION_SHA256
SERVED_CONFIG_SHA256 = v2run.SERVED_CONFIG_SHA256
PARITY_CONFIG = v2run.PARITY_CONFIG
PARITY_CONFIG_SHA256 = v2run.PARITY_CONFIG_SHA256
PROFILE_NAME = v2run.PROFILE_NAME
V7_STRICT_RECEIPT_SHA256 = "86ee36791e33ecad06bd525e8190d97f1c012ac6e3ede33ba05abe544f750422"
V7_STRICT_INTENT_SHA256 = "3e39e0699ac1ba27880ba35cac8d2d5e4a74fa58305aa63477dac4735a153b33"
V7_STRICT_AUDIT_REPORT_SHA256 = "9cfc7dbb7eaa6d33cd9b5d6e602cc412a68886128076fa96d9196887c7e86183"
V7_LAMBDA = 4.0
V7_METHOD = "free-space-veto/v7-space-preference(lambda=4.0)"
V7_SOURCE_SHA256S = {
    "src/evaluation/safety_veto.py": (
        "1b62d15c48987584533efbefdf3fd84e1e3f2e22cfe70dbb82b43ec57169c428"
    ),
    "src/evaluation/safety_veto_v3.py": (
        "ed3a6d860b09afd982bc6c87ea0a86566455dfcf9562133d1772e595b4bb5be1"
    ),
    "src/evaluation/safety_veto_v5.py": (
        "d86d084e7778fc514c4932b27f3750f5f11543e3c7afa44571869407870ec86c"
    ),
    "src/evaluation/safety_veto_v7.py": (
        "56ff7009ce2e0c4c93b6570336b35d48341757fc53b386d309b00126f67e2980"
    ),
}
V2_HARNESS = REPO / "research/apex_veto_serving_20261001/serving_run.py"
V5_HARNESS = REPO / "research/apex_veto_v5_serving_20261001/serving_run.py"

HORIZON = 5000
COUNTS = {"watch": 25, "play": 25, "parity": 2}
SMOKE_HORIZON = 500
SMOKE_PARITY_HORIZON = 200
SMOKE_COUNTS = {"watch": 1, "play": 1, "parity": 1}
SEED_DOMAIN = "apex-veto-v7-web-serving-v1"
SMOKE_SEED_DOMAIN = "apex-veto-v7-web-serving-smoke-v1"
SMALL_INTEGER_SEEDS = 1000
# In-flight v8 study domains (branch veto-v8; not on main): excluded for hygiene.
V8_DOMAINS = (
    "apex-veto-v8-dev-v1",
    "apex-veto-v8-dev-smoke-v1",
    "apex-veto-v8-screen-v1",
    "apex-veto-v8-screen-smoke-v1",
)
ARTIFACT_ROOT = v2run.ARTIFACT_ROOT

# The v7 release environment every served build reads (None = unset).
RELEASE_ENV = {
    "SNAKE_SERVE_VETO_WATCH_HERO": None,
    "SNAKE_SERVE_VETO_PLAY_AI": None,
    "SNAKE_SERVE_VETO_VARIANT": "v7",
}

require = v2run.require
write_new_json = v2run.write_new_json
world_seeds = v2run.world_seeds
git_state = v2run.git_state
DecisionCounter = v2run.DecisionCounter
ScriptedHuman = v2run.ScriptedHuman
control = v2run.control
set_world_seed = v2run.set_world_seed
served_identity = v2run.served_identity


def counter_delta(after: Mapping[str, int], before: Mapping[str, int]) -> Dict[str, int]:
    return {key: int(after[key]) - int(before.get(key, 0)) for key in schema.COUNTER_KEYS}


def seed_report(seeds: Mapping[str, Sequence[int]]) -> Dict[str, Any]:
    """Fail-closed freshness check against every earlier bank (in-code; no registry).

    Union of: the v5 serving lane's report (which contains the v2 serving lane's), the v7
    strict gate's ``EARLIER_DOMAINS`` (first 1000 seeds of each domain/purpose), the v7
    strict dev/final/serving banks and their smoke (first 1000 each), the in-flight v8
    domains (first 1000 each) and the small integer seeds 0..999.
    """
    from research.apex_safety_20260926 import dev_screen
    from research.apex_veto_v5_serving_20261001 import serving_run as v5run
    from research.apex_veto_v7_strict_20261002 import strict_run as v7strict

    flat = [seed for values in seeds.values() for seed in values]
    v5_report = v5run.seed_report(seeds)
    earlier = set(range(SMALL_INTEGER_SEEDS))
    for bank in v7strict.earlier_namespaces().values():
        earlier |= set(bank)
    v7_domains = [domain for domain, _ in v7strict.NAMESPACES.values()]
    for domain in v7_domains + [v7strict.SMOKE_DOMAIN]:
        earlier |= {
            dev_screen.uint32_seed(domain, v7strict.NAMESPACE_KEY, i)
            for i in range(v7strict.EXCLUSION_PREFIX)
        }
    for domain in V8_DOMAINS:
        earlier |= {
            dev_screen.uint32_seed(domain, "worlds", i) for i in range(v7strict.EXCLUSION_PREFIX)
        }
    overlap = sorted(earlier & set(flat))
    unique = len(set(flat)) == len(flat)
    return {
        "unique": unique,
        "v5_serving_report": v5_report,
        "v7_earlier_overlap": overlap,
        "checked_against": v5_report["checked_against"]
        + [f"{d}[:1000] (v7 strict EARLIER_DOMAINS)" for d in v7strict.EARLIER_DOMAINS]
        + [f"{d}/worlds[:1000]" for d in v7_domains + [v7strict.SMOKE_DOMAIN]]
        + [f"{d}/worlds[:1000] (in-flight v8)" for d in V8_DOMAINS]
        + ["small integer seeds 0..999"],
        "registry_checked": False,
        "registry_note": "governance namespace registry not implemented; in-code check only",
        "disjoint": unique and bool(v5_report["disjoint"]) and not overlap,
    }


@contextlib.contextmanager
def serving_env(values: Mapping[str, Optional[str]]) -> Iterator[None]:
    """Set the three serving env vars exactly as ``values`` says (None = unset)."""
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


def snakes_with_veto(sess: Any) -> List[int]:
    return sorted(int(s.id) for s in sess.game.snakes if getattr(s, "safety_veto", None))


V2_METHOD = "free-space-veto/v2-speed-preserving"
V5_METHOD = "free-space-veto/v5-boost-aware"
DEFAULT_CASES = {
    "empty_env": {},
    "variant_v5": {"SNAKE_SERVE_VETO_VARIANT": "v5"},
    "variant_v2": {"SNAKE_SERVE_VETO_VARIANT": "v2"},
    "watch_off_variant_v7": {
        "SNAKE_SERVE_VETO_WATCH_HERO": "0",
        "SNAKE_SERVE_VETO_VARIANT": "v7",
    },
}
# Which wrapper each released-default case must put on the Watch hero.
DEFAULT_CASE_WATCH = {
    "empty_env": ("v5", V5_METHOD),
    "variant_v5": ("v5", V5_METHOD),
    "variant_v2": ("v2", V2_METHOD),
}


def default_check(checkpoint: Path) -> Dict[str, Any]:
    """S6: the released v5 default and the rollbacks, read back from real builds."""
    from web.backend import session as web_session
    from web.backend.safety_veto_serving import ServingVetoFlags

    flags = ServingVetoFlags.from_env({})
    out: Dict[str, Any] = {
        "flags_from_empty_env": flags.to_dict(),
        "variant_from_empty_env": flags.variant,
        "default_checkpoint_basename": os.path.basename(web_session.DEFAULT_CHECKPOINT),
        "served_config_basename": os.path.basename(web_session._config_for(61)),
        "served_config_sha256": schema.sha256_file(Path(web_session._config_for(61))),
    }
    builds: Dict[str, Any] = {}
    for name, env in DEFAULT_CASES.items():
        with serving_env(env):
            sess = web_session.GameSession(checkpoint=str(checkpoint))
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
                    "snakes_with_veto": snakes_with_veto(sess),
                    "hero_id": int(sess.hero_id),
                }
        builds[name] = {"env": dict(env), "modes": modes}
    out["builds"] = builds
    out["pass"] = bool(default_check_failures(out) == [])
    return out


def default_check_failures(doc: Mapping[str, Any]) -> List[str]:
    """Producer-side S6 rules (the audit re-implements them)."""
    out = []
    if doc["flags_from_empty_env"] != {"watch_hero": True, "play_ai": False}:
        out.append("empty env flags are not the released {watch_hero: on, play_ai: off}")
    if doc["variant_from_empty_env"] != "v5":
        out.append("empty env variant is not the released v5")
    if doc["default_checkpoint_basename"] != CHAMPION_PATH.name:
        out.append("served default checkpoint changed")
    if (doc["served_config_basename"], doc["served_config_sha256"]) != (
        "mechanics_v2.yaml",
        SERVED_CONFIG_SHA256,
    ):
        out.append("served default config changed")
    for name, (variant, method) in DEFAULT_CASE_WATCH.items():
        modes = doc["builds"][name]["modes"]
        watch, play = modes["watch"], modes["play"]
        if not (
            watch["active"]
            and watch["variant"] == variant
            and watch["method"] == method
            and watch["snakes_with_veto"] == [watch["hero_id"]]
        ):
            out.append(f"{name}: Watch is not the {variant} hero wrapper")
        if play["active"] or play["snakes_with_veto"]:
            out.append(f"{name}: Play is wrapped")
    for mode, state in doc["builds"]["watch_off_variant_v7"]["modes"].items():
        if state["active"] or state["snakes_with_veto"]:
            out.append(f"watch_off_variant_v7: {mode} is wrapped")
    return out


def int_diagnostics(record: Mapping[str, Any]) -> Dict[str, Any]:
    """The integer v7 diagnostics with the integer v5 diagnostics nested under ``v5``."""
    out: Dict[str, Any] = {key: int(record[key]) for key in schema.DIAGNOSTIC_KEYS}
    nested = record[schema.V5_NESTED_KEY]
    out[schema.V5_NESTED_KEY] = {key: int(nested[key]) for key in schema.V5_DIAGNOSTIC_KEYS}
    return out


def diagnostics_total(per_snake: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """Per-snake sums (``area_cap_max``: the maximum), the nested v5 block likewise."""
    values = list(per_snake.values())
    out: Dict[str, Any] = {}
    for key in schema.DIAGNOSTIC_KEYS:
        column = [d[key] for d in values]
        out[key] = (
            (max(column) if column else 0) if key in schema.DIAGNOSTIC_MAX_KEYS else sum(column)
        )
    out[schema.V5_NESTED_KEY] = {
        key: sum(d[schema.V5_NESTED_KEY][key] for d in values) for key in schema.V5_DIAGNOSTIC_KEYS
    }
    return out


def veto_block(sess: Any, before: Mapping[str, Mapping[str, int]]) -> Dict[str, Any]:
    """Counter and diagnostics deltas over the episode, plus the build's veto state."""
    state = sess.safety_veto_state()
    per_snake = {
        sid: counter_delta(counters, before.get(sid, {}))
        for sid, counters in state["counters"].items()
    }
    diagnostics = {sid: int_diagnostics(d) for sid, d in state["diagnostics"].items()}
    timing = {
        sid: {key: d.get(key) for key in schema.DIAGNOSTIC_TIMING_KEYS}
        for sid, d in state["diagnostics"].items()
    }
    return {
        "active": state["active"],
        "variant": state["variant"],
        "variant_requested": state["variant_requested"],
        "scope": state["scope"],
        "reason": state["reason"],
        "flags": state["flags"],
        "strict_receipt_checkpoint_match": state["strict_receipt_checkpoint_match"],
        "strict_receipt_wrapper_match": state["strict_receipt_wrapper_match"],
        "wrapped_snake_ids": state["wrapped_snake_ids"],
        "per_snake": per_snake,
        "total": {key: sum(c[key] for c in per_snake.values()) for key in schema.COUNTER_KEYS},
        "diagnostics_per_snake": diagnostics,
        "diagnostics_total": diagnostics_total(diagnostics),
        "diagnostics_timing": timing,
    }


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
        "wrapper": None,
        "served": None,
        "veto": None,
        "snakes_with_veto": None,
        "decision_frames": None,
        "episode": None,
    }


def _finish(record: Dict, sess: Any, before: Mapping, counter: Any) -> None:
    state = sess.safety_veto_state()
    record["wrapper"] = state["wrapper"]
    record["checkpoint"]["session_sha256"] = state["checkpoint_sha256"]
    record["served"] = served_identity(sess)
    record["veto"] = veto_block(sess, before)
    record["snakes_with_veto"] = snakes_with_veto(sess)
    per_snake = {str(sid): n for sid, n in sorted(counter.counts.items())}
    record["decision_frames"] = {"per_snake": per_snake, "total": sum(per_snake.values())}


def run_watch_episode(index: int, seed: int, horizon: int, ctx: Mapping[str, Any]) -> Dict:
    """The served Watch configuration: all-respawn Watch, the v7 wrapper on the hero."""
    from web.backend.session import GameSession

    record = base_record(schema.KIND_WATCH, index, seed, horizon, ctx)
    started = time.monotonic()
    try:
        with serving_env(RELEASE_ENV):
            set_world_seed(seed)
            sess = GameSession(checkpoint=str(ctx["checkpoint"]))
        state = sess.safety_veto_state()
        before = state["counters"]
        require(
            all(d["decisions"] == 0 for d in state["diagnostics"].values()),
            "a fresh Watch build carried v7 diagnostics",
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
    from web.backend.session import GameSession

    with serving_env(RELEASE_ENV):
        sess = GameSession(checkpoint=str(ctx["checkpoint"]))
        control(sess, "set_mode", "play")
    require(sess.mode == "play", f"session did not enter play mode: {sess.last_error}")
    return sess


def run_play_episode(sess: Any, index: int, seed: int, horizon: int, ctx: Mapping) -> Dict:
    """One scored Play run (new_game, scripted human input) that must stay unwrapped."""
    record = base_record(schema.KIND_PLAY, index, seed, horizon, ctx)
    started = time.monotonic()
    try:
        set_world_seed(seed)
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


def _session_side(seed: int, horizon: int, ctx: Mapping[str, Any]) -> Dict[str, Any]:
    from web.backend.session import GameSession

    with v2run.parity_config(), serving_env(RELEASE_ENV):
        set_world_seed(seed)
        sess = GameSession(checkpoint=str(ctx["checkpoint"]))
    hero = sess.game.snakes[0]
    hero.auto_respawn = False  # the profile's hero_terminal; legacy served Watch respawns
    state = sess.safety_veto_state()
    counter = DecisionCounter(state["wrapped_snake_ids"])
    trace = []
    for _ in range(horizon):
        counter.step(sess)
        trace.append(v2run.frame_signature(sess.game))
    state = sess.safety_veto_state()
    return {
        "trace": trace,
        "config_sha256": schema.sha256_file(Path(sess.config_path)),
        "variant": state["variant"],
        "wrapper_method": (state["wrapper"] or {}).get("method"),
        "wrapped_snake_ids": state["wrapped_snake_ids"],
        "counters": state["counters"][str(hero.id)],
        "diagnostics": int_diagnostics(state["diagnostics"][str(hero.id)]),
        "decision_frames": counter.counts[int(hero.id)],
    }


def _rollout_side(seed: int, horizon: int, ctx: Mapping[str, Any]) -> Dict[str, Any]:
    """v2's traced rollout under the v7 strict gate's candidate installer."""
    from research.apex_safety_20260926 import dev_screen
    from research.apex_veto_v7_strict_20261002 import strict_run as v7strict

    require(v7strict.CANDIDATE_LAMBDA == V7_LAMBDA, "v7 strict candidate lambda changed")
    with dev_screen.hero_veto_installer(v7strict.install_candidate_veto) as installed:
        out = v2run._rollout_side(seed, horizon, ctx)
    require(len(installed) == 1, f"rollout installed {len(installed)} v7 vetoes")
    out["wrapper_method"] = installed[0].method
    out["diagnostics"] = int_diagnostics(installed[0].diagnostics_record())
    return out


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
        "rollout": None,
        "frames_compared": 0,
        "first_divergence_frame": None,
        "trace_sha256_equal": False,
        "veto_counters_equal": False,
        "diagnostics_equal": False,
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
        record["diagnostics_equal"] = left["diagnostics"] == right["diagnostics"]
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


def harness_sources() -> Dict[str, str]:
    """sha256 of this harness's code and the v2/v5 harness code it imports or loads."""
    paths = [HERE / "serving_run.py", HERE / "schema.py", HERE / "serving_audit.py"]
    paths += [V2_HARNESS, V2_HARNESS.parent / "schema.py", V2_HARNESS.parent / "serving_audit.py"]
    paths += [V5_HARNESS, V5_HARNESS.parent / "schema.py", V5_HARNESS.parent / "serving_audit.py"]
    return {str(p.relative_to(REPO)): schema.sha256_file(p) for p in paths}


def build_intent(args: argparse.Namespace, checkpoint_sha256: str) -> Dict[str, Any]:
    from web.backend import safety_veto_serving as serving

    counts = SMOKE_COUNTS if args.smoke else COUNTS
    domain = SMOKE_SEED_DOMAIN if args.smoke else SEED_DOMAIN
    seeds = world_seeds(domain, counts)
    report = seed_report(seeds)
    require(report["disjoint"], f"serving seeds overlap earlier banks: {report}")
    identity = serving.wrapper_identity_v7()
    require(
        identity["source_sha256s"] == V7_SOURCE_SHA256S,
        f"v7 veto sources differ from the v7 strict receipt: {identity['source_sha256s']}",
    )
    require(identity["method"] == V7_METHOD, f"v7 method {identity['method']!r}")
    require(
        serving.V7_STRICT_RECEIPT_SOURCE_SHA256S == V7_SOURCE_SHA256S
        and serving.V7_STRICT_RECEIPT_CHECKPOINT_SHA256 == CHAMPION_SHA256
        and serving.V7_STRICT_RECEIPT_SHA256 == V7_STRICT_RECEIPT_SHA256
        and serving.V7_STRICT_RECEIPT_METHOD == V7_METHOD
        and serving.V7_LAMBDA == V7_LAMBDA,
        "the web hook's v7 pins differ from this harness's pins",
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
        "checkpoint": {"path": str(args.checkpoint), "sha256": checkpoint_sha256},
        "release_env": dict(RELEASE_ENV),
        "expected": {
            "checkpoint_sha256": CHAMPION_SHA256,
            "v7_strict_receipt_sha256": V7_STRICT_RECEIPT_SHA256,
            "v7_strict_intent_sha256": V7_STRICT_INTENT_SHA256,
            "v7_strict_audit_report_sha256": V7_STRICT_AUDIT_REPORT_SHA256,
            "v7_method": V7_METHOD,
            "v7_lambda": V7_LAMBDA,
            "v7_source_sha256s": dict(V7_SOURCE_SHA256S),
            "served_config_basename": "mechanics_v2.yaml",
            "served_config_sha256": SERVED_CONFIG_SHA256,
            "parity_config_sha256": PARITY_CONFIG_SHA256,
        },
        "wrapper_identity": identity,
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
