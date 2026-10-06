#!/usr/bin/env python3
"""Audit of the FRP-v3 s12 checkpoint-swap web-serving run (standard library only).

``python -I serving_audit.py --root <run-out> --out <run-out>/audit``. Exit 0 =
SERVING_PASS, 1 = SERVING_FAIL, 2 = INVALID (evidence missing or unreadable).

It imports nothing from the repo except ``schema.py`` beside it and, by file path, the v8
serving audit (``research/apex_veto_v8_serving_20261003/serving_audit.py``, which itself loads
the v2 and v7 audits) for its evidence loader, its v8 wrapper identity, counter and
diagnostics rules, its S4 counters criterion (unchanged: the served veto is the released v8)
and its reported block. It also reads the repo's pins file
(``web/backend/served_checkpoint_pins.json``, JSON) and the strict run's receipt, intent and
audit report that pin names. The checkpoint, config, profile and release-env pins and every
swap-specific rule are re-implemented here; each rule names the code path that writes the
field it checks (protocol.md "Pass criteria").
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
V8_AUDIT = HERE.parent / "apex_veto_v8_serving_20261003" / "serving_audit.py"
PINS_FILE = REPO / "web" / "backend" / "served_checkpoint_pins.json"


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


schema = _load("frp3_checkpoint_serving_schema", HERE / "schema.py")
v8audit = _load("frp3_serving_v8_audit", V8_AUDIT)
# The owner pin tool's structured STRICT_PASS rule (stdlib at import; loaded by file path).
fill_tool = _load("frp3_serving_fill_strict_pin", HERE / "fill_strict_pin.py")
Invalid = v8audit.Invalid
uint32_seed = v8audit.uint32_seed
is_int = v8audit.is_int
counter_failures = v8audit.counter_failures
load_evidence = v8audit.load_evidence
episode_records = v8audit.episode_records
parity_records = v8audit.parity_records
wrapper_failures = v8audit.wrapper_failures
diagnostics_failures = v8audit.diagnostics_failures

CHECKPOINT_NAME = "frp3-s12"
CHECKPOINT_SHA256 = "eec144bf92509a42664e2d650b8741f7d7774b86d7b6601d012bdd9425dd3723"
CHECKPOINT_FILENAME = "frp3_m3_s12_u60000_20261005.pth"
CHECKPOINT_ARTIFACT_PATH = (
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/frp-v3-20261005/train/arm-M3/seed-12/"
    "checkpoints/apex_mark_u60000.pth"
)
CHAMPION_SHA256 = v8audit.CHAMPION_SHA256
CHAMPION_BASENAME = v8audit.CHAMPION_BASENAME
V8_METHOD = v8audit.V8_METHOD
V7_METHOD = v8audit.V7_METHOD
V8_SOURCE_SHA256S = v8audit.V8_SOURCE_SHA256S
SERVED_CONFIG = v8audit.SERVED_CONFIG
PARITY_CONFIG_SHA256 = v8audit.PARITY_CONFIG_SHA256
PROFILE_DIGEST = "d396d3ed93e3a264d050674887eb47e59de67d3c6696c2c41fa5f8b145ea0e8b"
RELEASE_ENV = {
    "SNAKE_SERVE_CHECKPOINT": CHECKPOINT_NAME,
    "SNAKE_SERVE_VETO_WATCH_HERO": None,
    "SNAKE_SERVE_VETO_PLAY_AI": None,
    "SNAKE_SERVE_VETO_VARIANT": None,
}
RELEASE_FLAGS = v8audit.RELEASE_FLAGS
PLAY_REASON = v8audit.PLAY_REASON
SMOKE_PIN_ROOT = "/smoke-placeholder-not-a-strict-run"
STRICT_FILES = ("receipt", "intent", "audit_report")
# protocol.md as pre-registered for the real run. A non-smoke intent must carry exactly
# this sha256 (a later protocol edit cannot audit as SERVING_PASS).
PROTOCOL_SHA256 = "9163461767343f94a52e9a64f83f201b68ed3b4b322400854284fffe868727c6"
DESIGN = {
    False: {
        "domain": "frp3-checkpoint-web-serving-v1",
        "counts": {"watch": 25, "play": 25, "parity": 2},
        "horizon": 5000,
        "parity_horizon": 5000,
    },
    True: {
        "domain": "frp3-checkpoint-web-serving-smoke-v1",
        "counts": {"watch": 1, "play": 1, "parity": 1},
        "horizon": 500,
        "parity_horizon": 200,
    },
}
SIMD_PROVENANCE = {
    "engine": "simd",
    "policy": "Vector61SimdPolicy",
    "forward": "rowwise",
    "bit_exact_forward": True,
    "hero_safety_veto": True,
    "safety_veto_method": V8_METHOD,
}
# S6: case -> (served name, session sha, Watch (variant, method) or None = unwrapped).
DEFAULT_CASE_EXPECT = {
    "empty_env": ("champion", CHAMPION_SHA256, ("v8", V8_METHOD)),
    "checkpoint_champion": ("champion", CHAMPION_SHA256, ("v8", V8_METHOD)),
    "checkpoint_unknown": ("champion", CHAMPION_SHA256, ("v8", V8_METHOD)),
    "frp3_pin_unfilled": ("champion", CHAMPION_SHA256, ("v8", V8_METHOD)),
    "frp3_rollback_variant_v7": (CHECKPOINT_NAME, CHECKPOINT_SHA256, None),
    "frp3_watch_off": (CHECKPOINT_NAME, CHECKPOINT_SHA256, None),
}
DEFAULT_CASE_ENV = {
    "empty_env": {},
    "checkpoint_champion": {"SNAKE_SERVE_CHECKPOINT": "champion"},
    "checkpoint_unknown": {"SNAKE_SERVE_CHECKPOINT": "frp3-s99"},
    "frp3_pin_unfilled": {"SNAKE_SERVE_CHECKPOINT": CHECKPOINT_NAME},
    "frp3_rollback_variant_v7": {
        "SNAKE_SERVE_CHECKPOINT": CHECKPOINT_NAME,
        "SNAKE_SERVE_VETO_VARIANT": "v7",
    },
    "frp3_watch_off": {
        "SNAKE_SERVE_CHECKPOINT": CHECKPOINT_NAME,
        "SNAKE_SERVE_VETO_WATCH_HERO": "0",
    },
}
REFUSED_CASES = ("checkpoint_unknown", "frp3_pin_unfilled")
KIND_PURPOSE = {schema.KIND_WATCH: "watch", schema.KIND_PLAY: "play"}
MAX_LISTED = 50


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_pins(path: Path = PINS_FILE) -> Mapping[str, Any]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise Invalid(f"pins file unreadable: {path}: {exc}")
    if not isinstance(doc, Mapping):
        raise Invalid(f"pins file is not an object: {path}")
    return doc


# ---------------------------------------------------------------- reusable rules


def served_checkpoint_failures(block: Any, where: str) -> List[str]:
    """The registry's resolution under the release env (writer:
    ``served_checkpoint.resolve_served_checkpoint`` via ``GameSession.__init__``)."""
    if not isinstance(block, Mapping) or set(block) != set(schema.SERVED_CHECKPOINT_KEYS):
        return [f"{where}: served_checkpoint keys"]
    checks = {
        "requested": block["requested"] == CHECKPOINT_NAME,
        "name": block["name"] == CHECKPOINT_NAME,
        "sha256": block["sha256"] == CHECKPOINT_SHA256,
        "reason": block["reason"] is None and block["pin_problems"] == [],
        "path": isinstance(block["path"], str)
        and (
            block["path"] == CHECKPOINT_ARTIFACT_PATH
            or block["path"].endswith("/saved_snakes/" + CHECKPOINT_FILENAME)
        ),
    }
    return [f"{where}: served_checkpoint.{k}" for k, ok in checks.items() if not ok]


def strict_pin_failures(intent: Mapping[str, Any], smoke: bool, pins_path: Path) -> List[str]:
    """S2 strict-receipt binding (writer: fill_strict_pin.py; read by serving_run).

    Real run: the intent's pin equals the repo pins file's filled entry, and the strict
    receipt, intent and audit report it names exist with the pinned sha256s and pass the pin
    tool's structured rule (``fill_strict_pin.evidence_problems``: audit status PASS with
    recomputed STRICT_PASS, receipt binds intent and audit bytes, candidate arm = frp3-s12 +
    v8 method). A smoke must carry the placeholder pin and is never checked further.
    """
    pin = intent.get("strict_pin")
    if not isinstance(pin, Mapping):
        return ["intent.strict_pin missing"]
    out = []
    if pin.get("checkpoint_sha256") != CHECKPOINT_SHA256:
        out.append("intent.strict_pin.checkpoint_sha256 is not the frp3-s12 sha256")
    if (pin.get("veto_variant"), pin.get("veto_method")) != ("v8", V8_METHOD):
        out.append("intent.strict_pin veto is not the released v8")
    receipt = pin.get("strict_receipt")
    if smoke:
        if intent.get("strict_pin_placeholder_for_smoke") is not True or not (
            isinstance(receipt, Mapping) and receipt.get("root") == SMOKE_PIN_ROOT
        ):
            out.append("smoke intent does not carry the placeholder pin")
        return out
    if intent.get("strict_pin_placeholder_for_smoke") is not False:
        out.append("a real run served under the smoke placeholder pin")
    repo = (read_pins(pins_path).get("checkpoints") or {}).get(CHECKPOINT_NAME)
    if pin != repo:
        out.append("intent.strict_pin differs from the repo pins file")
    if not isinstance(receipt, Mapping) or receipt.get("verdict") != "STRICT_PASS":
        return out + ["intent.strict_pin.strict_receipt is not a filled STRICT_PASS pin"]
    root = Path(str(receipt.get("root")))
    if not root.is_absolute() or str(root) == SMOKE_PIN_ROOT:
        return out + ["strict_receipt.root is not a real absolute path"]
    files = {}
    for key in STRICT_FILES:
        item = receipt.get(key) or {}
        path = root / str(item.get("path"))
        files[key] = str(item.get("path"))
        if not path.is_file():
            out.append(f"strict {key} missing: {path}")
        elif _sha(path) != item.get("sha256"):
            out.append(f"strict {key} sha256 differs from the pin")
    if not any(f.startswith("strict ") and "missing" in f for f in out):
        try:
            out += [f"strict evidence: {p}" for p in fill_tool.evidence_problems(root, files)]
        except SystemExit as exc:  # unreadable JSON
            out.append(f"strict evidence: {exc}")
    return out


# ---------------------------------------------------------------- criteria


def expected_files(design: Mapping[str, Any]) -> List[str]:
    counts = design["counts"]
    names = [schema.DEFAULT_CHECK]
    names += [
        f"{schema.RECORDS_DIR}/{schema.episode_id(kind, i)}.json"
        for kind, purpose in KIND_PURPOSE.items()
        for i in range(counts[purpose])
    ]
    names += [f"{schema.PARITY_DIR}/{schema.parity_id(i)}.json" for i in range(counts["parity"])]
    return sorted(names)


def preregistration_failures(intent: Mapping[str, Any]) -> List[str]:
    """A non-smoke intent names the pinned protocol, a clean tree and fresh seeds (including
    the strict run's observed seeds)."""
    out = []
    if intent.get("protocol_sha256") != PROTOCOL_SHA256:
        out.append(f"intent.protocol_sha256 {intent.get('protocol_sha256')} is not the pinned one")
    if (intent.get("git") or {}).get("dirty") is not False:
        out.append("intent.git.dirty is not false")
    report = intent.get("seed_report") or {}
    if report.get("disjoint") is not True:
        out.append("intent.seed_report.disjoint is not true")
    if report.get("strict_observed_checked") is not True:
        out.append("intent.seed_report did not check the strict run's seeds")
    return out


def s1_completeness(ev: Mapping[str, Any], design: Mapping[str, Any]) -> List[str]:
    """Counts, statuses, intent binding, seeds and frame accounting (writer: serving_run)."""
    out, intent, receipt = [], ev["intent"], ev["receipt"]
    if intent.get("smoke") is not True:
        out += preregistration_failures(intent)
    if receipt.get("intent_sha256") != ev["intent_sha256"]:
        out.append("receipt.intent_sha256 does not match intent.json")
    out += [f"receipt sha256 mismatch: {rel}" for rel in ev["hash_mismatch"]]
    if sorted(ev["docs"]) != expected_files(design):
        out.append(f"listed files {sorted(ev['docs'])} != expected design files")
    if intent.get("counts") != design["counts"]:
        out.append(f"intent.counts {intent.get('counts')} != {design['counts']}")
    seeds = {
        purpose: [uint32_seed(design["domain"], purpose, i) for i in range(count)]
        for purpose, count in design["counts"].items()
    }
    if intent.get("seeds") != seeds:
        out.append("intent.seeds differ from the re-derived recipe")
    if intent.get("horizon_frames") != design["horizon"]:
        out.append(f"intent.horizon_frames {intent.get('horizon_frames')}")
    if intent.get("parity_horizon_frames") != design["parity_horizon"]:
        out.append(f"intent.parity_horizon_frames {intent.get('parity_horizon_frames')}")
    if intent.get("release_env") != RELEASE_ENV:
        out.append(f"intent.release_env {intent.get('release_env')}")
    horizon = design["horizon"]
    for rec in episode_records(ev):
        where = rec.get("episode_id", "?")
        if set(rec) != set(schema.EPISODE_KEYS):
            out.append(f"{where}: keys differ from schema.EPISODE_KEYS")
            continue
        if rec["status"] != schema.STATUS_COMPLETE or rec["error"] is not None:
            out.append(f"{where}: status {rec['status']!r} error {rec['error']!r}")
        if rec["intent_sha256"] != ev["intent_sha256"] or rec["schema_version"] != schema.SCHEMA:
            out.append(f"{where}: not bound to this intent/schema")
        purpose = KIND_PURPOSE.get(rec["kind"])
        index = rec["episode_index"]
        if purpose is None or not is_int(index) or where != schema.episode_id(rec["kind"], index):
            out.append(f"{where}: kind/index/id mismatch")
            continue
        if rec["world_seed"] != uint32_seed(design["domain"], purpose, index):
            out.append(f"{where}: world_seed is not the recipe seed")
        frames, ended = rec["frames_stepped"], rec["ended_by"]
        if rec["horizon_frames"] != horizon or not is_int(frames):
            out.append(f"{where}: horizon {rec['horizon_frames']} frames {frames!r}")
            continue
        if rec["kind"] == schema.KIND_WATCH:
            ok = ended == schema.ENDED_HORIZON and frames == horizon
        elif ended == schema.ENDED_HUMAN_DEATH:
            ok = 1 <= frames <= horizon and bool((rec["episode"] or {}).get("run_over"))
        else:
            ok = ended == schema.ENDED_HORIZON and frames == horizon
        if not ok:
            out.append(f"{where}: ended_by {ended!r} with {frames} frames")
    return out


def s2_identity(ev: Mapping[str, Any], pins_path: Path = PINS_FILE) -> List[str]:
    """Checkpoint (via the registry), strict pin, v8 wrapper and served-config binding
    (writers: serving_run, served_checkpoint, safety_veto_serving, GameSession)."""
    intent = ev["intent"]
    smoke = intent.get("smoke") is True
    out = wrapper_failures(intent.get("wrapper_identity"), "intent.wrapper_identity")
    out += strict_pin_failures(intent, smoke, pins_path)
    if (intent.get("checkpoint") or {}).get("sha256") != CHECKPOINT_SHA256:
        out.append("intent.checkpoint.sha256 is not the frp3-s12 sha256")
    out += served_checkpoint_failures(intent.get("served_checkpoint"), "intent")
    expected = intent.get("expected") or {}
    checks = {
        "checkpoint_name": expected.get("checkpoint_name") == CHECKPOINT_NAME,
        "checkpoint_sha256": expected.get("checkpoint_sha256") == CHECKPOINT_SHA256,
        "champion_sha256": expected.get("champion_sha256") == CHAMPION_SHA256,
        "v8_source_sha256s": expected.get("v8_source_sha256s") == V8_SOURCE_SHA256S,
        "v8_method": expected.get("v8_method") == V8_METHOD,
        "v8_head": expected.get("v8_head_avoidance") is True
        and expected.get("v8_reference_lambda") is None,
        "parity_profile_digest": expected.get("parity_profile_digest") == PROFILE_DIGEST,
    }
    out += [f"intent.expected.{k}" for k, ok in checks.items() if not ok]
    for rec in episode_records(ev):
        where = rec.get("episode_id", "?")
        ckpt = rec.get("checkpoint") or {}
        if (ckpt.get("sha256"), ckpt.get("session_sha256")) != (CHECKPOINT_SHA256,) * 2:
            out.append(f"{where}: checkpoint sha (file/session) is not frp3-s12")
        out += served_checkpoint_failures(rec.get("served_checkpoint"), where)
        if rec.get("env") != RELEASE_ENV:
            out.append(f"{where}: env {rec.get('env')} is not the swap release env")
        if rec.get("kind") == schema.KIND_WATCH:
            out += wrapper_failures(rec.get("wrapper"), where)
        elif rec.get("wrapper") is not None:
            out.append(f"{where}: Play carries a wrapper identity")
        served = rec.get("served")
        if not isinstance(served, Mapping):
            out.append(f"{where}: no served identity")
            continue
        checks = {
            "mode": served.get("mode") == KIND_PURPOSE.get(rec.get("kind")),
            "obs_spec": served.get("obs_spec") == "vector61",
            "policy_class": served.get("policy_class") == "ApexPolicy",
            "policy_training": served.get("policy_training") is False,
            "epsilon": served.get("epsilon") == 0.0,
            "config": (served.get("config_basename"), served.get("config_sha256")) == SERVED_CONFIG,
            "veto_flags": served.get("veto_flags") == RELEASE_FLAGS,
        }
        out += [f"{where}: served.{name}" for name, ok in checks.items() if not ok]
    return out


def s3_scope(ev: Mapping[str, Any], smoke: bool = False) -> List[str]:
    """v8 wrapper active on exactly the Watch hero (now bound through the swap's strict pin);
    Play unwrapped (writer: install_serving_vetoes). Outside a smoke, the served Watch path
    must have replaced at least one action."""
    out = []
    if not smoke:
        activity = v8audit.watch_activity(ev)
        if not activity["vetoes_applied"]:
            out.append(f"Watch never replaced an action ({activity})")
    for rec in episode_records(ev):
        where = rec.get("episode_id", "?")
        veto = rec.get("veto")
        served = rec.get("served") or {}
        if not isinstance(veto, Mapping) or set(veto) != set(schema.VETO_KEYS):
            out.append(f"{where}: veto block keys")
            continue
        if veto["flags"] != RELEASE_FLAGS or veto["variant_requested"] != "v8":
            out.append(f"{where}: flags {veto['flags']} variant {veto['variant_requested']!r}")
        if rec.get("kind") == schema.KIND_WATCH:
            ids = served.get("snake_ids") or []
            hero = served.get("hero_id")
            checks = {
                "active": veto["active"] is True,
                "scope": veto["scope"] == "watch_hero" and veto["reason"] is None,
                "variant": veto["variant"] == "v8",
                "checkpoint_match": veto["strict_receipt_checkpoint_match"] is True,
                "wrapper_match": veto["strict_receipt_wrapper_match"] is True,
                "hero_is_slot0": bool(ids) and hero == ids[0],
                "wrapped": veto["wrapped_snake_ids"] == [hero],
                "only_hero_has_veto": rec.get("snakes_with_veto") == [hero],
            }
        else:
            checks = {
                "inactive": veto["active"] is False and veto["scope"] is None,
                "reason": veto["reason"] == PLAY_REASON,
                "variant": veto["variant"] is None,
                "wrapped": veto["wrapped_snake_ids"] == [],
                "no_snake_has_veto": rec.get("snakes_with_veto") == [],
                "wrapper": rec.get("wrapper") is None,
            }
        out += [f"{where}: {name}" for name, ok in checks.items() if not ok]
    return out


def s4_counters(ev: Mapping[str, Any]) -> List[str]:
    """The v8 lane's S4 rule, unchanged (the served wrapper is the released v8)."""
    return v8audit.s4_counters(ev)


def s5_parity(ev: Mapping[str, Any], design: Mapping[str, Any], smoke: bool) -> List[str]:
    """Session vs SIMD rollout of the same identity, same seed (writer: run_parity_probe).
    Cell-space traces, v2-shaped counters AND v8 diagnostics must be equal."""
    out = []
    horizon = design["parity_horizon"]
    for probe in parity_records(ev):
        where = probe.get("probe_id", "?")
        if set(probe) != set(schema.PARITY_KEYS):
            out.append(f"{where}: keys differ from schema.PARITY_KEYS")
            continue
        index = probe["probe_index"]
        if probe["status"] != schema.STATUS_COMPLETE or probe["error"] is not None:
            out.append(f"{where}: status {probe['status']!r} error {probe['error']!r}")
            continue
        if probe["intent_sha256"] != ev["intent_sha256"]:
            out.append(f"{where}: not bound to this intent")
        if not is_int(index) or probe["world_seed"] != uint32_seed(
            design["domain"], "parity", index
        ):
            out.append(f"{where}: world_seed is not the recipe seed")
        left, right = probe["session"] or {}, probe["simd"] or {}
        checks = {
            "env": probe["env"] == RELEASE_ENV,
            "horizon": probe["horizon_frames"] == horizon,
            "frames_compared": probe["frames_compared"] == horizon,
            "first_divergence_frame": probe["first_divergence_frame"] is None
            and probe["divergence"] is None,
            "trace_sha256": probe["trace_sha256_equal"] is True
            and left.get("trace_sha256") is not None
            and left.get("trace_sha256") == right.get("trace_sha256"),
            "veto_counters": probe["veto_counters_equal"] is True
            and left.get("counters") == right.get("counters"),
            "diagnostics": probe["diagnostics_equal"] is True
            and left.get("diagnostics") == right.get("diagnostics"),
            "session_checkpoint": left.get("checkpoint_sha256") == CHECKPOINT_SHA256,
            "session_served": not served_checkpoint_failures(left.get("served_checkpoint"), ""),
            "session_variant": left.get("variant") == "v8",
            "session_method": left.get("wrapper_method") == V8_METHOD,
            "simd_method": right.get("wrapper_method") == V8_METHOD,
            "simd_provenance": right.get("provenance") == SIMD_PROVENANCE,
            "simd_profile": smoke
            or (
                right.get("profile_digest") == PROFILE_DIGEST
                and right.get("profile_scored_horizon") == horizon
            ),
            "session_config": left.get("config_sha256") == PARITY_CONFIG_SHA256,
            "wrapped_hero_only": left.get("wrapped_snake_ids") == [0],
            "record_complete": right.get("record_complete") is True,
        }
        out += [f"{where}: {name}" for name, ok in checks.items() if not ok]
        for side, block in (("session", left), ("simd", right)):
            out += counter_failures(block.get("counters"), f"{where}.{side}")
            out += diagnostics_failures(
                block.get("diagnostics"), block.get("counters"), f"{where}.{side}"
            )
        if (left.get("counters") or {}).get("decisions") != left.get("decision_frames"):
            out.append(f"{where}: session decisions != decision frames")
    return out


def s6_defaults(ev: Mapping[str, Any]) -> List[str]:
    """Released defaults (champion + v8), fail-closed refusals and rollbacks intact
    (writer: serving_run.default_check, from real GameSession() builds)."""
    doc = ev["docs"].get(schema.DEFAULT_CHECK)
    if not isinstance(doc, Mapping):
        return ["default_check.json missing"]
    out = []
    if doc.get("intent_sha256") != ev["intent_sha256"]:
        out.append("default_check not bound to this intent")
    if doc.get("flags_from_empty_env") != RELEASE_FLAGS:
        out.append("empty env flags are not the released {watch_hero: on, play_ai: off}")
    if doc.get("variant_from_empty_env") != "v8":
        out.append("empty env variant is not the released v8")
    if doc.get("checkpoint_released_default") != "champion":
        out.append("released checkpoint default is not the champion")
    if doc.get("default_checkpoint_basename") != CHAMPION_BASENAME:
        out.append("served default checkpoint path changed")
    if (doc.get("served_config_basename"), doc.get("served_config_sha256")) != SERVED_CONFIG:
        out.append("served default config changed")
    builds = doc.get("builds") or {}
    if sorted(builds) != sorted(DEFAULT_CASE_EXPECT):
        out.append(f"default_check builds {sorted(builds)}")
    for name, (ckpt, sha, wrapper) in DEFAULT_CASE_EXPECT.items():
        build = builds.get(name) or {}
        if build.get("env") != DEFAULT_CASE_ENV[name]:
            out.append(f"{name}: env {build.get('env')}")
        expected_pins = "unfilled" if name == "frp3_pin_unfilled" else "run"
        if build.get("pins") != expected_pins:
            out.append(f"{name}: pins {build.get('pins')!r}")
        served = build.get("served_checkpoint") or {}
        if served.get("name") != ckpt:
            out.append(f"{name}: served {served.get('name')!r}, expected {ckpt!r}")
        if (served.get("reason") is not None) != (name in REFUSED_CASES):
            out.append(f"{name}: served_checkpoint.reason {served.get('reason')!r}")
        modes = build.get("modes") or {}
        watch, play = modes.get("watch") or {}, modes.get("play") or {}
        if watch.get("mode") != "watch" or play.get("mode") != "play":
            out.append(f"{name}: modes {sorted(modes)}")
        if watch.get("checkpoint_sha256") != sha or play.get("checkpoint_sha256") != sha:
            out.append(f"{name}: session checkpoint sha is not {ckpt}'s")
        if wrapper is None:
            if watch.get("active") is not False or watch.get("snakes_with_veto") != []:
                out.append(f"{name}: Watch is wrapped")
        elif not (
            watch.get("active") is True
            and (watch.get("variant"), watch.get("method")) == wrapper
            and watch.get("snakes_with_veto") == [watch.get("hero_id")]
        ):
            out.append(f"{name}: Watch is not the {wrapper[0]} hero wrapper")
        if play.get("active") is not False or play.get("snakes_with_veto") != []:
            out.append(f"{name}: Play is wrapped")
    if doc.get("pass") is not True:
        out.append("producer default_check.pass is not true")
    return out


# ---------------------------------------------------------------- verdict


def reported(ev: Mapping[str, Any]) -> Dict[str, Any]:
    """Informational only (protocol.md "Reported, not gated"): the v8 lane's block plus the
    served checkpoint paths."""
    out = v8audit.reported(ev)
    out["served_checkpoint_paths"] = sorted(
        {str((r.get("served_checkpoint") or {}).get("path")) for r in episode_records(ev)}
    )
    return out


def audit(root: Path, pins_path: Path = PINS_FILE) -> Dict[str, Any]:
    """Run every criterion; never raises for evidence problems (INVALID instead)."""
    try:
        ev = load_evidence(root)
        smoke = ev["intent"]["smoke"]
        design = DESIGN[smoke]
        criteria = {
            "S1": s1_completeness(ev, design),
            "S2": s2_identity(ev, pins_path),
            "S3": s3_scope(ev, smoke),
            "S4": s4_counters(ev),
            "S5": s5_parity(ev, design, smoke),
            "S6": s6_defaults(ev),
        }
    except Invalid as exc:
        return {"verdict": "INVALID", "reason": str(exc), "serving_path_qualified": False}
    passed = all(not failures for failures in criteria.values())
    return {
        "schema_version": schema.SCHEMA,
        "auditor": "serving_audit.py (stdlib)",
        "intent_sha256": ev["intent_sha256"],
        "smoke": smoke,
        "criteria": {
            name: {"pass": not failures, "failures": failures[:MAX_LISTED], "count": len(failures)}
            for name, failures in criteria.items()
        },
        "verdict": "SERVING_PASS" if passed else "SERVING_FAIL",
        # A smoke never qualifies the serving path, whatever its verdict.
        "serving_path_qualified": bool(passed and not smoke),
        "reported": reported(ev),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Audit the frp3-s12 checkpoint-swap serving run.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = audit(args.root.resolve())
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "audit.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"{result['verdict']} (serving_path_qualified={result['serving_path_qualified']})")
    return {"SERVING_PASS": 0, "SERVING_FAIL": 1}.get(result["verdict"], 2)


if __name__ == "__main__":
    sys.exit(main())
