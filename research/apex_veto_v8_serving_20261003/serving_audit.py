#!/usr/bin/env python3
"""Audit of the Apex v8-veto web-serving run (standard library only).

``python -I serving_audit.py --root <run-out> --out <run-out>/audit``. Exit 0 =
SERVING_PASS, 1 = SERVING_FAIL, 2 = INVALID (evidence missing or unreadable).

It imports nothing from the repo except ``schema.py`` beside it and, by file path, the
v2 serving audit (``research/apex_veto_serving_20261001/serving_audit.py``) for its
evidence loader and its ``SafetyVetoCounters`` partition rule, and the v7 serving audit
(``research/apex_veto_v7_serving_20261002/serving_audit.py``) for its v7-diagnostics rule
(which applies the v5 lane's v5 shape rule to the nested v5 block), applied to v8's nested
v7 block against v7's own probe counters (all stdlib). The pinned identities, the seed
recipe and every v8 rule are re-implemented here; each rule names the code path that writes
the field it checks (protocol.md "Pass criteria").
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
V2_AUDIT = HERE.parent / "apex_veto_serving_20261001" / "serving_audit.py"
V7_AUDIT = HERE.parent / "apex_veto_v7_serving_20261002" / "serving_audit.py"


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


schema = _load("apex_veto_v8_serving_schema", HERE / "schema.py")
v2audit = _load("apex_veto_v8_serving_v2_audit", V2_AUDIT)
v7audit = _load("apex_veto_v8_serving_v7_audit", V7_AUDIT)
Invalid = v2audit.Invalid
uint32_seed = v2audit.uint32_seed
is_int = v2audit.is_int
counter_failures = v2audit.counter_failures
load_evidence = v2audit.load_evidence
episode_records = v2audit.episode_records
parity_records = v2audit.parity_records
v7_diagnostics_failures = v7audit.diagnostics_failures

CHAMPION_SHA256 = "43d4e2c53919dd59416c145cf0ba7c4faf1c7f298eebbb1723146807d747ac93"
CHAMPION_BASENAME = "champion_a5_freespace_20260621.pth"
V8_STRICT_RECEIPT_SHA256 = "29b1f7f6f095cac1990e8f7eafccc806e498cd503bb964bd083b3436ef2b7507"
V8_STRICT_INTENT_SHA256 = "ca4aa97074456fc14281d43e8c999198f39f9576946eee3f953aad7bd56e7e62"
V8_STRICT_AUDIT_REPORT_SHA256 = "72410ec575301def146398c9acff33fdd1e20b45f84ed1e37e452851be8467c6"
V8_SOURCE_PATH = "src/evaluation/safety_veto_v8.py"
V8_SOURCE_SHA256S = {
    "src/evaluation/safety_veto.py": (
        "1b62d15c48987584533efbefdf3fd84e1e3f2e22cfe70dbb82b43ec57169c428"
    ),
    "src/evaluation/safety_veto_v3.py": (
        "ed3a6d860b09afd982bc6c87ea0a86566455dfcf9562133d1772e595b4bb5be1"
    ),
    "src/evaluation/safety_veto_v4.py": (
        "3f0881afb7ff8ff980c9b425ef40134f127cc90f1f40260aac39f2264ed44578"
    ),
    "src/evaluation/safety_veto_v5.py": (
        "d86d084e7778fc514c4932b27f3750f5f11543e3c7afa44571869407870ec86c"
    ),
    "src/evaluation/safety_veto_v6.py": (
        "a6117cb98383f9f28bf8ebcf22d2ef63d7a03995734a36fb96bb1bf8bb1757d5"
    ),
    "src/evaluation/safety_veto_v7.py": (
        "56ff7009ce2e0c4c93b6570336b35d48341757fc53b386d309b00126f67e2980"
    ),
    V8_SOURCE_PATH: "faf3695fa9d60550f1f681c04b8c8aba0447fecaa34cd75f16f75efe91e4ac05",
}
V8_METHOD = "free-space-veto/v8-space-and-head(lambda=8.0)"
V7_METHOD = "free-space-veto/v7-space-preference(lambda=4.0)"
V5_METHOD = "free-space-veto/v5-boost-aware"
V2_METHOD = "free-space-veto/v2-speed-preserving"
# SpaceAndHeadVeto(8.0).descriptor() as the v8 strict intent recorded it (arms.candidate).
DESCRIPTOR = {
    "method": V8_METHOD,
    "free_space_bfs_cap": 160,
    "free_space_min_cap": 32,
    "boost_approximation": "first-cell-and-two-cell-landing",
    "replacement_rule": "v7-space-preference-then-v6-head-avoidance-with-v7-score-replacement",
    "landing_rule": "post-move-body-blocked-except-head/other-snakes-static/v2-cap-need",
    "space_preference_lambda": 8.0,
    "space_preference_rule": (
        "C = v5-eligible actions in the speed mode of v5's choice; argmax score over C; "
        "ties -> v5's choice, then lowest index; unchanged when no_spacious, lambda == 0 "
        "or a landing veto's same-direction normal-speed replacement"
    ),
    "space_score_rule": "(q-max_C q)/max(|max_C q-min_C q|,1e-6) + lambda*min(area,cap)/cap",
    "area_rule": (
        "tail-aware-bfs-from-post-move-head/own-body-released-by-steps/slack-1/"
        "other-snakes-static"
    ),
    "area_cap_rule": (
        "min(max(32, 2*length), 4096, in-bounds cells not covered by other live snakes)"
    ),
    "head_avoidance": True,
    "head_rule": (
        "opponent-head-reach-current-and-3-moves-plus-boost-second-cell/"
        "hero-wins-iff-mechanics-v2-and-post-burn-length-ge-1.15x-opponent-plus-1/"
        "veto-if-v5-eligible-non-risky-alternative"
    ),
    "head_replacement_rule": (
        "R = masked-legal v5-eligible non-head-risky actions other than v7's choice, in its "
        "speed mode, else the other mode; argmax over R of (q-max_R q)/max(|max_R q-min_R q|,"
        "1e-6) + lambda*min(area,cap)/cap; ties -> highest-q member of R (lowest index), then "
        "lowest index"
    ),
}
SERVED_CONFIG = (
    "mechanics_v2.yaml",
    "c2db7607915f70eaa46489a48598db65c4593cf039a166f37842423bb1479911",
)
PARITY_CONFIG_SHA256 = "4146baa3a06102b8afd627b1fba8384e9a2f47aaac4a9bc96292c3eb71715aa5"
RELEASE_ENV = {
    "SNAKE_SERVE_VETO_WATCH_HERO": None,
    "SNAKE_SERVE_VETO_PLAY_AI": None,
    "SNAKE_SERVE_VETO_VARIANT": "v8",
}
RELEASE_FLAGS = {"watch_hero": True, "play_ai": False}
PLAY_REASON = "no serving veto flag applies to play mode"
# protocol.md as pre-registered for the real run. A non-smoke intent must carry exactly
# this sha256 (a later protocol edit cannot audit as SERVING_PASS).
PROTOCOL_SHA256 = "4bdb4c0ab3cb5e5c8dc52331725672577a03ec92e7e4cb37d7e10c593444853f"
DESIGN = {
    False: {
        "domain": "apex-veto-v8-web-serving-v1",
        "counts": {"watch": 25, "play": 25, "parity": 2},
        "horizon": 5000,
        "parity_horizon": 5000,
    },
    True: {
        "domain": "apex-veto-v8-web-serving-smoke-v1",
        "counts": {"watch": 1, "play": 1, "parity": 1},
        "horizon": 500,
        "parity_horizon": 200,
    },
}
# S6: the released default (v7) and the rollbacks, case -> (Watch variant, method).
DEFAULT_CASE_WATCH = {
    "empty_env": ("v7", V7_METHOD),
    "variant_v7": ("v7", V7_METHOD),
    "variant_v5": ("v5", V5_METHOD),
    "variant_v2": ("v2", V2_METHOD),
}
OFF_CASE = "watch_off_variant_v8"
KIND_PURPOSE = {schema.KIND_WATCH: "watch", schema.KIND_PLAY: "play"}
MAX_LISTED = 50


# ---------------------------------------------------------------- reusable rules


def wrapper_failures(wrapper: Any, where: str) -> List[str]:
    """S2 v8 identity. Writer: ``safety_veto_serving.wrapper_identity_v8``."""
    if not isinstance(wrapper, Mapping):
        return [f"{where}: no wrapper identity"]
    expected = {
        "method": V8_METHOD,
        "descriptor": DESCRIPTOR,
        "source_path": V8_SOURCE_PATH,
        "source_sha256": V8_SOURCE_SHA256S[V8_SOURCE_PATH],
        "source_sha256s": V8_SOURCE_SHA256S,
    }
    out = [f"{where}: wrapper.{k}" for k, v in expected.items() if wrapper.get(k) != v]
    if set(wrapper) != set(expected):
        out.append(f"{where}: wrapper keys {sorted(wrapper)}")
    return out


def diagnostics_failures(diag: Any, counters: Any, where: str) -> List[str]:
    """S4 v8 identities (writer: ``SpaceAndHeadVeto.apply``/``decide``/``_head_layer``; the
    identities are the ``SpaceAndHeadCounters`` docstring's and
    ``tests/test_safety_veto_v8.py``'s).

    The served v8 must have the head layer on and no reference lambda (so no reference
    decisions). The nested v7 block must satisfy the v7 lane's rule, including its probe
    mirrors against v7's own probe counters (v8 runs v7 unchanged before its head layer);
    v7's probe counters must satisfy the v2 partition. ``counters=None`` checks shape only.
    """
    keys = set(schema.DIAGNOSTIC_KEYS) | {
        schema.HEAD_AVOIDANCE_KEY,
        schema.REFERENCE_LAMBDA_KEY,
        schema.V7_NESTED_KEY,
        schema.V7_PROBE_COUNTERS_KEY,
    }
    if not isinstance(diag, Mapping) or set(diag) != keys:
        return [f"{where}: diagnostics keys"]
    if not all(is_int(diag[k]) and diag[k] >= 0 for k in schema.DIAGNOSTIC_KEYS):
        return [f"{where}: non-integer or negative diagnostics"]
    out = []
    if diag[schema.HEAD_AVOIDANCE_KEY] is not True:
        out.append(f"{where}: diagnostics head_avoidance is not true")
    if diag[schema.REFERENCE_LAMBDA_KEY] is not None:
        out.append(f"{where}: diagnostics reference_lambda is set (not the gated candidate)")
    probe = diag[schema.V7_PROBE_COUNTERS_KEY]
    probe_failures = counter_failures(probe, f"{where}.v7_probe_counters")
    out += probe_failures
    nested = diag[schema.V7_NESTED_KEY]
    nested_failures = v7_diagnostics_failures(
        nested, None if probe_failures else probe, f"{where}.v7"
    )
    out += nested_failures
    if out:
        return out
    d, v7 = diag, nested
    vetoes = d["head_risky_vetoes"]
    rules = {
        "partition": d["decisions"] == d["kept"] + d["vetoes_applied"] + d["no_spacious"],
        "v7_decisions": v7["decisions"] == d["decisions"],
        "kept_vs_v7": d["kept"] == v7["kept"] - d["head_vetoes_of_v7_kept"],
        "vetoes_vs_v7": d["vetoes_applied"] == v7["vetoes_applied"] + d["head_vetoes_of_v7_kept"],
        "no_spacious_vs_v7": d["no_spacious"] == v7["no_spacious"],
        "head_checks": d["head_checks"] == d["decisions"] - d["no_spacious"],
        "head_risky_le_checks": d["head_risky_decisions"] <= d["head_checks"],
        "head_split": d["head_risky_decisions"] == vetoes + d["head_risky_kept_no_alternative"],
        "differs_from_v7": d["action_differs_from_v7"] == vetoes,
        "head_of_v7_kept": d["head_vetoes_of_v7_kept"] <= vetoes,
        "head_of_v7_rerank": d["head_vetoes_of_v7_rerank"] <= vetoes,
        "head_speed_switched": d["head_vetoes_speed_switched"] <= vetoes,
        "head_space_differs": d["head_vetoes_space_differs_from_highest_q"] <= vetoes,
        "waived_le_checks": d["head_risk_waived_hero_wins"] <= d["head_checks"],
        "no_reference": d["reference_decisions"] == 0 and d["action_differs_from_reference"] == 0,
    }
    if counters is None:
        pass
    elif isinstance(counters, Mapping) and not counter_failures(counters, ""):
        rules.update(
            {
                "mirror_decisions": d["decisions"] == counters["decisions"],
                "mirror_kept": d["kept"] == counters["kept_base"],
                "mirror_vetoes": d["vetoes_applied"] == counters["vetoes_applied"],
                "mirror_no_spacious": d["no_spacious"] == counters["fallback_no_spacious"],
            }
        )
    else:
        rules["counters"] = False
    return [f"{where}: diagnostics {name}" for name, ok in rules.items() if not ok]


def diagnostics_total(per_snake: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """Re-derived ``veto.diagnostics_total``: sums of the v8 integer keys, the nested v7
    block (``area_cap_max`` the maximum; v5 nested, summed) and v7's probe counters."""
    values = list(per_snake.values())
    out: Dict[str, Any] = {key: sum(d[key] for d in values) for key in schema.DIAGNOSTIC_KEYS}
    v7 = [d[schema.V7_NESTED_KEY] for d in values]
    nested: Dict[str, Any] = {}
    for key in schema.V7_DIAGNOSTIC_KEYS:
        column = [d[key] for d in v7]
        if key in schema.V7_DIAGNOSTIC_MAX_KEYS:
            nested[key] = max(column) if column else 0
        else:
            nested[key] = sum(column)
    nested[schema.V5_NESTED_KEY] = {
        key: sum(d[schema.V5_NESTED_KEY][key] for d in v7) for key in schema.V5_DIAGNOSTIC_KEYS
    }
    out[schema.V7_NESTED_KEY] = nested
    out[schema.V7_PROBE_COUNTERS_KEY] = {
        key: sum(d[schema.V7_PROBE_COUNTERS_KEY][key] for d in values)
        for key in schema.COUNTER_KEYS
    }
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
    """A non-smoke intent names the pinned protocol, a clean tree and fresh seeds."""
    out = []
    if intent.get("protocol_sha256") != PROTOCOL_SHA256:
        out.append(f"intent.protocol_sha256 {intent.get('protocol_sha256')} is not the pinned one")
    if (intent.get("git") or {}).get("dirty") is not False:
        out.append("intent.git.dirty is not false")
    if (intent.get("seed_report") or {}).get("disjoint") is not True:
        out.append("intent.seed_report.disjoint is not true")
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


def s2_identity(ev: Mapping[str, Any]) -> List[str]:
    """Checkpoint, v8 wrapper and served-config binding (writers: harness,
    safety_veto_serving, GameSession). Play is served unwrapped: its wrapper is None."""
    intent = ev["intent"]
    out = wrapper_failures(intent.get("wrapper_identity"), "intent.wrapper_identity")
    if (intent.get("checkpoint") or {}).get("sha256") != CHAMPION_SHA256:
        out.append("intent.checkpoint.sha256 is not the pinned champion")
    expected = intent.get("expected") or {}
    if expected.get("v8_source_sha256s") != V8_SOURCE_SHA256S:
        out.append("intent.expected.v8_source_sha256s differ from the v8 receipt pins")
    if expected.get("v8_strict_receipt_sha256") != V8_STRICT_RECEIPT_SHA256:
        out.append("intent.expected.v8_strict_receipt_sha256 is not the v8 STRICT_PASS receipt")
    if expected.get("v8_strict_intent_sha256") != V8_STRICT_INTENT_SHA256:
        out.append("intent.expected.v8_strict_intent_sha256 is not the v8 strict intent")
    if expected.get("v8_strict_audit_report_sha256") != V8_STRICT_AUDIT_REPORT_SHA256:
        out.append("intent.expected.v8_strict_audit_report_sha256 is not the v8 strict audit")
    if expected.get("v8_head_avoidance") is not True or expected.get("v8_reference_lambda", 0):
        out.append("intent.expected does not name head layer on and no reference lambda")
    for rec in episode_records(ev):
        where = rec.get("episode_id", "?")
        ckpt = rec.get("checkpoint") or {}
        if ckpt.get("sha256") != CHAMPION_SHA256 or ckpt.get("session_sha256") != CHAMPION_SHA256:
            out.append(f"{where}: checkpoint sha (file/session) is not the pinned champion")
        if rec.get("env") != RELEASE_ENV:
            out.append(f"{where}: env {rec.get('env')} is not the v8 release env")
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


def watch_activity(ev: Mapping[str, Any]) -> Dict[str, Any]:
    """Summed Watch veto activity (writer: SpaceAndHeadVeto). ``vetoes_applied`` counts
    every replaced action (what S3 counts); the split is reported only."""
    watch = [r for r in episode_records(ev) if r.get("kind") == schema.KIND_WATCH]
    totals = [(r.get("veto") or {}).get("total") or {} for r in watch]
    diags = [(r.get("veto") or {}).get("diagnostics_total") or {} for r in watch]
    applied = [t.get("vetoes_applied") for t in totals]
    readable = all(is_int(v) for v in applied)
    v7 = [d.get(schema.V7_NESTED_KEY) or {} for d in diags]
    v5 = [d.get(schema.V5_NESTED_KEY) or {} for d in v7]

    def total(rows: Sequence[Mapping[str, Any]], key: str) -> Optional[int]:
        values = [row.get(key) for row in rows]
        return sum(values) if values and all(is_int(v) for v in values) else None

    return {
        "episodes": len(watch),
        "vetoes_applied": sum(applied) if readable else None,
        "head_risky_decisions": total(diags, "head_risky_decisions"),
        "head_risky_vetoes": total(diags, "head_risky_vetoes"),
        "head_risky_kept_no_alternative": total(diags, "head_risky_kept_no_alternative"),
        "head_vetoes_of_v7_kept": total(diags, "head_vetoes_of_v7_kept"),
        "head_risk_waived_hero_wins": total(diags, "head_risk_waived_hero_wins"),
        "v7_vetoes_applied": total(v7, "vetoes_applied"),
        "v7_rerank_changes": total(v7, "rerank_changes"),
        "v7_rerank_changes_of_v5_kept": total(v7, "rerank_changes_of_v5_kept"),
        "v7_rerank_decisions": total(v7, "rerank_decisions"),
        "v5_vetoes_applied": total(v5, "vetoes_applied"),
        "v5_boost_landing_vetoes": total(v5, "boost_landing_vetoes"),
    }


def s3_scope(ev: Mapping[str, Any], smoke: bool = False) -> List[str]:
    """v8 wrapper active on exactly the Watch hero; Play unwrapped (writer:
    install_serving_vetoes). Outside a smoke, the served Watch path must have replaced at
    least one action (summed ``vetoes_applied`` > 0; see watch_activity)."""
    out = []
    if not smoke:
        activity = watch_activity(ev)
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
    """Counters, v8 diagnostics and decision frames (writers: SpaceAndHeadVeto via
    AISnake.update, greedy path only; serving_run.DecisionCounter). Play: nothing wrapped,
    so no counters and no decision frames."""
    out = []
    for rec in episode_records(ev):
        where = rec.get("episode_id", "?")
        veto, frames = rec.get("veto") or {}, rec.get("decision_frames") or {}
        per_snake, per_frames = veto.get("per_snake"), frames.get("per_snake")
        diags = veto.get("diagnostics_per_snake")
        wrapped = [str(i) for i in veto.get("wrapped_snake_ids") or []]
        if not all(isinstance(m, Mapping) for m in (per_snake, per_frames, diags)):
            out.append(f"{where}: missing per-snake counters, diagnostics or decision frames")
            continue
        if any(sorted(m) != sorted(wrapped) for m in (per_snake, per_frames, diags)):
            out.append(f"{where}: per-snake keys differ from wrapped ids")
            continue
        if rec.get("kind") == schema.KIND_PLAY:
            if wrapped or frames.get("total") != 0:
                out.append(f"{where}: Play has wrapped decisions")
            continue
        if not wrapped:
            out.append(f"{where}: Watch has no wrapped snake")
            continue
        for sid in wrapped:
            failures = counter_failures(per_snake[sid], f"{where}[{sid}]")
            failures += diagnostics_failures(diags[sid], per_snake[sid], f"{where}[{sid}]")
            out += failures
            if not failures and per_snake[sid]["decisions"] != per_frames[sid]:
                out.append(f"{where}[{sid}]: decisions != decision frames {per_frames[sid]}")
        if any(counter_failures(per_snake[s], "") for s in wrapped):
            continue
        total = {k: sum(per_snake[s][k] for s in wrapped) for k in schema.COUNTER_KEYS}
        if veto.get("total") != total:
            out.append(f"{where}: veto.total is not the per-snake sum")
        if not any(diagnostics_failures(diags[s], None, "") for s in wrapped):
            if veto.get("diagnostics_total") != diagnostics_total({s: diags[s] for s in wrapped}):
                out.append(f"{where}: veto.diagnostics_total is not the per-snake aggregate")
        if frames.get("total") != sum(per_frames.values()):
            out.append(f"{where}: decision_frames.total is not the per-snake sum")
        if rec.get("frames_stepped", 0) > 0 and total["decisions"] <= 0:
            out.append(f"{where}: stepped frames but no wrapped decision")
    return out


def s5_parity(ev: Mapping[str, Any], design: Mapping[str, Any], smoke: bool) -> List[str]:
    """Session vs rollout under the v8 strict installer, same seed (writer:
    run_parity_probe). Counters AND v8 diagnostics (v7 and v5 nested, v7 probe counters)
    must be equal."""
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
        left, right = probe["session"] or {}, probe["rollout"] or {}
        checks = {
            "env": probe["env"] == RELEASE_ENV,
            "horizon": probe["horizon_frames"] == horizon,
            "frames_compared": probe["frames_compared"] == horizon,
            "first_divergence_frame": probe["first_divergence_frame"] is None,
            "trace_sha256": probe["trace_sha256_equal"] is True
            and left.get("trace_sha256") is not None
            and left.get("trace_sha256") == right.get("trace_sha256"),
            "veto_counters": probe["veto_counters_equal"] is True
            and left.get("counters") == right.get("counters"),
            "diagnostics": probe["diagnostics_equal"] is True
            and left.get("diagnostics") == right.get("diagnostics"),
            "session_variant": left.get("variant") == "v8",
            "session_method": left.get("wrapper_method") == V8_METHOD,
            "rollout_method": right.get("wrapper_method") == V8_METHOD,
            "session_config": left.get("config_sha256") == PARITY_CONFIG_SHA256,
            "wrapped_hero_only": left.get("wrapped_snake_ids") == [0],
            "record_complete": right.get("record_complete") is True or smoke,
        }
        out += [f"{where}: {name}" for name, ok in checks.items() if not ok]
        for side, block in (("session", left), ("rollout", right)):
            out += counter_failures(block.get("counters"), f"{where}.{side}")
            out += diagnostics_failures(
                block.get("diagnostics"), block.get("counters"), f"{where}.{side}"
            )
        if (left.get("counters") or {}).get("decisions") != left.get("decision_frames"):
            out.append(f"{where}: session decisions != decision frames")
    return out


def s6_defaults(ev: Mapping[str, Any]) -> List[str]:
    """Released v7 default and the rollbacks intact (writer: serving_run.default_check)."""
    doc = ev["docs"].get(schema.DEFAULT_CHECK)
    if not isinstance(doc, Mapping):
        return ["default_check.json missing"]
    out = []
    if doc.get("intent_sha256") != ev["intent_sha256"]:
        out.append("default_check not bound to this intent")
    if doc.get("flags_from_empty_env") != RELEASE_FLAGS:
        out.append("empty env flags are not the released {watch_hero: on, play_ai: off}")
    if doc.get("variant_from_empty_env") != "v7":
        out.append("empty env variant is not the released v7")
    if doc.get("default_checkpoint_basename") != CHAMPION_BASENAME:
        out.append("served default checkpoint changed")
    if (doc.get("served_config_basename"), doc.get("served_config_sha256")) != SERVED_CONFIG:
        out.append("served default config changed")
    builds = doc.get("builds") or {}
    if sorted(builds) != sorted(list(DEFAULT_CASE_WATCH) + [OFF_CASE]):
        out.append(f"default_check builds {sorted(builds)}")
    for name, (variant, method) in DEFAULT_CASE_WATCH.items():
        modes = (builds.get(name) or {}).get("modes") or {}
        watch, play = modes.get("watch") or {}, modes.get("play") or {}
        if not (
            watch.get("mode") == "watch"
            and watch.get("active") is True
            and watch.get("variant") == variant
            and watch.get("method") == method
            and watch.get("snakes_with_veto") == [watch.get("hero_id")]
        ):
            out.append(f"{name}: Watch is not the {variant} hero wrapper")
        if play.get("mode") != "play" or play.get("active") is not False:
            out.append(f"{name}: Play build is not unwrapped")
        if play.get("snakes_with_veto") != []:
            out.append(f"{name}: Play has wrapped snakes")
    off = (builds.get(OFF_CASE) or {}).get("modes") or {}
    for mode in ("watch", "play"):
        state = off.get(mode) or {}
        if state.get("mode") != mode or state.get("active") is not False:
            out.append(f"{OFF_CASE}: {mode} build is not unwrapped")
        if state.get("snakes_with_veto") != []:
            out.append(f"{OFF_CASE}: {mode} has wrapped snakes")
    if doc.get("pass") is not True:
        out.append("producer default_check.pass is not true")
    return out


# ---------------------------------------------------------------- verdict


def reported(ev: Mapping[str, Any]) -> Dict[str, Any]:
    """Informational only (protocol.md "Reported, not gated")."""
    out: Dict[str, Any] = {}
    for kind in KIND_PURPOSE:
        recs = [r for r in episode_records(ev) if r.get("kind") == kind]
        totals = [(r.get("veto") or {}).get("total") or {} for r in recs]
        decisions = sum(t.get("decisions", 0) for t in totals)
        vetoes = sum(t.get("vetoes_applied", 0) for t in totals)
        ended: Dict[str, int] = {}
        for r in recs:
            ended[str(r.get("ended_by"))] = ended.get(str(r.get("ended_by")), 0) + 1
        out[kind] = {
            "episodes": len(recs),
            "frames": sum(r.get("frames_stepped") or 0 for r in recs),
            "decisions": decisions,
            "vetoes_applied": vetoes,
            "veto_rate": vetoes / decisions if decisions else None,
            "ended_by": ended,
        }
        if kind == schema.KIND_WATCH:
            deaths = [((r.get("episode") or {}).get("hero_deaths")) for r in recs]
            out[kind]["hero_deaths"] = sum(d for d in deaths if is_int(d))
            seconds, calls = 0.0, 0
            for r in recs:
                for sid, t in ((r.get("veto") or {}).get("diagnostics_timing") or {}).items():
                    total = (t or {}).get("apply_seconds_total")
                    if isinstance(total, (int, float)):
                        seconds += float(total)
                        calls += int(
                            ((r.get("veto") or {}).get("per_snake") or {})
                            .get(sid, {})
                            .get("decisions", 0)
                        )
            out[kind]["mean_apply_seconds"] = seconds / calls if calls else None
    out["watch_activity"] = watch_activity(ev)
    out["parity"] = parity_exercise(ev)
    out["current_tree_v8_source_sha256s"] = {
        rel: (v2audit.schema.sha256_file(REPO / rel) if (REPO / rel).is_file() else None)
        for rel in V8_SOURCE_SHA256S
    }
    return out


def parity_exercise(ev: Mapping[str, Any]) -> Dict[str, Any]:
    """Which veto branches the parity probes exercised (reported, not gated).

    ``head-exercised`` = a v8 head veto replaced an action on the session side;
    ``rerank-exercised`` = a v7 re-rank changed an action; ``exercised`` = some
    replacement ran; otherwise S5's equality covers only the keep/fallback path.
    """
    per_probe = {}
    for probe in parity_records(ev):
        left = probe.get("session") or {}
        diag = left.get("diagnostics") or {}
        vetoes = (left.get("counters") or {}).get("vetoes_applied")
        head = diag.get("head_risky_vetoes")
        changes = (diag.get(schema.V7_NESTED_KEY) or {}).get("rerank_changes")
        per_probe[str(probe.get("probe_id"))] = {
            "vetoes_applied": vetoes if is_int(vetoes) else None,
            "head_risky_vetoes": head if is_int(head) else None,
            "v7_rerank_changes": changes if is_int(changes) else None,
        }
    vetoes = sum(v["vetoes_applied"] or 0 for v in per_probe.values())
    head = sum(v["head_risky_vetoes"] or 0 for v in per_probe.values())
    changes = sum(v["v7_rerank_changes"] or 0 for v in per_probe.values())
    if head:
        label = "head-exercised"
    elif changes:
        label = "rerank-exercised"
    else:
        label = "exercised" if vetoes else "non-exercising"
    return {
        "per_probe": per_probe,
        "vetoes_applied_total": vetoes,
        "head_risky_vetoes_total": head,
        "v7_rerank_changes_total": changes,
        "label": label,
    }


def audit(root: Path) -> Dict[str, Any]:
    """Run every criterion; never raises for evidence problems (INVALID instead)."""
    try:
        ev = load_evidence(root)
        smoke = ev["intent"]["smoke"]
        design = DESIGN[smoke]
        criteria = {
            "S1": s1_completeness(ev, design),
            "S2": s2_identity(ev),
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
    parser = argparse.ArgumentParser(description="Audit an Apex v8-veto web-serving run.")
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
