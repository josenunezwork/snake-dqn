#!/usr/bin/env python3
"""Audit of the Apex v7-veto web-serving run (standard library only).

``python -I serving_audit.py --root <run-out> --out <run-out>/audit``. Exit 0 =
SERVING_PASS, 1 = SERVING_FAIL, 2 = INVALID (evidence missing or unreadable).

It imports nothing from the repo except ``schema.py`` beside it and, by file path, the
v2 serving audit (``research/apex_veto_serving_20261001/serving_audit.py``) for its
evidence loader and its ``SafetyVetoCounters`` partition rule, and the v5 serving audit
(``research/apex_veto_v5_serving_20261001/serving_audit.py``) for its v5-diagnostics shape
rule, applied to the nested v5 block (all stdlib). The pinned identities, the seed recipe
and every v7 rule are re-implemented here; each rule names the code path that writes the
field it checks (protocol.md "Pass criteria").
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
V5_AUDIT = HERE.parent / "apex_veto_v5_serving_20261001" / "serving_audit.py"


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


schema = _load("apex_veto_v7_serving_schema", HERE / "schema.py")
v2audit = _load("apex_veto_v7_serving_v2_audit", V2_AUDIT)
v5audit = _load("apex_veto_v7_serving_v5_audit", V5_AUDIT)
Invalid = v2audit.Invalid
uint32_seed = v2audit.uint32_seed
is_int = v2audit.is_int
counter_failures = v2audit.counter_failures
load_evidence = v2audit.load_evidence
episode_records = v2audit.episode_records
parity_records = v2audit.parity_records
v5_diagnostics_failures = v5audit.diagnostics_failures

CHAMPION_SHA256 = "43d4e2c53919dd59416c145cf0ba7c4faf1c7f298eebbb1723146807d747ac93"
CHAMPION_BASENAME = "champion_a5_freespace_20260621.pth"
V7_STRICT_RECEIPT_SHA256 = "86ee36791e33ecad06bd525e8190d97f1c012ac6e3ede33ba05abe544f750422"
V7_SOURCE_PATH = "src/evaluation/safety_veto_v7.py"
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
    V7_SOURCE_PATH: "56ff7009ce2e0c4c93b6570336b35d48341757fc53b386d309b00126f67e2980",
}
V7_METHOD = "free-space-veto/v7-space-preference(lambda=4.0)"
V5_METHOD = "free-space-veto/v5-boost-aware"
V2_METHOD = "free-space-veto/v2-speed-preserving"
# SpacePreferenceVeto(4.0).descriptor() as the v7 strict intent recorded it.
DESCRIPTOR = {
    "method": V7_METHOD,
    "free_space_bfs_cap": 160,
    "free_space_min_cap": 32,
    "boost_approximation": "first-cell-and-two-cell-landing",
    "replacement_rule": (
        "landing-veto-same-direction-normal-speed-first/"
        "highest-q-eligible-same-speed-mode-then-other"
    ),
    "landing_rule": "post-move-body-blocked-except-head/other-snakes-static/v2-cap-need",
    "space_preference_lambda": 4.0,
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
}
SERVED_CONFIG = (
    "mechanics_v2.yaml",
    "c2db7607915f70eaa46489a48598db65c4593cf039a166f37842423bb1479911",
)
PARITY_CONFIG_SHA256 = "4146baa3a06102b8afd627b1fba8384e9a2f47aaac4a9bc96292c3eb71715aa5"
RELEASE_ENV = {
    "SNAKE_SERVE_VETO_WATCH_HERO": None,
    "SNAKE_SERVE_VETO_PLAY_AI": None,
    "SNAKE_SERVE_VETO_VARIANT": "v7",
}
RELEASE_FLAGS = {"watch_hero": True, "play_ai": False}
PLAY_REASON = "no serving veto flag applies to play mode"
# protocol.md as pre-registered for the real run. A non-smoke intent must carry exactly
# this sha256 (a later protocol edit cannot audit as SERVING_PASS).
PROTOCOL_SHA256 = "6028193ecda233cd84be7d63a0896362690e5e54f02fafc4198d27500124a6da"
DESIGN = {
    False: {
        "domain": "apex-veto-v7-web-serving-v1",
        "counts": {"watch": 25, "play": 25, "parity": 2},
        "horizon": 5000,
        "parity_horizon": 5000,
    },
    True: {
        "domain": "apex-veto-v7-web-serving-smoke-v1",
        "counts": {"watch": 1, "play": 1, "parity": 1},
        "horizon": 500,
        "parity_horizon": 200,
    },
}
# S6: the released default (v5) and the rollbacks, case -> (Watch variant, method).
DEFAULT_CASE_WATCH = {
    "empty_env": ("v5", V5_METHOD),
    "variant_v5": ("v5", V5_METHOD),
    "variant_v2": ("v2", V2_METHOD),
}
OFF_CASE = "watch_off_variant_v7"
KIND_PURPOSE = {schema.KIND_WATCH: "watch", schema.KIND_PLAY: "play"}
MAX_LISTED = 50


# ---------------------------------------------------------------- reusable rules


def wrapper_failures(wrapper: Any, where: str) -> List[str]:
    """S2 v7 identity. Writer: ``safety_veto_serving.wrapper_identity_v7``."""
    if not isinstance(wrapper, Mapping):
        return [f"{where}: no wrapper identity"]
    expected = {
        "method": V7_METHOD,
        "descriptor": DESCRIPTOR,
        "source_path": V7_SOURCE_PATH,
        "source_sha256": V7_SOURCE_SHA256S[V7_SOURCE_PATH],
        "source_sha256s": V7_SOURCE_SHA256S,
    }
    out = [f"{where}: wrapper.{k}" for k, v in expected.items() if wrapper.get(k) != v]
    if set(wrapper) != set(expected):
        out.append(f"{where}: wrapper keys {sorted(wrapper)}")
    return out


def diagnostics_failures(diag: Any, counters: Any, where: str) -> List[str]:
    """S4 v7 identities (writer: ``SpacePreferenceVeto.apply``/``decide``; the identities
    are the ``SpacePreferenceCounters`` docstring's and ``tests/test_safety_veto_v7.py``'s).

    The nested v5 block must satisfy the v5 lane's shape rules (its probe mirrors do not
    apply: v7 changes outcomes after v5 decides). ``counters=None`` checks shape only.
    """
    keys = set(schema.DIAGNOSTIC_KEYS) | {schema.V5_NESTED_KEY}
    if not isinstance(diag, Mapping) or set(diag) != keys:
        return [f"{where}: diagnostics keys"]
    if not all(is_int(diag[k]) and diag[k] >= 0 for k in schema.DIAGNOSTIC_KEYS):
        return [f"{where}: non-integer or negative diagnostics"]
    nested = diag[schema.V5_NESTED_KEY]
    out = v5_diagnostics_failures(nested, None, f"{where}.v5")
    if out:
        return out
    d, v5 = diag, nested
    changes = d["rerank_changes"]
    rules = {
        "partition": d["decisions"] == d["kept"] + d["vetoes_applied"] + d["no_spacious"],
        "v5_decisions": v5["decisions"] == d["decisions"],
        "kept_vs_v5": d["kept"] == v5["kept"] - d["rerank_changes_of_v5_kept"],
        "vetoes_vs_v5": d["vetoes_applied"]
        == v5["vetoes_applied"] + d["rerank_changes_of_v5_kept"],
        "no_spacious_vs_v5": d["no_spacious"] == v5["no_spacious"],
        "rerank_split": d["rerank_decisions"] == d["rerank_pruned"] + d["rerank_scored"],
        "rerank_changes": changes <= d["rerank_scored"],
        "changes_of_v5_kept": d["rerank_changes_of_v5_kept"] <= changes,
        "changes_boost_mode": d["rerank_changes_boost_mode"] <= changes,
        "tail_release_driven": d["rerank_changes_tail_release_driven"] <= changes,
        "static_area": d["static_area_evaluations"] == 2 * changes,
        "cap_hits": d["area_cap_hits"] <= d["area_evaluations"],
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
    """Re-derived ``veto.diagnostics_total`` (sums; ``area_cap_max`` the maximum)."""
    values = list(per_snake.values())
    out: Dict[str, Any] = {}
    for key in schema.DIAGNOSTIC_KEYS:
        column = [d[key] for d in values]
        if key in schema.DIAGNOSTIC_MAX_KEYS:
            out[key] = max(column) if column else 0
        else:
            out[key] = sum(column)
    out[schema.V5_NESTED_KEY] = {
        key: sum(d[schema.V5_NESTED_KEY][key] for d in values) for key in schema.V5_DIAGNOSTIC_KEYS
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
    """Checkpoint, v7 wrapper and served-config binding (writers: harness,
    safety_veto_serving, GameSession). Play is served unwrapped: its wrapper is None."""
    intent = ev["intent"]
    out = wrapper_failures(intent.get("wrapper_identity"), "intent.wrapper_identity")
    if (intent.get("checkpoint") or {}).get("sha256") != CHAMPION_SHA256:
        out.append("intent.checkpoint.sha256 is not the pinned champion")
    expected = intent.get("expected") or {}
    if expected.get("v7_source_sha256s") != V7_SOURCE_SHA256S:
        out.append("intent.expected.v7_source_sha256s differ from the v7 receipt pins")
    if expected.get("v7_strict_receipt_sha256") != V7_STRICT_RECEIPT_SHA256:
        out.append("intent.expected.v7_strict_receipt_sha256 is not the v7 STRICT_PASS receipt")
    for rec in episode_records(ev):
        where = rec.get("episode_id", "?")
        ckpt = rec.get("checkpoint") or {}
        if ckpt.get("sha256") != CHAMPION_SHA256 or ckpt.get("session_sha256") != CHAMPION_SHA256:
            out.append(f"{where}: checkpoint sha (file/session) is not the pinned champion")
        if rec.get("env") != RELEASE_ENV:
            out.append(f"{where}: env {rec.get('env')} is not the v7 release env")
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
    """Summed Watch veto activity (writer: SpacePreferenceVeto). ``vetoes_applied`` counts
    every replaced action (what S3 counts); the split is reported only."""
    watch = [r for r in episode_records(ev) if r.get("kind") == schema.KIND_WATCH]
    totals = [(r.get("veto") or {}).get("total") or {} for r in watch]
    diags = [(r.get("veto") or {}).get("diagnostics_total") or {} for r in watch]
    applied = [t.get("vetoes_applied") for t in totals]
    readable = all(is_int(v) for v in applied)
    nested = [d.get(schema.V5_NESTED_KEY) or {} for d in diags]

    def total(rows: Sequence[Mapping[str, Any]], key: str) -> Optional[int]:
        values = [row.get(key) for row in rows]
        return sum(values) if values and all(is_int(v) for v in values) else None

    return {
        "episodes": len(watch),
        "vetoes_applied": sum(applied) if readable else None,
        "rerank_changes": total(diags, "rerank_changes"),
        "rerank_changes_of_v5_kept": total(diags, "rerank_changes_of_v5_kept"),
        "rerank_changes_tail_release_driven": total(diags, "rerank_changes_tail_release_driven"),
        "rerank_decisions": total(diags, "rerank_decisions"),
        "v5_vetoes_applied": total(nested, "vetoes_applied"),
        "v5_boost_landing_vetoes": total(nested, "boost_landing_vetoes"),
        "area_evaluations": total(diags, "area_evaluations"),
    }


def s3_scope(ev: Mapping[str, Any], smoke: bool = False) -> List[str]:
    """v7 wrapper active on exactly the Watch hero; Play unwrapped (writer:
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
        if veto["flags"] != RELEASE_FLAGS or veto["variant_requested"] != "v7":
            out.append(f"{where}: flags {veto['flags']} variant {veto['variant_requested']!r}")
        if rec.get("kind") == schema.KIND_WATCH:
            ids = served.get("snake_ids") or []
            hero = served.get("hero_id")
            checks = {
                "active": veto["active"] is True,
                "scope": veto["scope"] == "watch_hero" and veto["reason"] is None,
                "variant": veto["variant"] == "v7",
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
    """Counters, v7 diagnostics and decision frames (writers: SpacePreferenceVeto via
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
    """Session vs rollout under the v7 strict installer, same seed (writer:
    run_parity_probe). Counters AND v7 diagnostics (v5 nested) must be equal."""
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
            "session_variant": left.get("variant") == "v7",
            "session_method": left.get("wrapper_method") == V7_METHOD,
            "rollout_method": right.get("wrapper_method") == V7_METHOD,
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
    """Released v5 default and the rollbacks intact (writer: serving_run.default_check)."""
    doc = ev["docs"].get(schema.DEFAULT_CHECK)
    if not isinstance(doc, Mapping):
        return ["default_check.json missing"]
    out = []
    if doc.get("intent_sha256") != ev["intent_sha256"]:
        out.append("default_check not bound to this intent")
    if doc.get("flags_from_empty_env") != RELEASE_FLAGS:
        out.append("empty env flags are not the released {watch_hero: on, play_ai: off}")
    if doc.get("variant_from_empty_env") != "v5":
        out.append("empty env variant is not the released v5")
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
    out["current_tree_v7_source_sha256s"] = {
        rel: (v2audit.schema.sha256_file(REPO / rel) if (REPO / rel).is_file() else None)
        for rel in V7_SOURCE_SHA256S
    }
    return out


def parity_exercise(ev: Mapping[str, Any]) -> Dict[str, Any]:
    """Which veto branches the parity probes exercised (reported, not gated).

    ``rerank-exercised`` = a v7 re-rank changed an action on the session side;
    ``exercised`` = some replacement ran; otherwise S5's equality covers only the
    keep/fallback path.
    """
    per_probe = {}
    for probe in parity_records(ev):
        left = probe.get("session") or {}
        vetoes = (left.get("counters") or {}).get("vetoes_applied")
        changes = (left.get("diagnostics") or {}).get("rerank_changes")
        per_probe[str(probe.get("probe_id"))] = {
            "vetoes_applied": vetoes if is_int(vetoes) else None,
            "rerank_changes": changes if is_int(changes) else None,
        }
    vetoes = sum(v["vetoes_applied"] or 0 for v in per_probe.values())
    changes = sum(v["rerank_changes"] or 0 for v in per_probe.values())
    label = "rerank-exercised" if changes else ("exercised" if vetoes else "non-exercising")
    return {
        "per_probe": per_probe,
        "vetoes_applied_total": vetoes,
        "rerank_changes_total": changes,
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
    parser = argparse.ArgumentParser(description="Audit an Apex v7-veto web-serving run.")
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
