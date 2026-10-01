#!/usr/bin/env python3
"""Audit of the Apex veto web-serving run (standard library only).

``python -I serving_audit.py --root <run-out> --out <run-out>/audit``. Exit 0 =
SERVING_PASS, 1 = SERVING_FAIL, 2 = INVALID (evidence missing or unreadable).

It imports nothing from the repo except ``schema.py`` beside it (record field
names and layout only, loaded by file path). The pinned identities, the seed
recipe and every decision rule are re-implemented here, and each rule names the
code path that writes the field it checks (protocol.md "Pass criteria").
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


def _load_schema() -> Any:
    spec = importlib.util.spec_from_file_location("apex_veto_serving_schema", HERE / "schema.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


schema = _load_schema()

CHAMPION_SHA256 = "43d4e2c53919dd59416c145cf0ba7c4faf1c7f298eebbb1723146807d747ac93"
WRAPPER_SOURCE_SHA256 = "1b62d15c48987584533efbefdf3fd84e1e3f2e22cfe70dbb82b43ec57169c428"
WRAPPER_SOURCE_PATH = "src/evaluation/safety_veto.py"
SERVED_CONFIG = (
    "mechanics_v2.yaml",
    "c2db7607915f70eaa46489a48598db65c4593cf039a166f37842423bb1479911",
)
PARITY_CONFIG_SHA256 = "4146baa3a06102b8afd627b1fba8384e9a2f47aaac4a9bc96292c3eb71715aa5"
CHAMPION_BASENAME = "champion_a5_freespace_20260621.pth"
VETO_METHOD = "free-space-veto/v2-speed-preserving"
# FreeSpaceVeto.descriptor() as recorded in the run-v3 STRICT_PASS records.
DESCRIPTOR = {
    "method": VETO_METHOD,
    "free_space_bfs_cap": 160,
    "free_space_min_cap": 32,
    "boost_approximation": "one-step-direction-feature",
    "replacement_rule": "highest-q-eligible-same-speed-mode-then-other",
}
DESIGN = {
    False: {
        "domain": "apex-veto-web-serving-v1",
        "counts": {"watch": 1, "play": 49, "parity": 2},
        "horizon": 5000,
        "parity_horizon": 5000,
    },
    True: {
        "domain": "apex-veto-web-serving-smoke-v1",
        "counts": {"watch": 1, "play": 1, "parity": 1},
        "horizon": 500,
        "parity_horizon": 200,
    },
}
KIND_PURPOSE = {schema.KIND_WATCH: "watch", schema.KIND_PLAY: "play"}
SUB_COUNTERS = ("vetoes_to_boost", "vetoed_base_boost", "vetoes_speed_switched")
MAX_LISTED = 50


class Invalid(Exception):
    """Evidence is missing or unreadable: the verdict is INVALID, not FAIL."""


def uint32_seed(domain: str, purpose: str, index: int) -> int:
    digest = hashlib.sha256(f"{domain}|{purpose}|{index}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, ValueError) as exc:
        raise Invalid(f"unreadable {path.name}: {exc}") from exc


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


# ---------------------------------------------------------------- reusable rules


def counter_failures(counters: Any, where: str) -> List[str]:
    """S4 partition and bounds on one ``SafetyVetoCounters.to_dict()`` mapping.

    Writer: ``SafetyVetoCounters.record`` via ``FreeSpaceVeto.apply`` (called by
    ``AISnake.update`` on the greedy path only).
    """
    if not isinstance(counters, Mapping) or set(counters) != set(schema.COUNTER_KEYS):
        shown = sorted(counters) if isinstance(counters, Mapping) else counters
        return [f"{where}: counter keys {shown}"]
    bad = [k for k in schema.COUNTER_KEYS if not is_int(counters[k]) or counters[k] < 0]
    if bad:
        return [f"{where}: non-integer or negative {bad}"]
    out = []
    parts = counters["kept_base"] + counters["vetoes_applied"] + counters["fallback_no_spacious"]
    if counters["decisions"] != parts:
        out.append(f"{where}: decisions {counters['decisions']} != partition {parts}")
    for key in SUB_COUNTERS:
        if counters[key] > counters["vetoes_applied"]:
            out.append(f"{where}: {key} > vetoes_applied")
    return out


def wrapper_failures(wrapper: Any, where: str) -> List[str]:
    """S2 wrapper identity. Writer: ``safety_veto_serving.wrapper_identity``."""
    if not isinstance(wrapper, Mapping):
        return [f"{where}: no wrapper identity"]
    out = []
    if wrapper.get("method") != VETO_METHOD:
        out.append(f"{where}: wrapper method {wrapper.get('method')!r}")
    if wrapper.get("descriptor") != DESCRIPTOR:
        out.append(f"{where}: wrapper descriptor differs from the pinned descriptor")
    if wrapper.get("source_sha256") != WRAPPER_SOURCE_SHA256:
        out.append(f"{where}: wrapper source sha {wrapper.get('source_sha256')}")
    if wrapper.get("source_path") != WRAPPER_SOURCE_PATH:
        out.append(f"{where}: wrapper source path {wrapper.get('source_path')!r}")
    return out


def probe_record_failures(probe: Any, where: str) -> List[str]:
    """The ``FreeSpaceVeto.record()`` shape (descriptor + counters), as in run-v3 records."""
    if not isinstance(probe, Mapping):
        return [f"{where}: no safety_veto probe"]
    descriptor = {k: v for k, v in probe.items() if k != "counters"}
    out = [] if descriptor == DESCRIPTOR else [f"{where}: probe descriptor differs"]
    return out + counter_failures(probe.get("counters"), f"{where}.counters")


# ---------------------------------------------------------------- evidence


def load_evidence(root: Path) -> Dict[str, Any]:
    """Read intent, receipt and every listed file; verify bytes against the receipt.

    Writer: ``serving_run.main`` (create-only). A missing or unreadable file is
    INVALID; a listed file whose sha256 differs is an S1 failure.
    """
    intent_path, receipt_path = root / schema.INTENT, root / schema.RECEIPT
    if not intent_path.is_file() or not receipt_path.is_file():
        raise Invalid("intent.json or receipt.json is missing")
    intent, receipt = read_json(intent_path), read_json(receipt_path)
    if not isinstance(intent, Mapping) or not isinstance(receipt, Mapping):
        raise Invalid("intent/receipt is not an object")
    if not isinstance(intent.get("smoke"), bool):
        raise Invalid("intent.smoke is not a boolean")
    files = receipt.get("files")
    if not isinstance(files, Mapping):
        raise Invalid("receipt.files is not an object")
    docs, hash_mismatch = {}, []
    for rel, sha in sorted(files.items()):
        path = root / rel
        if not path.is_file():
            raise Invalid(f"listed file missing: {rel}")
        if schema.sha256_file(path) != sha:
            hash_mismatch.append(rel)
        docs[rel] = read_json(path)
    return {
        "intent": intent,
        "intent_sha256": schema.sha256_file(intent_path),
        "receipt": receipt,
        "docs": docs,
        "hash_mismatch": hash_mismatch,
    }


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


def episode_records(ev: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    prefix = f"{schema.RECORDS_DIR}/"
    return [doc for rel, doc in sorted(ev["docs"].items()) if rel.startswith(prefix)]


def parity_records(ev: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    prefix = f"{schema.PARITY_DIR}/"
    return [doc for rel, doc in sorted(ev["docs"].items()) if rel.startswith(prefix)]


# ---------------------------------------------------------------- criteria


def s1_completeness(ev: Mapping[str, Any], design: Mapping[str, Any]) -> List[str]:
    """Counts, statuses, intent binding, seeds and frame accounting (writer: serving_run)."""
    out, intent, receipt = [], ev["intent"], ev["receipt"]
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
    """Checkpoint, wrapper and served-config binding (writers: harness, safety_veto_serving,
    GameSession)."""
    out = wrapper_failures(ev["intent"].get("wrapper_identity"), "intent.wrapper_identity")
    if (ev["intent"].get("checkpoint") or {}).get("sha256") != CHAMPION_SHA256:
        out.append("intent.checkpoint.sha256 is not the pinned champion")
    for rec in episode_records(ev):
        where = rec.get("episode_id", "?")
        ckpt = rec.get("checkpoint") or {}
        if ckpt.get("sha256") != CHAMPION_SHA256 or ckpt.get("session_sha256") != CHAMPION_SHA256:
            out.append(f"{where}: checkpoint sha (file/session) is not the pinned champion")
        out += wrapper_failures(rec.get("wrapper"), where)
        served = rec.get("served")
        if not isinstance(served, Mapping):
            out.append(f"{where}: no served identity")
            continue
        expected_mode = KIND_PURPOSE.get(rec.get("kind"))
        checks = {
            "mode": served.get("mode") == expected_mode,
            "obs_spec": served.get("obs_spec") == "vector61",
            "policy_class": served.get("policy_class") == "ApexPolicy",
            "policy_training": served.get("policy_training") is False,
            "epsilon": served.get("epsilon") == 0.0,
            "config": (served.get("config_basename"), served.get("config_sha256")) == SERVED_CONFIG,
        }
        out += [f"{where}: served.{name}" for name, ok in checks.items() if not ok]
    return out


def expected_wrapped(rec: Mapping[str, Any]) -> Optional[List[int]]:
    served = rec.get("served") or {}
    ids = served.get("snake_ids")
    if not isinstance(ids, list) or not ids:
        return None
    if rec.get("kind") == schema.KIND_WATCH:
        hero = served.get("hero_id")
        return [hero] if hero == ids[0] else None
    human = served.get("human_id")
    if human not in ids:
        return None
    return sorted(i for i in ids if i != human)


def s3_scope(ev: Mapping[str, Any]) -> List[str]:
    """Wrapper active on exactly the flag's snakes (writer: install_serving_vetoes)."""
    out = []
    flags = {
        schema.KIND_WATCH: ("watch_hero", {"watch_hero": True, "play_ai": False}),
        schema.KIND_PLAY: ("play_ai", {"watch_hero": False, "play_ai": True}),
    }
    for rec in episode_records(ev):
        where = rec.get("episode_id", "?")
        veto = rec.get("veto")
        scope, flag_values = flags.get(rec.get("kind"), (None, None))
        if not isinstance(veto, Mapping) or veto.get("active") is not True:
            out.append(f"{where}: wrapper not active")
            continue
        if veto.get("scope") != scope or veto.get("reason") is not None:
            out.append(f"{where}: scope {veto.get('scope')!r} reason {veto.get('reason')!r}")
        if veto.get("flags") != flag_values or (rec.get("served") or {}).get("veto_flags") != (
            flag_values
        ):
            out.append(f"{where}: flags {veto.get('flags')}")
        wanted = expected_wrapped(rec)
        if wanted is None or veto.get("wrapped_snake_ids") != wanted:
            out.append(f"{where}: wrapped {veto.get('wrapped_snake_ids')} != expected {wanted}")
        elif rec.get("kind") == schema.KIND_PLAY:
            if len(wanted) != len(rec["served"]["snake_ids"]) - 1:
                out.append(
                    f"{where}: Play wraps {len(wanted)} of {len(rec['served']['snake_ids'])}"
                )
    return out


def s4_counters(ev: Mapping[str, Any]) -> List[str]:
    """Counter partition, bounds, totals and equality with harness decision frames.

    Writers: counters from ``FreeSpaceVeto`` (delta over the episode); decision
    frames from ``serving_run.DecisionCounter`` (respawn-then-act order of
    ``GameState.update``). Both exist only with epsilon 0 (checked in S2).
    """
    out = []
    for rec in episode_records(ev):
        where = rec.get("episode_id", "?")
        veto, frames = rec.get("veto") or {}, rec.get("decision_frames") or {}
        per_snake, per_frames = veto.get("per_snake"), frames.get("per_snake")
        wrapped = [str(i) for i in veto.get("wrapped_snake_ids") or []]
        if not isinstance(per_snake, Mapping) or not isinstance(per_frames, Mapping):
            out.append(f"{where}: missing per-snake counters or decision frames")
            continue
        if sorted(per_snake) != sorted(wrapped) or sorted(per_frames) != sorted(wrapped):
            out.append(f"{where}: per-snake keys differ from wrapped ids")
            continue
        for sid in wrapped:
            failures = counter_failures(per_snake[sid], f"{where}[{sid}]")
            out += failures
            if not failures and per_snake[sid]["decisions"] != per_frames[sid]:
                out.append(
                    f"{where}[{sid}]: decisions {per_snake[sid]['decisions']} != "
                    f"decision frames {per_frames[sid]}"
                )
        if not any(counter_failures(per_snake[s], "") for s in wrapped):
            total = {k: sum(per_snake[s][k] for s in wrapped) for k in schema.COUNTER_KEYS}
            if veto.get("total") != total:
                out.append(f"{where}: veto.total is not the per-snake sum")
            if frames.get("total") != sum(per_frames.values()):
                out.append(f"{where}: decision_frames.total is not the per-snake sum")
            if rec.get("frames_stepped", 0) > 0 and total["decisions"] <= 0:
                out.append(f"{where}: stepped frames but no wrapped decision")
    return out


def s5_parity(ev: Mapping[str, Any], design: Mapping[str, Any], smoke: bool) -> List[str]:
    """Session vs tournament_eval.rollout on the same seed (writer: run_parity_probe)."""
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
            "horizon": probe["horizon_frames"] == horizon,
            "frames_compared": probe["frames_compared"] == horizon,
            "first_divergence_frame": probe["first_divergence_frame"] is None,
            "trace_sha256": probe["trace_sha256_equal"] is True
            and left.get("trace_sha256") is not None
            and left.get("trace_sha256") == right.get("trace_sha256"),
            "veto_counters": probe["veto_counters_equal"] is True
            and left.get("counters") == right.get("counters"),
            "session_config": left.get("config_sha256") == PARITY_CONFIG_SHA256,
            "wrapped_hero_only": left.get("wrapped_snake_ids") == [0],
            "record_complete": right.get("record_complete") is True or smoke,
        }
        out += [f"{where}: {name}" for name, ok in checks.items() if not ok]
        for side, block in (("session", left), ("rollout", right)):
            out += counter_failures(block.get("counters"), f"{where}.{side}")
        if (left.get("counters") or {}).get("decisions") != left.get("decision_frames"):
            out.append(f"{where}: session decisions != decision frames")
    return out


def s6_defaults(ev: Mapping[str, Any]) -> List[str]:
    """Flags off by default; served checkpoint and config unchanged (writer: default_check)."""
    doc = ev["docs"].get(schema.DEFAULT_CHECK)
    if not isinstance(doc, Mapping):
        return ["default_check.json missing"]
    out = []
    if doc.get("intent_sha256") != ev["intent_sha256"]:
        out.append("default_check not bound to this intent")
    if doc.get("flags_from_empty_env") != {"watch_hero": False, "play_ai": False}:
        out.append("flags are not off for an empty environment")
    if doc.get("default_checkpoint_basename") != CHAMPION_BASENAME:
        out.append("served default checkpoint changed")
    if (doc.get("served_config_basename"), doc.get("served_config_sha256")) != SERVED_CONFIG:
        out.append("served default config changed")
    modes = doc.get("modes") or {}
    for mode in ("watch", "play"):
        state = modes.get(mode) or {}
        if state.get("mode") != mode or state.get("active") is not False:
            out.append(f"default {mode} build is not unwrapped")
        if state.get("snakes_with_veto") != []:
            out.append(f"default {mode} build has wrapped snakes")
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
    source = REPO / WRAPPER_SOURCE_PATH
    out["current_tree_wrapper_source_sha256"] = (
        schema.sha256_file(source) if source.is_file() else None
    )
    return out


def audit(root: Path) -> Dict[str, Any]:
    """Run every criterion; never raises for evidence problems (INVALID instead)."""
    try:
        ev = load_evidence(root)
        smoke = ev["intent"]["smoke"]
        design = DESIGN[smoke]
        criteria = {
            "S1": s1_completeness(ev, design),
            "S2": s2_identity(ev),
            "S3": s3_scope(ev),
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
    parser = argparse.ArgumentParser(description="Audit an Apex veto web-serving run.")
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
