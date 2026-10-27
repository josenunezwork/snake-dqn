"""Background-load covariates for the served-tick timing diagnostic.

This module deliberately does not decide the panel verdict.  It aligns sealed,
off-tick host samples to individual trace rows so a report can describe the
environment in which timing was observed without treating routine macOS work as
an automatic contamination reason.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from research.bl_served_search.served_tick_probe import _distribution

POLICY = "classified-v1"
DESKTOP_POLICY = "classified-desktop-v1"
SUPPORTED_POLICIES = {POLICY, DESKTOP_POLICY}
LOW_BACKGROUND_PERCENT = 20.0
MAX_SAMPLE_NS = 5_000_000_000
PROCESS_KINDS = {"macos_service", "desktop_renderer", "agent_workload", "unknown"}
DESKTOP_PROCESS_KINDS = PROCESS_KINDS | {"desktop_app"}
BACKGROUND_KINDS = {"macos_service", "desktop_renderer", "desktop_app"}
BACKGROUND_CATEGORY_KINDS = {
    "service": "macos_service",
    "desktop_renderer": "desktop_renderer",
    "desktop_app": "desktop_app",
}
RAW_CATEGORY_FIELDS = {
    "service": "macos_service_cpu_percent",
    "desktop_renderer": "desktop_renderer_cpu_percent",
    "desktop_app": "desktop_app_cpu_percent",
}


def _integer(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"malformed {name}")
    return value


def _number(value: Any, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)) or value < 0:
        raise ValueError(f"malformed {name}")
    return float(value)


def _has_full_desktop_history(registry_evidence: Mapping[str, Any]) -> bool:
    if not isinstance(registry_evidence, Mapping):
        return False
    reg = registry_evidence.get("registry")
    return (
        "process_history" in registry_evidence
        or "history_log" in registry_evidence
        or "verified" in registry_evidence
        or (isinstance(reg, Mapping) and ("applications" in reg or "owner_uid" in reg))
    )


def _validate_registry_history(
    registry_evidence: Mapping[str, Any],
) -> dict[tuple[int, str], list[dict[str, Any]]]:
    """Validate exported per-(pid, birth) history chains and fail-closed permanence."""
    from research.bl_served_search.desktop_apps import (
        _valid_verified_identity,
        classify_row,
        decision_from_history,
    )

    verified = registry_evidence.get("verified")
    if not isinstance(verified, Mapping):
        raise ValueError("desktop registry verified map missing")
    for field in (
        "process_history",
        "history_log",
        "bindings",
        "binding_history",
        "revoked_processes",
        "revoked_bindings",
    ):
        if not isinstance(registry_evidence.get(field), list):
            raise ValueError(f"desktop registry {field} missing or malformed")
    reg_dict = registry_evidence.get("registry")
    if isinstance(reg_dict, Mapping) and isinstance(reg_dict.get("applications"), list):
        for cmd, ident in verified.items():
            if not isinstance(cmd, str) or not _valid_verified_identity(cmd, ident, reg_dict):
                raise ValueError("desktop verified executable does not match sealed registry")
        for field in ("bindings", "binding_history"):
            for item in registry_evidence[field]:
                if (
                    not isinstance(item, Mapping)
                    or not isinstance(item.get("command"), str)
                    or verified.get(item["command"]) != item.get("identity")
                    or not _valid_verified_identity(item["command"], item.get("identity"), reg_dict)
                ):
                    raise ValueError(f"desktop {field} entry does not match sealed registry")
    log_by_proc: dict[tuple[int, str], list[dict[str, Any]]] = {}
    seen_after: dict[tuple[int, str], list[Mapping[str, Any]]] = {}
    last_obs_by_proc: dict[tuple[int, str], int] = {}
    for entry in registry_evidence["history_log"]:
        if not isinstance(entry, Mapping):
            raise ValueError("malformed desktop history log entry")
        pid, birth = entry.get("pid"), entry.get("birth")
        before, after = entry.get("before"), entry.get("after")
        obs_ns = entry.get("observation_ns")
        source_row = entry.get("row")
        if (
            type(pid) is not int
            or pid <= 0
            or not isinstance(birth, str)
            or not birth
            or not isinstance(after, Mapping)
            or (before is not None and not isinstance(before, Mapping))
            or type(obs_ns) is not int
            or obs_ns < 0
            or not isinstance(source_row, Mapping)
        ):
            raise ValueError("malformed desktop history log state")
        proc_key = (pid, birth)
        if (after.get("pid"), after.get("birth")) != proc_key:
            raise ValueError("desktop history log pid/birth mismatch")
        if proc_key in last_obs_by_proc and obs_ns < last_obs_by_proc[proc_key]:
            raise ValueError("desktop history log observation_ns out of order")
        last_obs_by_proc[proc_key] = obs_ns
        priors = seen_after.setdefault(proc_key, [])
        if not priors:
            if before is not None:
                raise ValueError("desktop history log initial state must have before=None")
            if after.get("sequence") not in (None, 0):
                raise ValueError("desktop history log initial sequence must be 0")
        else:
            if not isinstance(before, Mapping) or before != priors[-1]:
                raise ValueError("desktop history log discontinuity")
            if (
                type(before.get("sequence")) is int
                and type(after.get("sequence")) is int
                and after["sequence"] != before["sequence"] + 1
            ):
                raise ValueError("desktop history log sequence discontinuity")
            if (before.get("status") == "revoked" or before.get("revoked")) and (
                after.get("status") != "revoked" or not after.get("revoked")
            ):
                raise ValueError("desktop revoked process re-admitted in history log")
            if before.get("status") == "non_desktop" and after.get("status") not in {
                "non_desktop",
                "revoked",
            }:
                raise ValueError("desktop non_desktop process admitted in history log")
            if before.get("identity_binding_changed") and not after.get("identity_binding_changed"):
                raise ValueError("desktop identity binding change cleared in history log")
        if "sequence" in entry and entry["sequence"] != after.get("sequence"):
            raise ValueError("desktop history log entry sequence mismatch")
        replayed_kind, replayed_after, replayed_reason = classify_row(
            before, source_row, registry_evidence
        )
        if replayed_after != dict(after):
            raise ValueError("desktop history log event replay differs")
        if "classification" in entry:
            expected_decision = decision_from_history(
                replayed_kind, replayed_after, replayed_reason
            )
            if not isinstance(entry["classification"], Mapping) or expected_decision != dict(
                entry["classification"]
            ):
                raise ValueError("desktop history log classification replay differs")
        priors.append(after)
        log_by_proc.setdefault(proc_key, []).append(dict(entry))
    hist_by_proc: dict[tuple[int, str], dict[str, Any]] = {}
    for item in registry_evidence["process_history"]:
        if not isinstance(item, Mapping):
            raise ValueError("malformed desktop process history entry")
        pid, birth = item.get("pid"), item.get("birth")
        if (
            type(pid) is not int
            or pid <= 0
            or not isinstance(birth, str)
            or not birth
            or (pid, birth) in hist_by_proc
        ):
            raise ValueError("malformed or duplicate desktop process history")
        hist_by_proc[(pid, birth)] = dict(item)
    if set(hist_by_proc) != set(log_by_proc) or any(
        hist_by_proc[key] != log_by_proc[key][-1]["after"] for key in hist_by_proc
    ):
        raise ValueError("desktop process history does not reconcile with history log")
    active_proc_keys = {
        (item.get("pid"), item.get("birth"))
        for item in registry_evidence["bindings"]
        if isinstance(item, Mapping)
    }
    for proc_key, final_state in hist_by_proc.items():
        if final_state.get("status") == "revoked" and proc_key in active_proc_keys:
            raise ValueError("revoked desktop process retains active binding")
    return log_by_proc


def _verify_step_in_log(
    proc_key: tuple[int, str] | None,
    history_before: Mapping[str, Any] | None,
    history_after: Mapping[str, Any],
    log_by_proc: Mapping[tuple[int, str], list[dict[str, Any]]],
    *,
    observation_ns: int | None = None,
    is_point: bool = False,
    expected_point_sequence: int | None = None,
) -> None:
    if proc_key is None:
        return
    entries = log_by_proc.get(proc_key)
    if not entries:
        raise ValueError("desktop process missing from history log")
    if history_before != history_after:
        matching_indices = [
            idx
            for idx, entry in enumerate(entries)
            if entry["before"] == history_before and entry["after"] == history_after
        ]
        if not matching_indices:
            raise ValueError("desktop process transition missing from history log")
    else:
        matching_indices = [
            idx for idx, entry in enumerate(entries) if entry["after"] == history_after
        ]
        if not matching_indices:
            raise ValueError("desktop process state missing from history log")
    if is_point:
        row_seq = history_after.get("sequence")
        final_after = entries[-1]["after"]
        final_seq = final_after.get("sequence")
        if type(row_seq) is not int or type(final_seq) is not int:
            if final_after != history_after:
                raise ValueError("desktop process state superseded before observation")
            return
        if row_seq < final_seq:
            for later_entry in entries[row_seq + 1 :]:
                later_after = later_entry.get("after", {})
                if (
                    later_after.get("status") != "non_desktop"
                    or later_after.get("revoked")
                    or later_after.get("identity_binding_changed")
                ):
                    raise ValueError("desktop process state superseded before observation")
        target_seq = expected_point_sequence if type(expected_point_sequence) is int else final_seq
        if row_seq != target_seq:
            raise ValueError("desktop process state superseded before observation")
        return
    if type(observation_ns) is int:
        valid_at_obs = False
        for idx in matching_indices:
            entry_ns = entries[idx].get("observation_ns")
            if type(entry_ns) is int and observation_ns < entry_ns:
                continue
            if idx + 1 < len(entries):
                next_ns = entries[idx + 1].get("observation_ns")
                if type(next_ns) is int and observation_ns >= next_ns:
                    continue
            valid_at_obs = True
            break
        if not valid_at_obs:
            raise ValueError("desktop process state superseded before observation")
    elif entries[-1]["after"] != history_after:
        raise ValueError("desktop process state superseded before observation")


def _replay_row_strict(
    row: Mapping[str, Any],
    registry_evidence: Mapping[str, Any],
    log_by_proc: Mapping[tuple[int, str], list[dict[str, Any]]] | None = None,
    replayed_proc_history: dict[tuple[int, str], dict[str, Any]] | None = None,
    *,
    is_point: bool = False,
    sample_start_ns: int | None = None,
    sample_end_ns: int | None = None,
    point_obs_ns: int | None = None,
    expected_point_sequence: int | None = None,
) -> dict[str, Any]:
    """Replay a process row via pure ``classify_row`` and require exact reproduction."""
    from research.bl_served_search.desktop_apps import (
        _reconstruct_exported_history,
        classify_row,
        decision_from_history,
    )
    from research.bl_served_search.host_load import _argv_changed, classification_evidence

    pid, birth = row.get("pid"), row.get("birth")
    proc_key = (
        (pid, birth) if type(pid) is int and pid > 0 and isinstance(birth, str) and birth else None
    )
    transition = row.get("interval_transition")
    if transition is not None:
        if not isinstance(transition, Mapping):
            raise ValueError("malformed desktop interval transition")
        previous = transition.get("previous")
        prior_classification = transition.get("previous_classification")
        if (
            not isinstance(previous, Mapping)
            or not isinstance(prior_classification, Mapping)
            or previous.get("pid") != row.get("pid")
            or previous.get("birth") != row.get("birth")
            or (
                previous.get("uid") == row.get("uid")
                and previous.get("command") == row.get("command")
                and not _argv_changed(dict(previous), dict(row))
            )
        ):
            raise ValueError("malformed desktop interval transition identity")
        if (
            not is_point
            and sample_start_ns is not None
            and "previous_observation_ns" in transition
            and transition["previous_observation_ns"] != sample_start_ns
        ):
            raise ValueError("desktop previous observation timestamp does not match interval")
        if (
            not is_point
            and sample_end_ns is not None
            and "observation_ns" in row
            and row["observation_ns"] != sample_end_ns
        ):
            raise ValueError("desktop process observation timestamp does not match interval")
        if "previous_history_before" in transition:
            prev_history_before = transition["previous_history_before"]
            if prev_history_before is not None and not isinstance(prev_history_before, Mapping):
                raise ValueError("malformed desktop previous history")
        elif replayed_proc_history is not None and proc_key in replayed_proc_history:
            prev_history_before = replayed_proc_history[proc_key]
        else:
            prev_history_before = _reconstruct_exported_history(
                {**previous, "classification": prior_classification}, registry_evidence
            )
        if (
            replayed_proc_history is not None
            and proc_key is not None
            and proc_key in replayed_proc_history
            and prev_history_before != replayed_proc_history[proc_key]
        ):
            raise ValueError("desktop process history continuity differs")
        prev_input: dict[str, Any] = {
            **previous,
            "admit": False,
            "classification": prior_classification,
        }
        if isinstance(transition.get("previous_executable_status"), str):
            prev_input["executable_status"] = transition["previous_executable_status"]
        prior_kind, prior_history_after, prior_reason = classify_row(
            prev_history_before, prev_input, registry_evidence
        )
        prior_decision = decision_from_history(prior_kind, prior_history_after, prior_reason)
        if prior_kind != prior_classification.get("kind") or prior_decision != dict(
            prior_classification
        ):
            raise ValueError("desktop interval previous classification replay differs")
        if (
            "previous_history_after" in transition
            and prior_history_after != transition["previous_history_after"]
        ):
            raise ValueError("desktop interval previous history replay differs")
        if (
            "previous_sequence" in transition
            and prior_history_after.get("sequence") != transition["previous_sequence"]
        ):
            raise ValueError("desktop interval previous sequence replay differs")
        if "history_before" in row and row["history_before"] != prior_history_after:
            raise ValueError("desktop interval transition history continuity differs")
        prev_obs = (
            sample_start_ns
            if sample_start_ns is not None
            else (
                transition.get("previous_observation_ns")
                if type(transition.get("previous_observation_ns")) is int
                else None
            )
        )
        if log_by_proc is not None and (
            "previous_history_before" in transition
            or (proc_key is not None and proc_key in log_by_proc)
        ):
            _verify_step_in_log(
                proc_key,
                prev_history_before,
                prior_history_after,
                log_by_proc,
                observation_ns=prev_obs,
            )
        cur_kind, cur_history_after, cur_reason = classify_row(
            prior_history_after, {**row, "admit": False}, registry_evidence
        )
        cur_decision = decision_from_history(cur_kind, cur_history_after, cur_reason)
        kinds = {cur_kind, prior_kind}
        conservative_kind = (
            "agent_workload"
            if "agent_workload" in kinds
            else "unknown" if "unknown" in kinds else cur_kind
        )
        expected_classification = classification_evidence(conservative_kind, cur_decision)
        classification = row.get("classification")
        if isinstance(classification, Mapping) and expected_classification != dict(classification):
            raise ValueError("desktop registry replay differs")
        if "history_after" in row and cur_history_after != row["history_after"]:
            raise ValueError("desktop process history replay differs")
        if "sequence" in row and cur_history_after.get("sequence") != row["sequence"]:
            raise ValueError("desktop process sequence replay differs")
        if transition.get("conservative_kind") != conservative_kind:
            raise ValueError("desktop interval conservative classification differs")
        cur_obs = (
            sample_end_ns
            if sample_end_ns is not None
            else (row.get("observation_ns") if type(row.get("observation_ns")) is int else None)
        )
        if log_by_proc is not None and (
            "history_after" in row or (proc_key is not None and proc_key in log_by_proc)
        ):
            _verify_step_in_log(
                proc_key,
                prior_history_after,
                cur_history_after,
                log_by_proc,
                observation_ns=cur_obs,
            )
        if replayed_proc_history is not None and proc_key is not None:
            replayed_proc_history[proc_key] = cur_history_after
        return expected_classification

    if (
        not is_point
        and sample_end_ns is not None
        and "observation_ns" in row
        and row["observation_ns"] != sample_end_ns
    ):
        raise ValueError("desktop process observation timestamp does not match interval")
    if "history_before" in row:
        history_before = row["history_before"]
        if history_before is not None and not isinstance(history_before, Mapping):
            raise ValueError("malformed desktop process history")
    elif (
        replayed_proc_history is not None
        and proc_key is not None
        and proc_key in replayed_proc_history
    ):
        history_before = replayed_proc_history[proc_key]
    else:
        history_before = _reconstruct_exported_history(row, registry_evidence)
    if (
        replayed_proc_history is not None
        and proc_key is not None
        and proc_key in replayed_proc_history
        and history_before != replayed_proc_history[proc_key]
    ):
        raise ValueError("desktop process history continuity differs")
    if "admit" in row:
        admit = bool(row["admit"])
    elif "history_before" in row:
        admit = not is_point or "admission" in row
    else:
        admit = False
    cur_kind, cur_history_after, cur_reason = classify_row(
        history_before, {**row, "admit": admit}, registry_evidence
    )
    if not is_point and isinstance(history_before, Mapping):
        if (
            history_before.get("uid") != cur_history_after.get("uid")
            or history_before.get("command") != cur_history_after.get("command")
            or _argv_changed(
                {
                    "uid": history_before.get("uid"),
                    "command": history_before.get("command"),
                    "argv": history_before.get("argv"),
                    "argv_display": history_before.get("argv_display"),
                    **(
                        {"identity_observation_error": "argv_identity_unavailable"}
                        if history_before.get("argv_status") != "ok"
                        else {}
                    ),
                },
                dict(row),
            )
        ):
            raise ValueError("desktop interval transition evidence missing")
    cur_decision = decision_from_history(cur_kind, cur_history_after, cur_reason)
    classification = row.get("classification")
    if isinstance(classification, Mapping) and cur_decision != dict(classification):
        raise ValueError(
            "desktop point classification replay differs"
            if is_point
            else "desktop registry replay differs"
        )
    if "history_after" in row and cur_history_after != row["history_after"]:
        raise ValueError("desktop process history replay differs")
    if "sequence" in row and cur_history_after.get("sequence") != row["sequence"]:
        raise ValueError("desktop process sequence replay differs")
    eff_obs_ns = (
        (row.get("observation_ns") if type(row.get("observation_ns")) is int else point_obs_ns)
        if is_point
        else (
            sample_end_ns
            if sample_end_ns is not None
            else (row.get("observation_ns") if type(row.get("observation_ns")) is int else None)
        )
    )
    if log_by_proc is not None and (
        "history_before" in row or (proc_key is not None and proc_key in log_by_proc)
    ):
        _verify_step_in_log(
            proc_key,
            history_before,
            cur_history_after,
            log_by_proc,
            observation_ns=eff_obs_ns,
            is_point=is_point,
            expected_point_sequence=expected_point_sequence,
        )
    if replayed_proc_history is not None and proc_key is not None:
        replayed_proc_history[proc_key] = cur_history_after
    return cur_decision


def _replay_kind(
    row: Mapping[str, Any],
    registry_evidence: Mapping[str, Any],
    log_by_proc: Mapping[tuple[int, str], list[dict[str, Any]]] | None = None,
    replayed_proc_history: dict[tuple[int, str], dict[str, Any]] | None = None,
    *,
    is_point: bool = False,
    sample_start_ns: int | None = None,
    sample_end_ns: int | None = None,
    point_obs_ns: int | None = None,
    expected_point_sequence: int | None = None,
) -> dict[str, Any]:
    """Replay both identities of an interval transition with conservative precedence."""
    from research.bl_served_search.host_load import _argv_changed

    if _has_full_desktop_history(registry_evidence):
        return _replay_row_strict(
            row,
            registry_evidence,
            log_by_proc,
            replayed_proc_history,
            is_point=is_point,
            sample_start_ns=sample_start_ns,
            sample_end_ns=sample_end_ns,
            point_obs_ns=point_obs_ns,
            expected_point_sequence=expected_point_sequence,
        )
    transition = row.get("interval_transition")
    if transition is None:
        return _replay_identity(row, registry_evidence)
    if not isinstance(transition, Mapping):
        raise ValueError("malformed desktop interval transition")
    previous = transition.get("previous")
    prior_classification = transition.get("previous_classification")
    if (
        not isinstance(previous, Mapping)
        or not isinstance(prior_classification, Mapping)
        or previous.get("pid") != row.get("pid")
        or previous.get("birth") != row.get("birth")
        or (
            previous.get("uid") == row.get("uid")
            and previous.get("command") == row.get("command")
            and not _argv_changed(dict(previous), dict(row))
        )
    ):
        raise ValueError("malformed desktop interval transition identity")
    current = _replay_identity(row, registry_evidence)
    prior = _replay_identity(
        {**previous, "classification": prior_classification}, registry_evidence
    )
    if prior.get("kind") != prior_classification.get("kind"):
        raise ValueError("desktop interval previous classification replay differs")
    kinds = {current.get("kind"), prior.get("kind")}
    kind = (
        "agent_workload"
        if "agent_workload" in kinds
        else "unknown" if "unknown" in kinds else current.get("kind")
    )
    if transition.get("conservative_kind") != kind:
        raise ValueError("desktop interval conservative classification differs")
    return current if current.get("kind") == kind else {"kind": kind}


def _replay_identity(
    row: Mapping[str, Any], registry_evidence: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply workload and system precedence before sealed desktop admission."""
    from research.bl_k1.m1_background import SYSTEM_EXECUTABLE_ROOTS
    from research.bl_served_search.host_load import _workload_kind

    command = row.get("command")
    if not isinstance(command, str):
        return {"kind": "unknown", "reason": "missing_command"}
    # A declared substantive workload always wins, even when hosted by an
    # enrolled GUI executable.  Bare executable names alone cannot decide
    # this: Claude/Codex desktop mains are eligible only through their sealed
    # PID/birth/UID binding, while their CLI names remain workloads below.
    if row.get("known_workload") or row.get("workload"):
        return {"kind": "agent_workload", "reason": "known_workload"}
    if any(
        item.get("exact_executable") == command
        for item in registry_evidence.get("registry", {}).get("workload_exclusions", [])
    ):
        return {"kind": "agent_workload", "reason": "workload_exclusion"}
    workload = _workload_kind(command, registry_evidence)
    from research.bl_served_search.desktop_apps import classify_exported

    enrolled = classify_exported(dict(row), dict(registry_evidence))
    if enrolled.get("kind") == "agent_workload" or workload == "agent_workload":
        return {"kind": "agent_workload", "reason": "workload_precedence"}
    if enrolled.get("identity_binding_changed") or workload == "unknown":
        return {**enrolled, "kind": "unknown"}
    if command.startswith(SYSTEM_EXECUTABLE_ROOTS) and ".." not in Path(command).parts:
        return {"kind": "macos_service", "reason": "system_executable"}
    return enrolled


def _samples(host_evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    policy = host_evidence.get("load_policy")
    if policy not in SUPPORTED_POLICIES:
        raise ValueError("unsupported host load policy")
    log_by_proc: dict[tuple[int, str], list[dict[str, Any]]] | None = None
    replayed_proc_history: dict[tuple[int, str], dict[str, Any]] | None = None
    if policy == DESKTOP_POLICY:
        from research.bl_served_search.desktop_apps import (
            _valid_verified_identity,
            canonical_bytes,
        )

        registry = host_evidence.get("desktop_registry")
        registry_sha256 = host_evidence.get("desktop_registry_sha256")
        registry_evidence = host_evidence.get("desktop_registry_evidence")
        if (
            not isinstance(registry, Mapping)
            or registry.get("policy") != DESKTOP_POLICY
            or not isinstance(registry_sha256, str)
            or len(registry_sha256) != 64
            or not isinstance(registry_evidence, Mapping)
            or registry_evidence.get("policy") != DESKTOP_POLICY
            or registry_evidence.get("registry") != registry
            or registry_evidence.get("registry_sha256") != registry_sha256
        ):
            raise ValueError("desktop policy registry evidence differs")
        if isinstance(registry.get("applications"), list):
            if hashlib.sha256(canonical_bytes(registry)).hexdigest() != registry_sha256:
                raise ValueError("desktop policy registry sha256 mismatch")
        if _has_full_desktop_history(registry_evidence):
            log_by_proc = _validate_registry_history(registry_evidence)
            replayed_proc_history = {}
    raw = host_evidence.get("cpu_samples")
    if not isinstance(raw, list) or not raw:
        raise ValueError("missing CPU samples")
    result: list[dict[str, Any]] = []
    previous_end = -1
    for index, sample in enumerate(raw):
        if not isinstance(sample, Mapping):
            raise ValueError("malformed CPU sample")
        start = _integer(sample.get("start_ns"), "sample start")
        end = _integer(sample.get("end_ns"), "sample end")
        if end <= start or start < previous_end:
            raise ValueError("out-of-order or overlapping CPU samples")
        previous_end = end
        if type(sample.get("coverage_complete")) is not bool:
            raise ValueError("malformed sample coverage")
        if not isinstance(sample.get("process_churn"), list):
            raise ValueError("malformed sample process churn")
        if sample["coverage_complete"] and (sample["process_churn"] or end - start > MAX_SAMPLE_NS):
            raise ValueError("sample coverage conflicts with churn or interval length")
        processes = sample.get("processes")
        if not isinstance(processes, list):
            raise ValueError("malformed sample processes")
        checked_processes = []
        for process in processes:
            if not isinstance(process, Mapping):
                raise ValueError("malformed sampled process")
            if type(process.get("pid")) is not int or type(process.get("ppid")) is not int:
                raise ValueError("malformed sampled process identity")
            if not isinstance(process.get("command"), str) or not isinstance(
                process.get("kind"), str
            ):
                raise ValueError("malformed sampled process classification")
            if process["kind"] not in (
                DESKTOP_PROCESS_KINDS if policy == DESKTOP_POLICY else PROCESS_KINDS
            ):
                raise ValueError("unknown sampled process kind")
            classification = process.get("classification")
            if policy == DESKTOP_POLICY and (
                not isinstance(classification, Mapping)
                or classification.get("kind") != process["kind"]
            ):
                raise ValueError("desktop process classification evidence differs")
            if policy == DESKTOP_POLICY:
                replay = _replay_kind(
                    process,
                    registry_evidence,
                    log_by_proc,
                    replayed_proc_history,
                    sample_start_ns=start,
                    sample_end_ns=end,
                )
                if replay.get("kind") != process["kind"]:
                    raise ValueError("desktop registry replay differs")
                if process["kind"] in {"desktop_app", "desktop_renderer"}:
                    if replay != dict(classification):
                        raise ValueError("desktop identity replay differs")
                    if isinstance(registry.get("applications"), list) and not (
                        _valid_verified_identity(
                            process["command"], classification.get("identity"), registry
                        )
                    ):
                        raise ValueError("desktop process identity does not match sealed registry")
            if process["kind"] == "desktop_app":
                # The collector binds the process to its sealed enrollment.  Do
                # not turn a friendly command name into an admission here.
                identity = process.get("desktop_identity") or classification or process
                if not isinstance(identity, Mapping):
                    raise ValueError("desktop app missing identity evidence")
                # Keep both the collector's compact process fields and the
                # nested export form so raw exports remain independently useful.
                for key, aliases in {
                    "entry": ("entry", "registry_id"),
                    "role": ("role",),
                    "reason": ("reason", "admission_reason"),
                    "identity": ("identity",),
                }.items():
                    value = next(
                        (identity.get(alias) for alias in aliases if identity.get(alias)), None
                    )
                    if not isinstance(value, (str, Mapping)) or not value:
                        raise ValueError("malformed desktop app identity evidence")
                if any(
                    classification.get(key) != identity.get(key)
                    for key in ("entry", "application", "role", "identity")
                ):
                    raise ValueError("desktop app identity/classification differs")
            checked_processes.append(
                {
                    "pid": process["pid"],
                    "ppid": process["ppid"],
                    "command": process["command"],
                    **({"birth": process["birth"]} if "birth" in process else {}),
                    **({"uid": process["uid"]} if "uid" in process else {}),
                    **({"argv": list(process["argv"])} if "argv" in process else {}),
                    **(
                        {"argv_display": process["argv_display"]}
                        if "argv_display" in process
                        else {}
                    ),
                    **({"sequence": process["sequence"]} if "sequence" in process else {}),
                    "kind": process["kind"],
                    "cpu_percent": _number(process.get("cpu_percent"), "process CPU"),
                    **({"classification": dict(classification)} if classification else {}),
                    **(
                        {"desktop_identity": dict(identity)}
                        if process["kind"] == "desktop_app"
                        else {}
                    ),
                    **(
                        {"interval_transition": dict(process["interval_transition"])}
                        if "interval_transition" in process
                        else {}
                    ),
                }
            )
        category_totals = {
            category: sum(row["cpu_percent"] for row in checked_processes if row["kind"] == kind)
            for category, kind in BACKGROUND_CATEGORY_KINDS.items()
        }
        totals = {
            "background": sum(category_totals.values()),
            "workload": sum(
                row["cpu_percent"] for row in checked_processes if row["kind"] == "agent_workload"
            ),
            "unknown": sum(
                row["cpu_percent"] for row in checked_processes if row["kind"] == "unknown"
            ),
        }
        reported = {
            "background": _number(sample.get("background_cpu_percent"), "background CPU"),
            "workload": _number(sample.get("workload_cpu_percent"), "workload CPU"),
            "unknown": _number(sample.get("unknown_cpu_percent"), "unknown CPU"),
        }
        if any(not math.isclose(totals[key], reported[key], abs_tol=1e-6) for key in totals):
            raise ValueError("sample CPU totals do not reconcile with process rows")
        # The desktop policy exports independently auditable categories.  They
        # are not separate allowances: their sum is the sole shared pool.
        if policy == DESKTOP_POLICY:
            for category, total in category_totals.items():
                field = RAW_CATEGORY_FIELDS[category]
                if not math.isclose(_number(sample.get(field), field), total, abs_tol=1e-6):
                    raise ValueError("sample desktop CPU totals do not reconcile with process rows")
        result.append(
            {
                "index": index,
                "start_ns": start,
                "end_ns": end,
                "background_cpu_percent": reported["background"],
                **{f"{category}_cpu_percent": total for category, total in category_totals.items()},
                "workload_cpu_percent": reported["workload"],
                "unknown_cpu_percent": reported["unknown"],
                "coverage_complete": sample["coverage_complete"],
                "processes": checked_processes,
            }
        )
    return result


def _background_summary(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Reconcile the shared pool without converting it into separate quotas."""
    high_since: int | None = None
    last_end: int | None = None
    max_sustained = 0
    for sample in samples:
        duration = sample["end_ns"] - sample["start_ns"]
        high = sample["background_cpu_percent"] > 150.0 and duration <= MAX_SAMPLE_NS
        if high:
            if high_since is None or last_end != sample["start_ns"]:
                high_since = sample["start_ns"]
            max_sustained = max(max_sustained, sample["end_ns"] - high_since)
        else:
            high_since = None
        last_end = sample["end_ns"]
    maxima = {
        category: max((sample[f"{category}_cpu_percent"] for sample in samples), default=None)
        for category in ("service", "desktop_renderer", "desktop_app")
    }
    maxima["combined_background"] = max(
        (sample["background_cpu_percent"] for sample in samples), default=None
    )
    return {
        "cpu_percent_max": maxima,
        "shared_background_pool": {
            "strict_percent": 150.0,
            "strict_duration_ns": 60_000_000_000,
            "max_observed_sustained_ns": max_sustained,
            "exceeded": max_sustained > 60_000_000_000,
        },
    }


def _process_observations(samples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Keep raw role/identity decisions visible without deduplicating CPU evidence."""
    rows = []
    for sample in samples:
        for process in sample["processes"]:
            row = {
                "sample_index": sample["index"],
                "coverage_complete": sample["coverage_complete"],
                **process,
            }
            rows.append(row)
    return rows


def _classification_changes(samples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    previous: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    changes = []
    for sample in samples:
        for process in sample["processes"]:
            key = (process["pid"], process.get("birth"), process.get("uid"))
            current = {
                **process.get("classification", {"kind": process["kind"]}),
                "command": process["command"],
                **({"argv_display": process["argv_display"]} if "argv_display" in process else {}),
            }
            old = previous.get(key)
            if old is not None and old != current:
                changes.append(
                    {
                        "sample_index": sample["index"],
                        "pid": process["pid"],
                        "previous_command": old.get("command"),
                        "command": process["command"],
                        "previous": dict(old),
                        "current": dict(current),
                    }
                )
            previous[key] = current
    return changes


def _application_coverage(
    samples: Sequence[Mapping[str, Any]], ticks: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for sample in samples:
        interval_cpu: dict[tuple[str, str], float] = {}
        for process in sample["processes"]:
            if process["kind"] not in {"desktop_app", "desktop_renderer"}:
                continue
            identity = process.get("classification", {})
            key = (str(identity.get("entry", "unknown")), str(identity.get("role", "unknown")))
            row = by_key.setdefault(
                key,
                {
                    "entry": key[0],
                    "application": identity.get("application"),
                    "role": key[1],
                    "kind": process["kind"],
                    "cpu_percent_max": 0.0,
                    "sample_indexes": [],
                    "coverage_complete_samples": 0,
                },
            )
            interval_cpu[key] = interval_cpu.get(key, 0.0) + process["cpu_percent"]
        for key, cpu in interval_cpu.items():
            row = by_key[key]
            row["cpu_percent_max"] = max(row["cpu_percent_max"], cpu)
            row["sample_indexes"].append(sample["index"])
            row["coverage_complete_samples"] += int(sample["coverage_complete"])
    for row in by_key.values():
        indexes = set(row["sample_indexes"])
        row["affected_tick_count"] = sum(
            bool(indexes & set(tick["sample_indexes"])) for tick in ticks
        )
    return list(by_key.values())


def desktop_overall(host_evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Independently reconcile the full desktop-policy host observation.

    This deliberately consumes every interval, including incomplete/churned
    ones.  The reducer uses it alongside per-cell alignment so the shared pool
    cannot be made quiet by retaining only favorable overlap windows.
    """
    if host_evidence.get("load_policy") != DESKTOP_POLICY:
        raise ValueError("desktop overall requires classified-desktop-v1")
    samples = _samples(host_evidence)
    _validate_desktop_guard(samples)
    applications = _application_coverage(samples, [])
    for application in applications:
        application.pop("affected_tick_count")
        application["observed_interval_count"] = len(application["sample_indexes"])
    return {
        "load_policy": DESKTOP_POLICY,
        "sample_count": len(samples),
        "samples": samples,
        "process_observations": _process_observations(samples),
        "classification_changes": _classification_changes(samples),
        "background_pool": _background_summary(samples),
        "applications": applications,
        "coverage_complete_count": sum(sample["coverage_complete"] for sample in samples),
    }


def _validate_desktop_guard(samples: Sequence[Mapping[str, Any]]) -> None:
    """Replay the interval guard; report generation cannot bless bad evidence."""
    for sample in samples:
        for process in sample["processes"]:
            classification = process.get("classification", {})
            nested = classification.get("classifier_decision", {})
            if (
                classification.get("reason") == "known_workload"
                or classification.get("known_workload")
                or (
                    isinstance(nested, Mapping)
                    and (nested.get("reason") == "known_workload" or nested.get("known_workload"))
                )
            ):
                raise ValueError("prohibited workload under desktop role in interval evidence")
            if classification.get("identity_binding_changed") or (
                isinstance(nested, Mapping) and nested.get("identity_binding_changed")
            ):
                raise ValueError("desktop identity binding changed in interval evidence")
            if process["kind"] not in BACKGROUND_KINDS and process["cpu_percent"] >= 20.0:
                raise ValueError("ordinary workload/unknown >=20% in desktop CPU evidence")
    shared = _background_summary(samples)["shared_background_pool"]
    if shared["exceeded"]:
        raise ValueError("desktop shared background >150% for >60s")


def validate_desktop_host(host_evidence: Mapping[str, Any]) -> None:
    """Fail closed before panel reduction if desktop interval evidence violates its guard."""
    from research.bl_served_search.desktop_apps import _valid_verified_identity

    if host_evidence.get("load_policy") != DESKTOP_POLICY:
        raise ValueError("desktop validation requires classified-desktop-v1")
    samples = _samples(host_evidence)
    _validate_desktop_guard(samples)
    registry_evidence = host_evidence["desktop_registry_evidence"]
    reg_dict = (
        registry_evidence.get("registry")
        if isinstance(registry_evidence.get("registry"), Mapping)
        else {}
    )
    has_apps = isinstance(reg_dict.get("applications"), list)
    log_by_proc = (
        _validate_registry_history(registry_evidence)
        if _has_full_desktop_history(registry_evidence)
        else None
    )
    observations = host_evidence.get("observations")
    if not isinstance(observations, list) or not observations:
        raise ValueError("desktop point observations missing")
    prev_enclosing_ns: int | None = None
    prev_cap_start_ns: int | None = None
    prev_screen_obs_ns: int | None = None
    prev_h_seq: int | None = None
    for observation in observations:
        if not isinstance(observation, Mapping):
            raise ValueError("malformed desktop point observation")
        if "monotonic_ns" in observation:
            enclosing_ns: int | None = _integer(
                observation["monotonic_ns"], "point observation monotonic_ns"
            )
        elif "observation_ns" in observation:
            enclosing_ns = _integer(observation["observation_ns"], "point observation_ns")
        else:
            enclosing_ns = None
        if enclosing_ns is not None:
            if prev_enclosing_ns is not None and enclosing_ns < prev_enclosing_ns:
                raise ValueError("desktop point observation clock regressed")
            prev_enclosing_ns = enclosing_ns
        screen = observation.get("process_screen", observation)
        if not isinstance(screen, Mapping):
            raise ValueError("malformed desktop point screen")
        cap_start: int | None = None
        cap_end: int | None = None
        if "capture_start_ns" in screen or "capture_end_ns" in screen:
            cap_start = _integer(screen.get("capture_start_ns"), "point capture_start_ns")
            cap_end = _integer(screen.get("capture_end_ns"), "point capture_end_ns")
            if cap_end < cap_start:
                raise ValueError("desktop point capture bracket inverted")
            if enclosing_ns is not None and cap_start < enclosing_ns:
                raise ValueError("desktop point capture precedes enclosing snapshot")
            if prev_cap_start_ns is not None and cap_start < prev_cap_start_ns:
                raise ValueError("desktop point capture inventory regressed")
            prev_cap_start_ns = cap_start
        screen_obs_ns: int | None = None
        if "observation_ns" in screen:
            screen_obs_ns = _integer(screen["observation_ns"], "point screen observation_ns")
            if (
                cap_start is not None
                and cap_end is not None
                and not (cap_start <= screen_obs_ns <= cap_end)
            ):
                raise ValueError("desktop point screen observation_ns outside capture bracket")
            if enclosing_ns is not None and screen_obs_ns < enclosing_ns:
                raise ValueError("desktop point screen precedes enclosing snapshot")
            if prev_screen_obs_ns is not None and screen_obs_ns < prev_screen_obs_ns:
                raise ValueError("desktop point screen observation_ns regressed")
            prev_screen_obs_ns = screen_obs_ns
        expected_seq_by_proc: dict[tuple[int, str], int] | None = None
        if log_by_proc is not None and "history_sequence" in screen:
            full_log = registry_evidence["history_log"]
            h_seq = _integer(screen["history_sequence"], "point screen history_sequence")
            if h_seq > len(full_log):
                raise ValueError("desktop point screen history_sequence out of bounds")
            if prev_h_seq is not None and h_seq < prev_h_seq:
                raise ValueError("desktop point screen history_sequence regressed")
            prev_h_seq = h_seq
            expected_seq_by_proc = {}
            for entry in full_log[:h_seq]:
                expected_seq_by_proc[(entry["pid"], entry["birth"])] = int(
                    entry["after"]["sequence"]
                )
        processes = screen.get("classified_processes")
        exempt = screen.get("probe_exempt_pids")
        if (
            not isinstance(processes, list)
            or not isinstance(exempt, list)
            or not all(type(pid) is int for pid in exempt)
        ):
            raise ValueError("desktop point screen evidence missing")
        exempt_pids = set(exempt)
        obs_level_ns = screen_obs_ns if screen_obs_ns is not None else enclosing_ns
        for process in processes:
            if not isinstance(process, Mapping) or type(process.get("pid")) is not int:
                raise ValueError("malformed desktop point process")
            if process["pid"] in exempt_pids:
                continue
            proc_obs_ns: int | None = None
            if "observation_ns" in process:
                proc_obs_ns = _integer(process["observation_ns"], "point process observation_ns")
                if (
                    cap_start is not None
                    and cap_end is not None
                    and not (cap_start <= proc_obs_ns <= cap_end)
                ):
                    raise ValueError("desktop point process observation_ns outside capture bracket")
                if screen_obs_ns is not None and proc_obs_ns != screen_obs_ns:
                    raise ValueError(
                        "desktop point process observation_ns differs from point screen"
                    )
            proc_key = (
                (process["pid"], process["birth"])
                if isinstance(process.get("birth"), str) and process["birth"]
                else None
            )
            expected_point_seq = (
                expected_seq_by_proc.get(proc_key, -1)
                if (expected_seq_by_proc is not None and proc_key is not None)
                else None
            )
            classification = process.get("classification")
            replay = _replay_kind(
                process,
                registry_evidence,
                log_by_proc,
                is_point=True,
                point_obs_ns=obs_level_ns,
                expected_point_sequence=expected_point_seq,
            )
            if proc_obs_ns is not None and enclosing_ns is not None and proc_obs_ns < enclosing_ns:
                raise ValueError(
                    "desktop process state superseded before observation: "
                    "point process precedes enclosing snapshot"
                )
            if (
                not isinstance(classification, Mapping)
                or classification.get("kind") != process.get("kind")
                or replay.get("kind") != process.get("kind")
            ):
                raise ValueError("desktop point classification replay differs")
            if (
                process.get("kind") in {"desktop_app", "desktop_renderer"}
                and has_apps
                and not _valid_verified_identity(
                    str(process.get("command", "")),
                    classification.get("identity"),
                    reg_dict,
                )
            ):
                raise ValueError("desktop point process identity does not match sealed registry")
            if (
                classification.get("identity_binding_changed")
                or classification.get("reason") == "known_workload"
            ):
                if classification.get("identity_binding_changed"):
                    raise ValueError("desktop identity binding changed in point evidence")
                raise ValueError("desktop point prohibited workload evidence")
            cpu = _number(process.get("cpu_percent"), "desktop point CPU")
            if process["kind"] not in BACKGROUND_KINDS and cpu >= 20.0:
                raise ValueError("ordinary workload/unknown >=20% in desktop point evidence")


def _measured_rows(result: Mapping[str, Any], production: bool) -> list[Mapping[str, Any]]:
    rows = result.get("trace")
    if not isinstance(rows, list):
        raise ValueError("missing trace")
    _integer(result.get("warmup_ticks", 0), "warmup_ticks")
    # Each arm exports its trace after warmup.  Keep the asserted warmup field
    # as provenance, but never slice trace again here.
    if not rows:
        raise ValueError("no measured trace rows")
    return rows


def _production_delivery(
    result: Mapping[str, Any], count: int
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    warmup = _integer(result.get("warmup_ticks", 0), "warmup_ticks")
    receipts = result.get("client_receive")
    sends = result.get("frame_send")
    if not isinstance(receipts, list) or not isinstance(sends, list):
        raise ValueError("missing production delivery evidence")
    receipts = [
        row for row in receipts if isinstance(row, Mapping) and row.get("source") == "hub_broadcast"
    ]
    if len(receipts) != warmup + count or len(sends) != warmup + count:
        raise ValueError("incomplete production delivery evidence")
    measured_receipts, measured_sends = list(receipts[warmup:]), list(sends[warmup:])
    for receipt, send in zip(measured_receipts, measured_sends):
        if receipt.get("frame") != send.get("frame") or send.get("status") != "ok":
            raise ValueError("misaligned production delivery evidence")
        if _integer(receipt.get("receive_ns"), "client receive") < _integer(
            send.get("start_ns"), "send start"
        ):
            raise ValueError("client receipt precedes send")
    return measured_receipts, measured_sends


def _row_window(
    row: Mapping[str, Any], index: int, production: bool, receipt: Mapping[str, Any] | None
) -> tuple[int, int]:
    target = _integer(row.get("target_ns"), "trace target")
    wake = _integer(row.get("wake_ns"), "trace wake")
    start = _integer(row.get("start_ns"), "trace start")
    end = _integer(row.get("end_ns"), "trace end")
    if end < start:
        raise ValueError("trace end precedes start")
    window_start = min(target, wake, start)
    window_end = end
    if production:
        snapshot_end = _integer(row.get("snapshot_end_ns"), "snapshot end")
        if snapshot_end < start:
            raise ValueError("snapshot end precedes trace start")
        if receipt is None:
            raise ValueError("missing production receipt")
        receive = _integer(receipt.get("receive_ns"), "client receive")
        window_end = max(window_end, snapshot_end, receive)
    if window_end <= window_start:
        raise ValueError(f"empty trace window at tick {index}")
    return window_start, window_end


def _covariate(samples: Sequence[Mapping[str, Any]], start: int, end: int) -> dict[str, Any]:
    overlaps = []
    covered = start
    complete = True
    observed_weight = 0
    reliable_weight = 0
    weighted = {
        "background": 0.0,
        "service": 0.0,
        "desktop_renderer": 0.0,
        "desktop_app": 0.0,
        "workload": 0.0,
        "unknown": 0.0,
    }
    maximum = {category: None for category in weighted}
    for sample in samples:
        if sample["end_ns"] <= start or sample["start_ns"] >= end:
            continue
        overlap_start, overlap_end = max(start, sample["start_ns"]), min(end, sample["end_ns"])
        if overlap_start > covered:
            complete = False
        covered = max(covered, overlap_end)
        duration = overlap_end - overlap_start
        usable = (
            sample["coverage_complete"] and sample["end_ns"] - sample["start_ns"] <= MAX_SAMPLE_NS
        )
        overlaps.append(sample["index"])
        observed_weight += duration
        for category in weighted:
            value = sample[f"{category}_cpu_percent"]
            weighted[category] += value * duration
            maximum[category] = max(maximum[category] or 0.0, value)
        if usable:
            reliable_weight += duration
        else:
            complete = False
    if covered < end:
        complete = False
    coverage_fraction = reliable_weight / (end - start)
    means = {
        category: weighted[category] / observed_weight if observed_weight else None
        for category in weighted
    }
    low = bool(
        complete
        and coverage_fraction == 1.0
        and means["background"] is not None
        and maximum["background"] is not None
        and maximum["background"] <= LOW_BACKGROUND_PERCENT
        and all(
            sample["background_cpu_percent"] <= LOW_BACKGROUND_PERCENT
            and sample["workload_cpu_percent"] < LOW_BACKGROUND_PERCENT
            and sample["unknown_cpu_percent"] < LOW_BACKGROUND_PERCENT
            for sample in samples
            if sample["index"] in overlaps
            and sample["coverage_complete"]
            and sample["end_ns"] - sample["start_ns"] <= MAX_SAMPLE_NS
        )
    )
    return {
        "sample_indexes": overlaps,
        "coverage_complete": complete and coverage_fraction == 1.0,
        "coverage_fraction": coverage_fraction,
        **{f"{category}_cpu_percent_time_weighted_mean": means[category] for category in means},
        **{f"{category}_cpu_percent_max": maximum[category] for category in maximum},
        "low_background": low,
    }


def _metric_rows(
    rows: list[Mapping[str, Any]],
    production: bool,
    receipts: list[Mapping[str, Any]],
    sends: list[Mapping[str, Any]],
    fps: int = 24,
) -> list[dict[str, int]]:
    if fps <= 0:
        raise ValueError("fps must be positive")
    values: list[dict[str, int]] = []
    for index, row in enumerate(rows):
        item: dict[str, int] = {}
        for name, start_key, end_key in (
            ("tick_wall_ns", "start_ns", "end_ns"),
            ("tick_thread_cpu_ns", "start_cpu_ns", "end_cpu_ns"),
        ):
            start, end = _integer(row.get(start_key), start_key), _integer(
                row.get(end_key), end_key
            )
            if end < start:
                raise ValueError(f"negative {name}")
            item[name] = end - start
        target, wake = _integer(row.get("target_ns"), "target_ns"), _integer(
            row.get("wake_ns"), "wake_ns"
        )
        item["wake_lateness_ns"] = max(0, wake - target)
        update = (
            row.get("spans", {}).get("game_update")
            if isinstance(row.get("spans"), Mapping)
            else None
        )
        if isinstance(update, Mapping):
            item["update_ns"] = _integer(update.get("wall_ns"), "game update wall")
        if not production:
            epoch = _integer(rows[0].get("target_ns"), "epoch target")
            deadline = epoch + (_integer(row.get("tick"), "trace tick") + 1) * 1_000_000_000 // fps
            item["absolute_overrun_severity_ns"] = max(
                0, _integer(row["end_ns"], "end_ns") - deadline
            )
            item["absolute_overrun"] = int(_integer(row["end_ns"], "end_ns") >= deadline)
        else:
            snapshot_start = _integer(row.get("snapshot_start_cpu_ns"), "snapshot start CPU")
            snapshot_end = _integer(row.get("snapshot_end_cpu_ns"), "snapshot end CPU")
            if snapshot_end < snapshot_start:
                raise ValueError("negative snapshot CPU")
            item["snapshot_thread_cpu_ns"] = snapshot_end - snapshot_start
            receive = _integer(receipts[index].get("receive_ns"), "client receive")
            item["step_to_client_ns"] = receive - _integer(row.get("start_ns"), "start_ns")
            if item["step_to_client_ns"] < 0:
                raise ValueError("client receipt precedes step")
            item["send_to_client_ns"] = receive - _integer(
                sends[index].get("start_ns"), "send start"
            )
        values.append(item)
    return values


def _timing(
    metrics: list[dict[str, int]],
    indexes: list[int],
    metric_keys: list[str],
    receipts: list[Mapping[str, Any]],
    samples: Sequence[Mapping[str, Any]],
    *,
    require_low_gap: bool,
) -> dict[str, Any]:
    selected = [metrics[index] for index in indexes]
    distributions = {
        key: _distribution([item[key] for item in selected if key in item])
        for key in metric_keys
        if key != "absolute_overrun"
    }
    # Delivery gaps are only valid for consecutive retained trace rows.  Do
    # not bridge a high-background or coverage-excluded tick in the low subset.
    receipt_indexes = indexes if receipts else []
    gaps = [
        _integer(receipts[current].get("receive_ns"), "client receive")
        - _integer(receipts[previous].get("receive_ns"), "client receive")
        for previous, current in zip(receipt_indexes, receipt_indexes[1:])
        if current == previous + 1
        and (
            not require_low_gap
            or receipts[current]["receive_ns"] == receipts[previous]["receive_ns"]
            or (
                receipts[current]["receive_ns"] > receipts[previous]["receive_ns"]
                and _covariate(
                    samples,
                    _integer(receipts[previous].get("receive_ns"), "client receive"),
                    _integer(receipts[current].get("receive_ns"), "client receive"),
                )["low_background"]
            )
        )
    ]
    if any(gap < 0 for gap in gaps):
        raise ValueError("client receipt ordering")
    if receipts:
        distributions["client_frame_gap_ns"] = _distribution(gaps)
    return {
        "count": len(selected),
        "absolute_overrun_count": sum(item.get("absolute_overrun", 0) for item in selected),
        "metrics": distributions,
    }


def summarize_background(
    result: Mapping[str, Any],
    host_evidence: Mapping[str, Any],
    *,
    production: bool = False,
    fps: int = 24,
) -> dict[str, Any]:
    """Align host CPU samples and publish overall plus low-background timings.

    This is a covariate report only: it neither filters the diagnostic reducer nor
    changes its verdict. Missing coverage remains explicit and cannot become a
    low-background observation by interpolation.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    if not isinstance(result, Mapping) or not isinstance(host_evidence, Mapping):
        raise ValueError("result and host evidence must be mappings")
    samples = _samples(host_evidence)
    policy = host_evidence["load_policy"]
    rows = _measured_rows(result, production)
    receipts, sends = _production_delivery(result, len(rows)) if production else ([], [])
    ticks = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError("malformed trace row")
        start, end = _row_window(row, index, production, receipts[index] if production else None)
        tick = {
            "index": index,
            "tick": _integer(row.get("tick"), "trace tick"),
            "window_start_ns": start,
            "window_end_ns": end,
        }
        tick.update(_covariate(samples, start, end))
        ticks.append(tick)
    metrics = _metric_rows(rows, production, receipts, sends, fps=fps)
    metric_keys = sorted({key for item in metrics for key in item})
    all_indexes = list(range(len(rows)))
    low_indexes = [tick["index"] for tick in ticks if tick["low_background"]]
    scoped_indexes = {sample_index for tick in ticks for sample_index in tick["sample_indexes"]}
    scoped_samples = [sample for sample in samples if sample["index"] in scoped_indexes]
    return {
        "schema": "served-tick-background-report/v1",
        "load_policy": policy,
        "production": production,
        "low_background_definition": {
            "background_cpu_percent_max": LOW_BACKGROUND_PERCENT,
            "workload_cpu_percent_exclusive_max": LOW_BACKGROUND_PERCENT,
            "unknown_cpu_percent_exclusive_max": LOW_BACKGROUND_PERCENT,
            "requires_complete_coverage": True,
            "max_sample_interval_ns": MAX_SAMPLE_NS,
        },
        "ticks": ticks,
        "process_observations": _process_observations(scoped_samples),
        "classification_changes": _classification_changes(scoped_samples),
        "background_pool": _background_summary(scoped_samples),
        "applications": _application_coverage(scoped_samples, ticks),
        "counts": {
            "ticks": len(ticks),
            "coverage_eligible": sum(tick["coverage_complete"] for tick in ticks),
            "low_background": len(low_indexes),
            "high_background": sum(
                (tick["background_cpu_percent_max"] or 0) > LOW_BACKGROUND_PERCENT for tick in ticks
            ),
            "high_workload": sum(
                (tick["workload_cpu_percent_max"] or 0) >= LOW_BACKGROUND_PERCENT for tick in ticks
            ),
            "high_unknown": sum(
                (tick["unknown_cpu_percent_max"] or 0) >= LOW_BACKGROUND_PERCENT for tick in ticks
            ),
        },
        "timing": {
            "overall": _timing(
                metrics, all_indexes, metric_keys, receipts, samples, require_low_gap=False
            ),
            "low_background": _timing(
                metrics, low_indexes, metric_keys, receipts, samples, require_low_gap=True
            ),
        },
        "interpretation": (
            "Background load is an observational covariate only; these summaries establish neither "
            "causation nor a timing gate."
        ),
    }
