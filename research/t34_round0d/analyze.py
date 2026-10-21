"""The sole fixed-N Round 0-D decision; Round 0's bars (imported) + the relative boost floor.

``research.t34_round0.analyze.summarize`` (unchanged) computes every clause and
the GO/STOP/INCONCLUSIVE order, so Round 0-D cannot drift from Round 0's bars.
This module only adds Round 0-D provenance checks (its own schemas, exports,
second-L4 replay) and the frozen descriptive decomposition (plan §3.1), which is
reported and never gates.
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
from collections import deque
from pathlib import Path
from typing import Any, Mapping

from research.t34_round0.analyze import (
    HORIZON,
    MIXES,
    InvalidEvidence,
    _bound_path,
    _integer,
    inconclusive,
    summarize,
)
from research.t34_round0_amend.rule import amend, pooled_boosts
from research.t34_round0d.common import B8_SHA, PACKAGE, ROOT, read_json, sha256

SCHEMA = "t34-round0d-decision/v1"
OUTPUT = "round0d"
T3_SCRIPTED_NO_DEATH_GAIN_B8_S3 = 0.28  # E1, docs/research/t3_ceiling_test_2026-10-15.md


def _admission(chunk: Path, identity: Mapping[str, Any], repo: Path) -> dict[str, Any]:
    """Revalidate supervisor publication and terminal success (Round 0 semantics)."""
    chunk = chunk.resolve()
    if chunk.name != OUTPUT or chunk.parent.name != "exports":
        raise InvalidEvidence("chunks must be supervisor job/exports/round0d directories")
    job = chunk.parent.parent
    receipt = read_json(job / "receipt.json")
    manifest_path = job / "manifest.json"
    manifest = read_json(manifest_path)
    terminal = read_json(job / "terminal.json")
    fault_terminal = receipt.get("terminal") is None and (chunk / "fault.json").is_file()
    if fault_terminal:
        polls = [read_json(path).get("terminal") for path in sorted(job.glob("poll-*.json"))]
        if terminal not in polls or receipt.get("stop_verified") is not True:
            raise InvalidEvidence("fault terminal lacks preserved telemetry and verified release")
    if (
        receipt.get("manifest_sha256") != sha256(manifest_path)
        or receipt.get("commit") != identity["commit"]
        or manifest.get("commit") != identity["commit"]
        or receipt.get("job_id") != manifest.get("job_id")
        or (not fault_terminal and terminal != receipt.get("terminal"))
        or terminal.get("generation") != receipt.get("generation")
        or terminal.get("job_id") != manifest.get("job_id")
    ):
        raise InvalidEvidence("supervisor receipt/manifest/terminal identity mismatch")
    if receipt.get("supervisor_sha256") != sha256(repo / "research/colab_cli/supervisor.py"):
        raise InvalidEvidence("receipt producer differs from registered supervisor source")
    inputs = {item.get("dest"): item.get("sha256") for item in manifest.get("inputs", [])}
    for dest, key in (
        ("BANK_ROUND0D.json", "bank_sha256"),
        ("SEED_AUDIT_ROUND0D.json", "audit_sha256"),
        ("RATIFICATION.json", "ratification_sha256"),
    ):
        if inputs.get(dest) != identity[key]:
            raise InvalidEvidence("manifest input differs from registered analysis input")
    inventory = terminal.get("exports") if fault_terminal else receipt.get("exports")
    if not isinstance(inventory, list) or sorted(inventory, key=lambda i: i["path"]) != sorted(
        terminal.get("exports") or [], key=lambda i: i["path"]
    ):
        raise InvalidEvidence("supervisor export inventories differ")
    exported = {}
    for item in inventory:
        path = _bound_path(job / "exports", item)
        sidecar = Path(str(path) + ".sha256")
        if (
            path in exported
            or item.get("size") != path.stat().st_size
            or not Path(str(path) + ".complete").is_file()
            or not sidecar.is_file()
            or sidecar.read_text().split()[0] != item["sha256"]
        ):
            raise InvalidEvidence("supervisor export publication is incomplete")
        exported[path] = item["sha256"]
    return dict(manifest=manifest, exported=exported, receipt=receipt, terminal=terminal)


def _published(path: Path, admission: Mapping[str, Any]) -> None:
    if admission["exported"].get(path.resolve()) != sha256(path):
        raise InvalidEvidence("artifact is absent from immutable supervisor exports")


def descriptive(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Plan §3.1 decomposition: reported only, never a gate, never changes the branch."""
    out: dict[str, Any] = {"authority": "DESCRIPTIVE_ONLY_NEVER_A_GATE", "per_mix": {}}
    for mix in MIXES:
        rows = [r for r in records if r["world"]["mix"] == mix]
        teacher = [r["arms"]["teacher"] for r in rows]
        bare = [r["arms"]["bare"] for r in rows]
        deaths = [t for t in teacher if not t["no_death"]]
        out["per_mix"][mix] = {
            "paired_mass_delta_mean": statistics.fmean(
                t["mass_integral"] - b["mass_integral"] for t, b in zip(teacher, bare)
            ),
            "paired_alive_delta_mean": statistics.fmean(
                t["survival_fraction"] - b["survival_fraction"] for t, b in zip(teacher, bare)
            ),
            "paired_no_death_delta_mean": statistics.fmean(
                t["no_death"] - b["no_death"] for t, b in zip(teacher, bare)
            ),
            "teacher_boost_share": sum(t["boost_decisions"] for t in teacher)
            / sum(t["decisions"] for t in teacher),
            "interventions_per_world": statistics.fmean(
                t["dense"]["action_changes"] for t in teacher
            ),
            "batches_per_world": statistics.fmean(t["dense"]["batches"] for t in teacher),
            "rescue_changes_per_world": statistics.fmean(
                t["dense"]["rescue_changes"] for t in teacher
            ),
            "delay_changes_per_world": statistics.fmean(
                t["dense"]["delay_changes"] for t in teacher
            ),
            "teacher_deaths": len(deaths),
            "teacher_deaths_with_executed_rescue": sum(
                d["dense"]["rescue_changes"] > 0 for d in deaths
            ),
            "teacher_deaths_without_any_rescue": sum(
                d["dense"]["rescue_changes"] == 0 for d in deaths
            ),
        }
    out["scripted_no_death_gain"] = out["per_mix"]["scripted"]["paired_no_death_delta_mean"]
    out["t3_reference_scripted_no_death_gain_b8_s3"] = T3_SCRIPTED_NO_DEATH_GAIN_B8_S3
    return out


def _window_faults(
    bank_rows: list[dict[str, Any]], streams: dict[str, dict[str, list[int]]]
) -> list:
    faults = []
    for arm in ("bare", "teacher"):
        window: deque[int] = deque()
        counts = [0] * 6
        for row in bank_rows:
            hit = False
            for action in streams[row["world_id"]][arm]:
                if len(window) == 100000:
                    counts[window.popleft()] -= 1
                window.append(action)
                counts[action] += 1
                if len(window) == 100000 and max(counts) > 90000:
                    faults.append(
                        dict(world_id=row["world_id"], arm=arm, fault="action_collapse_100k")
                    )
                    hit = True
                    break
            if hit:
                break
    return faults


def analyze_paths(  # noqa: C901 - mirrors Round 0's fail-closed consumer
    chunks: list[Path], ratification_path: Path, replay_path: Path, *, repo: Path | None = None
) -> dict[str, Any]:
    """Only production decision entrypoint: complete, immutable provenance or no verdict."""
    from research.t34_pathb.source_binding_v2 import verify_frozen_worktree
    from research.t34_round0d.history import (
        CapExpiredHistory,
        HistoryError,
        require_active_phase,
        require_chunks,
        selected_rows,
    )
    from research.t34_round0d.registration import require_registration

    repo = repo or ROOT
    bank_path, audit_path = PACKAGE / "BANK_ROUND0D.json", PACKAGE / "SEED_AUDIT_ROUND0D.json"
    try:
        ratification = read_json(ratification_path)
        bank = require_registration(ratification, bank_path, audit_path, repo=repo, mac_side=True)
        require_active_phase(ratification, "analysis")
        verify_frozen_worktree(repo, ratification["commit"])
        try:
            require_chunks(chunks, ratification)
        except FileNotFoundError as exc:
            raise HistoryError("unfinished claim is missing its terminal history") from exc
        identity = dict(
            bank_sha256=sha256(bank_path),
            audit_sha256=sha256(audit_path),
            ratification_sha256=sha256(ratification_path),
            checkpoint_sha256=B8_SHA,
            commit=ratification["commit"],
        )
        expected = {row["world_id"]: row for row in bank["rows"]}
        if not chunks or len({p.resolve() for p in chunks}) != len(chunks):
            raise InvalidEvidence("missing or duplicate chunks")
        admissions = []
        for chunk_root in chunks:
            admission = _admission(chunk_root, identity, repo)
            fault_path = chunk_root / "fault.json"
            if fault_path.is_file():
                _published(fault_path, admission)
                fault = read_json(fault_path)
                if (
                    fault.get("schema") != "t34-round0d-fault/v1"
                    or fault.get("preflight") is not False
                    or fault.get("unconditional") is not True
                    or fault.get("fault")
                    not in {
                        "illegal_action",
                        "nonfinite_q",
                        "nonfinite_loss",
                        "action_collapse_100k",
                    }
                    or fault.get("arm") not in {"bare", "teacher"}
                    or fault.get("world_id") not in expected
                    or any(fault.get("run_identity", {}).get(k) != v for k, v in identity.items())
                ):
                    raise InvalidEvidence("fault receipt lacks registered unconditional identity")
                return dict(
                    schema=SCHEMA,
                    decision="STOP",
                    reason="unconditional fault; this teacher/controller failed its registered bar",
                    evidence_authority="TIER_1_DEVELOPMENT",
                    release_authorized=False,
                    unconditional_faults=[fault],
                    fault_sha256=sha256(fault_path),
                )
            admissions.append((chunk_root, admission))
        records, evidence, streams, chunk_hashes, runtimes = [], {}, {}, {}, set()
        for chunk_root, admission in admissions:
            receipt_path = chunk_root / "chunk.json"
            _published(receipt_path, admission)
            terminal, transport = admission["terminal"], admission["receipt"]
            if (
                transport.get("status") != "PASS"
                or transport.get("stop_verified") is not True
                or transport.get("error") is not None
                or transport.get("cleanup_error") is not None
                or terminal.get("status") not in {"completed", "success", "succeeded", "PASS"}
                or terminal.get("returncode") != 0
                or terminal.get("reason") is not None
            ):
                raise InvalidEvidence("supervisor did not finish and verify runtime release")
            receipt = read_json(receipt_path)
            run_identity = receipt.get("run_identity", {})
            if (
                receipt.get("schema") != "t34-round0d-chunk/v1"
                or receipt.get("status") != "complete"
                or receipt.get("preflight") is not False
                or any(run_identity.get(k) != v for k, v in identity.items())
            ):
                raise InvalidEvidence("chunk receipt is not a registered scientific completion")
            runtime_id = run_identity.get("runtime_id")
            runtimes.add(runtime_id)
            allocated, entries = receipt["slice"]["count"], receipt["pairs"]
            if (allocated - len(entries)) / allocated > 0.02:
                raise InvalidEvidence("STOP: >2% dropped worlds in a chunk; preserve and report")
            chunk_hashes[str(receipt_path)] = sha256(receipt_path)
            for entry in entries:
                pair_path = _bound_path(chunk_root, entry)
                _published(pair_path, admission)
                pair = read_json(pair_path)
                world = pair["world"]
                world_id = world["world_id"]
                if (
                    pair.get("schema") != "t34-round0d-pair/v1"
                    or pair.get("status") != "complete"
                    or pair.get("preflight") is not False
                    or pair["run_identity"].get("runtime_id") != runtime_id
                    or any(pair["run_identity"].get(k) != v for k, v in identity.items())
                    or world_id not in expected
                    or world_id in evidence
                    or world != expected[world_id]
                    or world.get("horizon") != HORIZON
                ):
                    raise InvalidEvidence("pair identity, runtime, or registered world differs")
                traces, streams[world_id] = {}, {}
                for name in ("bare", "teacher"):
                    arm = pair["arms"][name]
                    trace_path = _bound_path(chunk_root, arm["trace"])
                    _published(trace_path, admission)
                    trace = read_json(trace_path)
                    if (
                        trace.get("schema") != "t34-exit-world/v1"
                        or trace.get("world_id") != world_id
                        or trace.get("row") != expected[world_id]
                        or trace.get("frames") != HORIZON
                        or trace.get("beta") != (0.0 if name == "bare" else 1.0)
                        or trace.get("checkpoint_sha256") != B8_SHA
                        or trace.get("labels") != []
                        or trace.get("complete") is not True
                    ):
                        raise InvalidEvidence("trace world identity mismatch or labels present")
                    actions = trace["actions"]
                    counts = [0] * 6
                    for frame in actions:
                        if not isinstance(frame, list) or len(frame) != 6:
                            raise InvalidEvidence("trace requires six actions per frame")
                        for action in frame:
                            _integer(action, "trace action", 0, 5)
                        counts[frame[0]] += 1
                    if len(actions) != arm["decisions"] or counts != arm["action_counts"]:
                        raise InvalidEvidence("trace hero actions differ from arm metrics")
                    required = {"0", str(len(actions))} | {
                        str(f) for f in range(250, len(actions) + 1, 250)
                    }
                    if (
                        set(trace["hashes"]) != required
                        or trace["hashes"][str(len(actions))] != arm["trace"]["final_state_hash"]
                    ):
                        raise InvalidEvidence("trace state-hash inventory or terminal hash differs")
                    streams[world_id][name] = [frame[0] for frame in actions]
                    traces[name] = sha256(trace_path)
                evidence[world_id] = dict(
                    world_id=world_id, pair_sha256=sha256(pair_path), traces=traces
                )
                records.append(pair)
        if set(evidence) != set(expected) or len(records) != 240:
            raise InvalidEvidence("fixed look lacks exactly the whole registered 240-world bank")
        replay = read_json(replay_path)
        selected = [
            row["world_id"]
            for row in selected_rows(bank["rows"], [read_json(c / "chunk.json") for c in chunks])
        ]
        if (
            replay.get("schema") != "t34-round0d-replay/v1"
            or replay.get("status") != "PASS"
            or replay.get("divergences") != []
            or replay.get("selected_ids") != selected
            or replay.get("platform", {}).get("system") != "Linux"
            or replay.get("replay_runtime_id") in runtimes
            or sorted(runtimes) != replay.get("producer_runtime_ids")
            or replay.get("checked_pairs") != [evidence[world_id] for world_id in selected]
        ):
            raise InvalidEvidence("missing cross-runtime L4 replay or unexplained divergence")
        registered = summarize(records)
        # Owner-approved relative boost floor (research/t34_round0_amend/rule.py, pinned):
        # 0.5 x paired bare pooled share <= teacher pooled share <= 0.10. Only the
        # boost clause is re-evaluated; Round 0's decision order is otherwise unchanged.
        result = amend(
            registered, pooled_boosts(records, "teacher"), pooled_boosts(records, "bare")
        )
        result["schema"] = SCHEMA
        result["registered_round0_clause_decision"] = registered["decision"]
        result["boost_rule"] = "relative floor (research/t34_round0_amend/rule.py)"
        faults = _window_faults(bank["rows"], streams)
        if faults:
            result.update(
                decision="STOP",
                reason="unconditional action-collapse fault; this teacher/controller failed its "
                "registered bar",
            )
            result["unconditional_faults"].extend(faults)
        result.update(
            evidence_authority="TIER_1_DEVELOPMENT",
            release_authorized=False,
            bank_sha256=identity["bank_sha256"],
            chunk_sha256=chunk_hashes,
            replay_sha256=sha256(replay_path),
            ratification_sha256=identity["ratification_sha256"],
            descriptive=descriptive(records),
            sealed_labels_admissible_to_r1=result["decision"] == "GO",
        )
        return result
    except CapExpiredHistory as exc:
        return dict(inconclusive(str(exc)), schema=SCHEMA)
    except HistoryError as exc:
        return dict(
            schema=SCHEMA,
            decision="KILL",
            reason=str(exc),
            evidence_authority="INTEGRITY_FAILURE",
            release_authorized=False,
        )
    except (
        ValueError,
        KeyError,
        TypeError,
        OSError,
        ArithmeticError,
        subprocess.SubprocessError,
        IndexError,
    ) as exc:
        return dict(inconclusive(str(exc)), schema=SCHEMA)


def main() -> None:
    from research.t34_round0d.history import policy, run_phase

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", nargs="+", type=Path, required=True)
    parser.add_argument("--ratification", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    registration = read_json(args.ratification)
    root, _, _ = policy(registration)
    if args.out != root / "analysis/decision.json" or args.replay != root / "replay/replay.json":
        parser.error("analysis must use the fixed replay and analysis/decision.json paths")
    run_phase(
        registration,
        "analysis",
        lambda _out: analyze_paths(args.chunks, args.ratification, args.replay),
    )


if __name__ == "__main__":
    main()
