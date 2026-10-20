"""Fixed-N Round 0 statistics and fail-closed artifact consumer.

A numerical summary is explicitly synthetic/unverified until analyze_paths validates
registered immutable chunks, bank identity, and the prospective Mac replay sample.
"""

from __future__ import annotations

import argparse
import math
import statistics
import subprocess
from collections import deque
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.scripts.eval_stats import student_t_isf

MIXES = ("scripted", "frozen", "mixed")
COUNTS = {"scripted": 96, "frozen": 72, "mixed": 72}
HORIZON = 5000


class InvalidEvidence(ValueError):
    """Evidence cannot support the registered look."""


def _finite(
    value: Any, name: str, minimum: float | None = None, maximum: float | None = None
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise InvalidEvidence(f"{name} must be finite numeric")
    if (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
        raise InvalidEvidence(f"{name} out of range")
    return float(value)


def _integer(value: Any, name: str, minimum: int = 0, maximum: int | None = None) -> int:
    number = _finite(value, name, minimum, maximum)
    if int(number) != number:
        raise InvalidEvidence(f"{name} must be integer")
    return int(number)


def _interval(mean: float, se: float, df: float, tail: float, n: int) -> dict[str, Any]:
    margin = student_t_isf(tail, df) * se
    return dict(
        n=n, mean=mean, se=se, df=df, lower=mean - margin, upper=mean + margin, tail_alpha=tail
    )


def _ci(values: Iterable[float], tail: float) -> dict[str, Any]:
    values = [_finite(x, "observation") for x in values]
    if len(values) < 2:
        raise InvalidEvidence("at least two paired worlds required")
    return _interval(
        statistics.fmean(values),
        statistics.stdev(values) / math.sqrt(len(values)),
        len(values) - 1,
        tail,
        len(values),
    )


def _stratified(groups: Mapping[str, list[float]], tail: float) -> dict[str, Any]:
    variances = [statistics.variance(groups[m]) / len(groups[m]) / 9 for m in MIXES]
    variance = sum(variances)
    denominator = sum(v * v / (len(groups[m]) - 1) for m, v in zip(MIXES, variances))
    df = variance * variance / denominator if denominator else sum(map(len, groups.values())) - 3
    return _interval(
        statistics.fmean(statistics.fmean(groups[m]) for m in MIXES),
        math.sqrt(variance),
        df,
        tail,
        sum(map(len, groups.values())),
    )


def _validate_arm(arm: Mapping[str, Any]) -> None:
    _finite(arm["mass_integral"], "mass integral", 0)
    survival = _finite(arm["survival_fraction"], "survival fraction", 0, 1)
    no_death = _integer(arm["no_death"], "no-death", 0, 1)
    deaths = _integer(arm["deaths"], "terminal deaths", 0, 1)
    decisions = _integer(arm["decisions"], "decisions", 1, HORIZON)
    boosts = _integer(arm["boost_decisions"], "boost decisions", 0, decisions)
    if no_death != int(deaths == 0) or (no_death and survival != 1):
        raise InvalidEvidence("terminal death/survival counters do not reconcile")
    counts = arm["action_counts"]
    if not isinstance(counts, list) or len(counts) != 6:
        raise InvalidEvidence("missing six-action counts")
    counts = [_integer(c, "action count", 0, decisions) for c in counts]
    if sum(counts) != decisions or sum(counts[3:]) != boosts:
        raise InvalidEvidence("action counters do not reconcile")
    faults = arm["unconditional_faults"]
    if not isinstance(faults, list) or any(not isinstance(f, str) or not f for f in faults):
        raise InvalidEvidence("missing unconditional-fault evidence")
    for field, expected in (
        ("boost_share", boosts / decisions),
        ("one_action_share", max(counts) / decisions),
    ):
        if field in arm and not math.isclose(
            _finite(arm[field], field, 0, 1), expected, abs_tol=1e-12
        ):
            raise InvalidEvidence(f"{field} does not reconcile")


def _validate_records(records: list[Mapping[str, Any]]) -> None:
    counts = dict.fromkeys(MIXES, 0)
    seen_ids, seen_seeds = set(), set()
    for row in records:
        world = row["world"]
        mix, seed, world_id = world["mix"], world["world_seed"], world["world_id"]
        if mix not in MIXES or world.get("horizon", HORIZON) != HORIZON:
            raise InvalidEvidence("wrong mix or horizon")
        _integer(seed, "world seed")
        if not isinstance(world_id, str) or not world_id:
            raise InvalidEvidence("missing world identity")
        if seed in seen_seeds or world_id in seen_ids:
            raise InvalidEvidence("duplicate world or non-disjoint seed")
        seen_seeds.add(seed)
        seen_ids.add(world_id)
        counts[mix] += 1
        if set(row["arms"]) != {"bare", "teacher"}:
            raise InvalidEvidence("missing or unexpected arm")
        for arm in row["arms"].values():
            _validate_arm(arm)
    if counts != COUNTS:
        raise InvalidEvidence("fixed look requires exactly 96/72/72 paired worlds")


def summarize(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Pure arithmetic only; this function grants no scientific evidence authority."""
    _validate_records(records)
    groups = {m: [r for r in records if r["world"]["mix"] == m] for m in MIXES}
    deltas = {
        metric: {
            m: [r["arms"]["teacher"][metric] - r["arms"]["bare"][metric] for r in groups[m]]
            for m in MIXES
        }
        for metric in ("mass_integral", "survival_fraction", "no_death")
    }
    mass = _stratified(deltas["mass_integral"], 0.10)
    alive = _stratified(deltas["survival_fraction"], 0.05)
    per_mix = {
        m: dict(
            mass=_ci(deltas["mass_integral"][m], 0.10),
            alive=_ci(deltas["survival_fraction"][m], 0.05),
            no_death=_ci(deltas["no_death"][m], 0.10),
            teacher_no_death=statistics.fmean(r["arms"]["teacher"]["no_death"] for r in groups[m]),
            teacher_alive_fraction=statistics.fmean(
                r["arms"]["teacher"]["survival_fraction"] for r in groups[m]
            ),
        )
        for m in MIXES
    }
    boosts = sum(r["arms"]["teacher"]["boost_decisions"] for r in records)
    decisions = sum(r["arms"]["teacher"]["decisions"] for r in records)
    boost = boosts / decisions
    clauses = {
        "scripted_no_death_ge_066": per_mix["scripted"]["teacher_no_death"] >= 0.66,
        "scripted_no_death_gain_ge_010": per_mix["scripted"]["no_death"]["mean"] >= 0.10,
        "mass_gain_ge_40": mass["mean"] >= 40,
        "mass_lb90_positive": mass["lower"] > 0,
        "survival_guard_a_pooled": alive["lower"] > -0.05,
        "survival_guard_a_each_mix": all(per_mix[m]["alive"]["lower"] > -0.075 for m in MIXES),
        "survival_guard_a_absolute_floor": all(
            per_mix[m]["teacher_alive_fraction"] >= 0.30 for m in MIXES
        ),
        "survival_guard_b_scripted": per_mix["scripted"]["no_death"]["mean"] >= 0.03,
        "delivered_boost_share": 0.005 <= boost <= 0.10,
    }
    faults = [
        {"world_id": r["world"]["world_id"], "arm": name, "fault": fault}
        for r in records
        for name, arm in r["arms"].items()
        for fault in arm["unconditional_faults"]
    ]
    point_clauses = (
        "scripted_no_death_ge_066",
        "scripted_no_death_gain_ge_010",
        "mass_gain_ge_40",
        "survival_guard_a_absolute_floor",
        "survival_guard_b_scripted",
        "delivered_boost_share",
    )
    clear_survival_failure = alive["upper"] <= -0.05 or any(
        per_mix[m]["alive"]["upper"] <= -0.075 for m in MIXES
    )
    if faults:
        decision, reason = (
            "STOP",
            "unconditional fault; this teacher/controller failed its registered bar",
        )
    elif any(not clauses[c] for c in point_clauses) or clear_survival_failure:
        decision, reason = "STOP", "this teacher/controller failed its registered bar"
    elif all(clauses.values()):
        decision, reason = "GO", "all Round 0 development premise bars pass; no release authority"
    else:
        decision, reason = (
            "INCONCLUSIVE",
            "point bars pass but confidence bounds remain inconclusive",
        )
    return dict(
        schema="t34-round0-decision/v1",
        decision=decision,
        reason=reason,
        evidence_authority="UNVERIFIED_NUMERICAL_SUMMARY",
        independent_unit="paired_world",
        look="fixed-N-240-only",
        mass=mass,
        pooled_alive=alive,
        per_mix=per_mix,
        boost_share=boost,
        clauses=clauses,
        unconditional_faults=faults,
    )


def inconclusive(reason: str) -> dict[str, Any]:
    return dict(
        schema="t34-round0-decision/v1",
        decision="INCONCLUSIVE",
        reason=reason,
        evidence_authority="INVALID_OR_INCOMPLETE",
        release_authorized=False,
    )


def _bound_path(root: Path, entry: Mapping[str, Any]) -> Path:
    from research.t34_round0.common import sha256

    name = Path(entry["path"])
    if name.is_absolute() or ".." in name.parts:
        raise InvalidEvidence("artifact path must stay inside its immutable chunk")
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise InvalidEvidence("artifact missing or escapes its chunk")
    if sha256(path) != entry["sha256"]:
        raise InvalidEvidence(f"artifact SHA mismatch: {name}")
    return path


def _admission(chunk: Path, identity: Mapping[str, Any], repo: Path) -> dict[str, Any]:
    """Revalidate supervisor publication and terminal success before any GO."""
    from research.t34_round0.common import read_json, sha256

    chunk = chunk.resolve()
    if chunk.name != "round0" or chunk.parent.name != "exports":
        raise InvalidEvidence("production chunks must be supervisor job/exports/round0 directories")
    job = chunk.parent.parent
    receipt = read_json(job / "receipt.json")
    manifest_path = job / "manifest.json"
    manifest = read_json(manifest_path)
    terminal = read_json(job / "terminal.json")
    fault_terminal = receipt.get("terminal") is None and (chunk / "fault.json").is_file()
    if fault_terminal:
        # execute() writes the terminal before rejecting a nonzero workload;
        # its caller then has no returned terminal/exports. Bind the independently
        # preserved poll rather than inventing a successful transport receipt.
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
        ("BANK_ROUND0.json", "bank_sha256"),
        ("SEED_AUDIT_ROUND0.json", "audit_sha256"),
        ("RATIFICATION.json", "ratification_sha256"),
    ):
        if inputs.get(dest) != identity[key]:
            raise InvalidEvidence("manifest input differs from registered analysis input")
    inventory = terminal.get("exports") if fault_terminal else receipt.get("exports")
    terminal_exports = terminal.get("exports")
    if (
        not isinstance(inventory, list)
        or not isinstance(terminal_exports, list)
        or sorted(inventory, key=lambda item: item["path"])
        != sorted(terminal_exports, key=lambda item: item["path"])
    ):
        raise InvalidEvidence("supervisor export inventories differ")
    exported = {}
    for item in inventory:
        path = _bound_path(job / "exports", item)
        if path in exported or item.get("size") != path.stat().st_size:
            raise InvalidEvidence("duplicate export or immutable export size mismatch")
        sidecar = Path(str(path) + ".sha256")
        marker = Path(str(path) + ".complete")
        if (
            not marker.is_file()
            or not sidecar.is_file()
            or sidecar.read_text().split()[0] != item["sha256"]
        ):
            raise InvalidEvidence("supervisor export publication is incomplete")
        exported[path] = item["sha256"]
    result = dict(manifest=manifest, exported=exported, receipt=receipt, terminal=terminal)
    return result


def _published(path: Path, admission: Mapping[str, Any]) -> None:
    from research.t34_round0.common import sha256

    if admission["exported"].get(path.resolve()) != sha256(path):
        raise InvalidEvidence("artifact is absent from immutable supervisor exports")


def analyze_paths(
    chunks: list[Path],
    bank_path: Path,
    audit_path: Path,
    ratification_path: Path,
    replay_path: Path,
    *,
    repo: Path | None = None,
) -> dict[str, Any]:
    """Only production decision entrypoint: require complete, immutable provenance.

    Trusted paths name the independently frozen registration and bank. Chunk
    records cannot self-assert a fresh bank, successful replay or ratification.
    """
    from research.t34_round0.common import B8_SHA, ROOT, read_json, sha256
    from research.t34_round0.history import CapExpiredHistory, HistoryError
    from research.t34_round0.registration import require_registration

    try:
        ratification = read_json(ratification_path)
        bank = require_registration(ratification, bank_path, audit_path, repo=repo or ROOT)
        from research.t34_pathb.source_binding_v2 import verify_frozen_worktree
        from research.t34_round0.history import require_active_phase, require_chunks, selected_rows

        require_active_phase(ratification, "analysis")
        verify_frozen_worktree(repo or ROOT, ratification["commit"])
        try:
            require_chunks(chunks, ratification)
        except FileNotFoundError as exc:
            raise HistoryError("unfinished claim is missing its terminal history") from exc
        expected_identity = dict(
            bank_sha256=sha256(bank_path),
            audit_sha256=sha256(audit_path),
            ratification_sha256=sha256(ratification_path),
            checkpoint_sha256=B8_SHA,
            commit=ratification["commit"],
        )
        expected = {row["world_id"]: row for row in bank["rows"]}
        records, evidence, seen = [], {}, set()
        action_streams = {}
        if not chunks or len(set(p.resolve() for p in chunks)) != len(chunks):
            raise InvalidEvidence("missing or duplicate chunks")
        chunk_hashes = {}
        admissions, admission_errors = [], []
        for chunk_root in chunks:
            try:
                admission = _admission(chunk_root, expected_identity, repo or ROOT)
                fault_path = chunk_root / "fault.json"
                if fault_path.is_file():
                    _published(fault_path, admission)
                    fault = read_json(fault_path)
                    fault_identity = fault.get("run_identity", {})
                    if (
                        fault.get("schema") != "t34-round0-fault/v1"
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
                        or any(fault_identity.get(k) != v for k, v in expected_identity.items())
                    ):
                        raise InvalidEvidence(
                            "fault receipt lacks registered unconditional-fault identity"
                        )
                    return dict(
                        schema="t34-round0-decision/v1",
                        decision="STOP",
                        reason=(
                            "unconditional fault; this teacher/controller "
                            "failed its registered bar"
                        ),
                        evidence_authority="TIER_1_DEVELOPMENT",
                        release_authorized=False,
                        unconditional_faults=[fault],
                        fault_sha256=sha256(fault_path),
                    )
                admissions.append((chunk_root, admission))
            except (ValueError, KeyError, TypeError, OSError, IndexError) as exc:
                admission_errors.append(str(exc))
        if admission_errors:
            raise InvalidEvidence(admission_errors[0])
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
            if (
                receipt.get("schema") != "t34-round0-chunk/v1"
                or receipt.get("status") != "complete"
                or receipt.get("preflight") is not False
            ):
                raise InvalidEvidence("missing scientific chunk completion; preflight is excluded")
            identity = receipt.get("run_identity", receipt.get("identities", {}))
            if any(identity.get(key) != value for key, value in expected_identity.items()):
                raise InvalidEvidence("chunk identity differs from registered inputs")
            runtime_id = identity.get("runtime_id", receipt.get("runtime_id"))
            if not isinstance(runtime_id, str) or not runtime_id:
                raise InvalidEvidence("missing paired runtime identity")
            pair_entries = receipt.get("pairs")
            if not isinstance(pair_entries, list) or not pair_entries:
                raise InvalidEvidence("chunk has no immutable pair inventory")
            allocated = receipt.get("slice", {}).get("count", 0)
            if type(allocated) is not int or allocated < 1 or len(pair_entries) > allocated:
                raise InvalidEvidence("invalid allocated/completed chunk counts")
            if (allocated - len(pair_entries)) / allocated > 0.02:
                raise InvalidEvidence("STOP: >2% dropped worlds in a chunk; preserve and report")
            chunk_hashes[str(receipt_path)] = sha256(receipt_path)
            for entry in pair_entries:
                pair_path = _bound_path(chunk_root, entry)
                _published(pair_path, admission)
                pair = read_json(pair_path)
                if (
                    pair.get("schema") != "t34-round0-pair/v1"
                    or pair.get("status") != "complete"
                    or pair.get("preflight") is not False
                ):
                    raise InvalidEvidence("non-scientific or incomplete pair")
                pair_identity = pair["run_identity"]
                if (
                    any(pair_identity.get(k) != v for k, v in expected_identity.items())
                    or pair_identity.get("runtime_id") != runtime_id
                ):
                    raise InvalidEvidence("pair or arm runtime identity mismatch")
                world = pair["world"]
                world_id = world["world_id"]
                if world_id not in expected or world_id in seen:
                    raise InvalidEvidence("unexpected or duplicate pair")
                wanted = expected[world_id]
                if any(world.get(k) != v for k, v in wanted.items()):
                    raise InvalidEvidence("pair differs from registered world/roster")
                if world.get("horizon") != HORIZON:
                    raise InvalidEvidence("scientific pair requires H5000")
                trace_hashes = {}
                action_streams[world_id] = {}
                for name in ("bare", "teacher"):
                    arm = pair["arms"][name]
                    trace_path = _bound_path(chunk_root, arm["trace"])
                    _published(trace_path, admission)
                    trace = read_json(trace_path)
                    if (
                        trace.get("schema") != "t34-exit-world/v1"
                        or trace.get("seed") != world["world_seed"]
                        or trace.get("world_id") != world_id
                        or trace.get("mix") != world["mix"]
                        or trace.get("row") != wanted
                        or trace.get("frames") != HORIZON
                        or trace.get("horizon") != 300
                        or trace.get("rewinds") != 0
                        or trace.get("beta") != (0.0 if name == "bare" else 1.0)
                        or trace.get("rollout_policy_sha256") != B8_SHA
                        or trace.get("checkpoint_sha256") != B8_SHA
                        or trace.get("labels") != []
                        or trace.get("complete") is not True
                    ):
                        raise InvalidEvidence("trace world identity mismatch")
                    actions = trace.get("actions")
                    if not isinstance(actions, list) or len(actions) != arm["decisions"]:
                        raise InvalidEvidence("trace decision count differs from arm metrics")
                    counts = [0] * 6
                    for frame in actions:
                        if not isinstance(frame, list) or len(frame) != 6:
                            raise InvalidEvidence("trace requires six legal actions per frame")
                        for action in frame:
                            _integer(action, "trace action", 0, 5)
                        counts[frame[0]] += 1
                    if counts != arm["action_counts"]:
                        raise InvalidEvidence("trace hero actions differ from arm counts")
                    final_hash = trace.get("hashes", {}).get(str(len(actions)))
                    required_frames = {"0", str(len(actions))} | {
                        str(frame) for frame in range(250, len(actions) + 1, 250)
                    }
                    if set(trace.get("hashes", {})) != required_frames:
                        raise InvalidEvidence("trace lacks the registered state-hash inventory")
                    if (
                        not isinstance(final_hash, str)
                        or len(final_hash) != 40
                        or any(c not in "0123456789abcdef" for c in final_hash)
                        or final_hash != arm["trace"].get("final_state_hash")
                    ):
                        raise InvalidEvidence("trace terminal state hash differs")
                    action_streams[world_id][name] = [frame[0] for frame in actions]
                    trace_hashes[name] = sha256(trace_path)
                evidence[world_id] = dict(
                    world_id=world_id, pair_sha256=sha256(pair_path), traces=trace_hashes
                )
                seen.add(world_id)
                records.append(pair)
        if seen != set(expected) or len(records) != 240:
            raise InvalidEvidence("fixed look lacks exactly the whole registered 240-world bank")
        replay = read_json(replay_path)
        selected = [
            row["world_id"]
            for row in selected_rows(
                bank["rows"], [read_json(root / "chunk.json") for root in chunks]
            )
        ]
        if (
            replay.get("schema") != "t34-round0-replay/v1"
            or replay.get("status") != "PASS"
            or replay.get("divergences") != []
            or replay.get("selected_ids") != selected
            or replay.get("platform", {}).get("system") != "Darwin"
            or replay.get("platform", {}).get("machine") not in ("arm64", "aarch64")
        ):
            raise InvalidEvidence("missing full prospective Mac replay or unexplained divergence")
        replay_identity = replay.get("run_identity", replay.get("identities", {}))
        if any(replay_identity.get(k) != v for k, v in expected_identity.items()):
            raise InvalidEvidence("Mac replay input identity mismatch")
        checks = replay.get("checked_pairs")
        if not isinstance(checks, list) or len(checks) != len(selected):
            raise InvalidEvidence("Mac replay requires the ceil(10%) sample from every chunk")
        if checks != [evidence[world_id] for world_id in selected]:
            raise InvalidEvidence("Mac replay does not bind exact exported pairs and traces")
        # Canonical bank order is independent of chunking and completion order.
        # Check every rolling 100k window, including windows across chunk edges.
        window_faults = []
        for arm_name in ("bare", "teacher"):
            window, counts = deque(), [0] * 6
            for row in bank["rows"]:
                for action in action_streams[row["world_id"]][arm_name]:
                    if len(window) == 100000:
                        counts[window.popleft()] -= 1
                    window.append(action)
                    counts[action] += 1
                    if len(window) == 100000 and max(counts) > 90000:
                        window_faults.append(
                            dict(
                                world_id=row["world_id"], arm=arm_name, fault="action_collapse_100k"
                            )
                        )
                        break
                if window_faults and window_faults[-1]["arm"] == arm_name:
                    break
        result = summarize(records)
        if window_faults:
            result.update(
                decision="STOP",
                reason="unconditional action-collapse fault; "
                "this teacher/controller failed its registered bar",
            )
            result["unconditional_faults"].extend(window_faults)
        result.update(
            evidence_authority="TIER_1_DEVELOPMENT",
            release_authorized=False,
            bank_sha256=expected_identity["bank_sha256"],
            chunk_sha256=chunk_hashes,
            replay_sha256=sha256(replay_path),
            ratification_sha256=expected_identity["ratification_sha256"],
        )
        return result
    except CapExpiredHistory as exc:
        return inconclusive(str(exc))
    except HistoryError as exc:
        return dict(
            schema="t34-round0-decision/v1",
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
        return inconclusive(str(exc))


def main() -> None:
    from research.t34_round0.common import read_json
    from research.t34_round0.history import policy, run_phase

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", nargs="+", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
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
        lambda _out: analyze_paths(
            args.chunks, args.bank, args.audit, args.ratification, args.replay
        ),
    )


if __name__ == "__main__":
    main()
