"""Round0 threshold and malformed-data checks; all records are synthetic."""

from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest

from research.t34_round0.analyze import COUNTS, InvalidEvidence, _stratified, summarize


def records() -> list[dict]:
    rows = []
    for mix, n in COUNTS.items():
        for i in range(n):
            arms = {}
            for name in ("bare", "teacher"):
                alive = i < int(n * (0.50 if name == "bare" else 0.75))
                decisions = 5000 if alive else 2500
                boost = decisions // 50
                counts = [decisions // 3, decisions // 3, 0, boost, 0, 0]
                counts[2] = decisions - sum(counts)
                arms[name] = dict(
                    mass_integral=100 + (50 if name == "teacher" else 0),
                    survival_fraction=1.0 if alive else 0.5,
                    no_death=int(alive),
                    deaths=int(not alive),
                    decisions=decisions,
                    boost_decisions=boost,
                    action_counts=counts,
                    unconditional_faults=[],
                )
            rows.append(
                dict(
                    world=dict(
                        world_id=f"world-{len(rows)}",
                        world_seed=len(rows) + 100,
                        mix=mix,
                        horizon=5000,
                    ),
                    arms=arms,
                )
            )
    return rows


def test_all_bars_pass_but_pure_summary_never_has_artifact_authority() -> None:
    result = summarize(records())
    assert result["decision"] == "GO"
    assert all(result["clauses"].values())
    assert result["evidence_authority"] == "UNVERIFIED_NUMERICAL_SUMMARY"
    assert result["mass"]["mean"] == 50


def test_equal_mix_weight_despite_unequal_counts() -> None:
    groups = {mix: [delta] * n for (mix, n), delta in zip(COUNTS.items(), (0.0, 0.0, 90.0))}
    summary = _stratified(groups, 0.1)
    assert summary["mean"] == 30
    assert summary["lower"] == 30


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "bad_mix", "nan", "counter", "bool"])
def test_malformed_records_refused(mutation: str) -> None:
    rows = records()
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[-1] = copy.deepcopy(rows[0])
    elif mutation == "bad_mix":
        rows[0]["world"]["mix"] = "unknown"
    elif mutation == "nan":
        rows[0]["arms"]["teacher"]["mass_integral"] = math.nan
    elif mutation == "counter":
        rows[0]["arms"]["teacher"]["action_counts"][0] += 1
    else:
        rows[0]["arms"]["teacher"]["decisions"] = True
    with pytest.raises(InvalidEvidence):
        summarize(rows)


def test_registered_fault_is_stop_even_with_positive_efficacy() -> None:
    rows = records()
    rows[0]["arms"]["teacher"]["unconditional_faults"] = ["illegal action"]
    assert summarize(rows)["decision"] == "STOP"


def test_gain_boundary_inclusive_and_below_boundary_stops() -> None:
    rows = records()
    for row in rows:
        row["arms"]["teacher"]["mass_integral"] = 140
    assert summarize(rows)["decision"] == "GO"
    rows[0]["arms"]["teacher"]["mass_integral"] -= 0.01
    assert summarize(rows)["decision"] == "STOP"


def test_uncertainty_is_not_a_point_failure_or_go() -> None:
    rows = records()
    for mix in COUNTS:
        group = [r for r in rows if r["world"]["mix"] == mix]
        for i, row in enumerate(group):
            row["arms"]["bare"]["mass_integral"] = 10000
            row["arms"]["teacher"]["mass_integral"] = 10050 + (9000 if i % 2 else -9000)
    result = summarize(rows)
    assert result["clauses"]["mass_gain_ge_40"]
    assert not result["clauses"]["mass_lb90_positive"]
    assert result["decision"] == "INCONCLUSIVE"


def test_delivered_boost_counts_are_pooled_decisions() -> None:
    rows = records()
    for row in rows:
        arm = row["arms"]["teacher"]
        arm["action_counts"][0] += arm["boost_decisions"]
        arm["action_counts"][3:] = [0, 0, 0]
        arm["boost_decisions"] = 0
    result = summarize(rows)
    assert result["boost_share"] == 0
    assert result["decision"] == "STOP"


@pytest.fixture
def artifacts(tmp_path, monkeypatch):
    """Synthetic provenance fixture; real registration validation is tested separately."""
    import json

    from research.t34_round0 import registration
    from research.t34_round0.common import B8_SHA, sha256, write_once

    bank_path, audit_path, ratification_path = (
        tmp_path / f"{name}.json" for name in ("bank", "audit", "ratification")
    )
    rows = records()
    bank = {"rows": [row["world"] for row in rows]}
    write_once(bank_path, bank)
    write_once(audit_path, {"test_fixture": True})
    write_once(ratification_path, {"commit": "a" * 40})
    monkeypatch.setattr(registration, "require_registration", lambda *a, **kw: bank)
    identity = dict(
        bank_sha256=sha256(bank_path),
        audit_sha256=sha256(audit_path),
        ratification_sha256=sha256(ratification_path),
        checkpoint_sha256=B8_SHA,
        commit="a" * 40,
        runtime_id="synthetic-runtime",
    )
    chunk = tmp_path / "job" / "exports" / "round0"
    pair_index, evidence = [], {}
    for row in rows:
        world_id = row["world"]["world_id"]
        for arm in ("bare", "teacher"):
            path = chunk / f"traces/{world_id}/{arm}.json"
            actions = [
                [action, 0, 0, 0, 0, 0]
                for action, count in enumerate(row["arms"][arm]["action_counts"])
                for _ in range(count)
            ]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    dict(
                        schema="t34-exit-world/v1",
                        seed=row["world"]["world_seed"],
                        world_id=world_id,
                        mix=row["world"]["mix"],
                        row=row["world"],
                        beta=0.0 if arm == "bare" else 1.0,
                        rollout_policy_sha256=B8_SHA,
                        checkpoint_sha256=B8_SHA,
                        labels=[],
                        complete=True,
                        frames=5000,
                        horizon=300,
                        rewinds=0,
                        actions=actions,
                        hashes={
                            str(frame): "f" * 40
                            for frame in {0, len(actions), *range(250, len(actions) + 1, 250)}
                        },
                    ),
                    separators=(",", ":"),
                )
            )
            row["arms"][arm]["trace"] = {
                "path": str(path.relative_to(chunk)),
                "sha256": sha256(path),
                "final_state_hash": "f" * 40,
            }
        row.update(
            schema="t34-round0-pair/v1", status="complete", preflight=False, run_identity=identity
        )
        path = chunk / f"pairs/{world_id}.json"
        write_once(path, row)
        pair_index.append(dict(path=str(path.relative_to(chunk)), sha256=sha256(path)))
        evidence[world_id] = dict(
            world_id=world_id,
            pair_sha256=sha256(path),
            traces={arm: row["arms"][arm]["trace"]["sha256"] for arm in ("bare", "teacher")},
        )
    write_once(
        chunk / "chunk.json",
        dict(
            schema="t34-round0-chunk/v1",
            status="complete",
            preflight=False,
            run_identity=identity,
            pairs=pair_index,
        ),
    )
    selected = [
        row["world_id"]
        for mix, n in zip(COUNTS, (10, 7, 7))
        for row in [r["world"] for r in rows if r["world"]["mix"] == mix][:n]
    ]
    from research.t34_round0.common import ROOT

    inventory = []
    for path in sorted(chunk.rglob("*.json")):
        digest = sha256(path)
        Path(str(path) + ".sha256").write_text(digest)
        Path(str(path) + ".complete").touch()
        inventory.append(
            dict(path=str(path.relative_to(chunk.parent)), sha256=digest, size=path.stat().st_size)
        )
    job = chunk.parent.parent
    manifest = dict(
        job_id="test-job",
        commit=identity["commit"],
        inputs=[
            dict(dest=dest, sha256=identity[key])
            for dest, key in (
                ("BANK_ROUND0.json", "bank_sha256"),
                ("SEED_AUDIT_ROUND0.json", "audit_sha256"),
                ("RATIFICATION.json", "ratification_sha256"),
            )
        ],
    )
    write_once(job / "manifest.json", manifest)
    terminal = dict(
        status="succeeded",
        returncode=0,
        reason=None,
        job_id="test-job",
        generation="generation",
        exports=inventory,
    )
    write_once(job / "terminal.json", terminal)
    write_once(
        job / "receipt.json",
        dict(
            status="PASS",
            stop_verified=True,
            terminal=terminal,
            exports=inventory,
            job_id="test-job",
            generation="generation",
            error=None,
            cleanup_error=None,
            commit=identity["commit"],
            manifest_sha256=sha256(job / "manifest.json"),
            supervisor_sha256=sha256(ROOT / "research/colab_cli/supervisor.py"),
        ),
    )
    replay_path = tmp_path / "replay.json"
    replay = dict(
        schema="t34-round0-replay/v1",
        status="PASS",
        divergences=[],
        run_identity=identity,
        selected_ids=selected,
        checked_pairs=[evidence[world_id] for world_id in selected],
        platform=dict(system="Darwin", machine="arm64"),
    )
    write_once(replay_path, replay)
    return dict(
        chunks=[chunk],
        bank_path=bank_path,
        audit_path=audit_path,
        ratification_path=ratification_path,
        replay_path=replay_path,
    )


def test_complete_artifact_chain_is_required(artifacts) -> None:
    from research.t34_round0.analyze import analyze_paths

    assert analyze_paths(**artifacts)["decision"] == "GO"
    artifacts["replay_path"].unlink()
    assert analyze_paths(**artifacts)["decision"] == "INCONCLUSIVE"


@pytest.mark.parametrize(
    "corruption",
    ["trace", "pair", "duplicate_chunk", "preflight", "missing_pair", "replay_divergence"],
)
def test_provenance_corruption_fails_closed(artifacts, corruption) -> None:
    import json

    from research.t34_round0.analyze import analyze_paths

    chunk = artifacts["chunks"][0]
    if corruption == "trace":
        next((chunk / "traces").rglob("*.json")).write_text("{}")
    elif corruption == "pair":
        next((chunk / "pairs").glob("*.json")).write_text("{}")
    elif corruption == "duplicate_chunk":
        artifacts["chunks"].append(chunk)
    elif corruption == "preflight":
        receipt = json.loads((chunk / "chunk.json").read_text())
        receipt["preflight"] = True
        (chunk / "chunk.json").write_text(json.dumps(receipt))
    elif corruption == "missing_pair":
        receipt = json.loads((chunk / "chunk.json").read_text())
        receipt["pairs"].pop()
        (chunk / "chunk.json").write_text(json.dumps(receipt))
    else:
        report = json.loads(artifacts["replay_path"].read_text())
        report["divergences"] = [{"reason": "unexplained"}]
        artifacts["replay_path"].write_text(json.dumps(report))
    assert analyze_paths(**artifacts)["decision"] == "INCONCLUSIVE"


def test_supervisor_failure_prevents_go(artifacts) -> None:
    import json

    from research.t34_round0.analyze import analyze_paths

    job = artifacts["chunks"][0].parent.parent
    receipt = json.loads((job / "receipt.json").read_text())
    receipt["stop_verified"] = False
    (job / "receipt.json").write_text(json.dumps(receipt))
    assert analyze_paths(**artifacts)["decision"] == "INCONCLUSIVE"


def test_admitted_unconditional_fault_stops_before_missing_chunk_or_replay(artifacts) -> None:
    import json

    from research.t34_round0.analyze import analyze_paths
    from research.t34_round0.common import sha256

    chunk = artifacts["chunks"][0]
    pair = json.loads(next((chunk / "pairs").glob("*.json")).read_text())
    fault_path = chunk / "fault.json"
    fault_path.write_text(
        json.dumps(
            dict(
                schema="t34-round0-fault/v1",
                preflight=False,
                unconditional=True,
                run_identity=pair["run_identity"],
                arm="teacher",
                world_id=pair["world"]["world_id"],
                fault="nonfinite_q",
            )
        )
    )
    digest = sha256(fault_path)
    Path(str(fault_path) + ".sha256").write_text(digest)
    Path(str(fault_path) + ".complete").touch()
    job = chunk.parent.parent
    terminal = json.loads((job / "terminal.json").read_text())
    terminal["exports"].append(
        dict(path="round0/fault.json", sha256=digest, size=fault_path.stat().st_size)
    )
    terminal.update(status="failed", returncode=2)
    (job / "terminal.json").write_text(json.dumps(terminal))
    receipt = json.loads((job / "receipt.json").read_text())
    receipt.update(terminal=terminal, exports=terminal["exports"], status="INCOMPLETE")
    (job / "receipt.json").write_text(json.dumps(receipt))
    # The incomplete output never had a scientific chunk receipt.
    (chunk / "chunk.json").unlink()
    terminal["exports"][:] = [x for x in terminal["exports"] if x["path"] != "round0/chunk.json"]
    (job / "terminal.json").write_text(json.dumps(terminal))
    receipt.update(terminal=terminal, exports=terminal["exports"])
    (job / "receipt.json").write_text(json.dumps(receipt))
    artifacts["replay_path"].unlink()
    assert analyze_paths(**artifacts)["decision"] == "STOP"
