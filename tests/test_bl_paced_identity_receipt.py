"""Fail-closed receipt admission tests using audited-reference-shaped metadata only."""

from __future__ import annotations

from dataclasses import asdict

from research.bl_paced_identity.events import sha256
from research.bl_paced_identity.fixtures import fixture_identity
from research.bl_paced_identity.receipt import (
    PACED_SCHEMA,
    REPLAY_SCHEMA,
    SEMANTIC_SCHEMA,
    check_paced,
    make_receipt,
)

SHA = "a" * 64


def _seal() -> dict[str, object]:
    return {
        "schema": "bl-paced-identity-journal/v1",
        "status": "SEALED",
        "event_count": 1,
        "chunk_count": 1,
        "chunk_sha256": [SHA],
        "header_sha256": SHA,
        "terminal_chunk_sha256": SHA,
    }


def _valid_runs(candidate: str) -> tuple[list[dict[str, object]], list[str]]:
    rows: list[dict[str, object]] = []
    common = {
        "synthetic": False,
        "platform": "Darwin",
        "candidate_sha256": candidate,
        "replay_verdict": "PASS",
        "recording_enabled": True,
        "trace_sha256": SHA,
        "replay_sha256": SHA,
        "trace_identity_sha256": candidate,
        "replay_identity_sha256": candidate,
        "load_envelope_sha256": SHA,
        "terminal_seal": _seal(),
        "natural_interventions": 0,
        "world_sha256": SHA,
        "roster_sha256": SHA,
        "profile_sha256": SHA,
        "fps": 24,
        "horizon": 5000,
        "snakes": 6,
    }
    for mix in ("scripted", "frozen", "mixed"):
        for number in range(16):
            for arm in ("candidate", "bare"):
                rows.append(
                    {
                        **common,
                        "run_id": f"world-{mix}-{number:02d}-{arm}",
                        "run_kind": "world",
                        "mix": mix,
                        "world_pair_id": f"{mix}-{number:02d}",
                        "world_sha256": sha256({"mix": mix, "world": number}),
                        "arm_role": arm,
                        "natural_interventions": 1 if arm == "candidate" and number == 0 else 0,
                    }
                )
    for number in range(6):
        for arm in ("candidate", "bare"):
            rows.append(
                {
                    **common,
                    "run_id": f"session-{number:02d}-{arm}",
                    "run_kind": "session",
                    "session_pair_id": f"session-{number:02d}",
                    "arm_role": arm,
                    "duration_seconds": 600,
                }
            )
    ids = [str(row["run_id"]) for row in rows]
    order = sha256(ids)
    for row in rows:
        row["chronological_order_sha256"] = order
    return rows, ids


def test_full_audited_metadata_corpus_can_pass_receipt_gate() -> None:
    identity = asdict(fixture_identity())
    candidate = sha256(identity)
    runs, ids = _valid_runs(candidate)
    paced = check_paced(
        runs, expected_run_ids=ids, candidate_sha256=candidate, frozen_order_sha256=sha256(ids)
    )
    assert paced["verdict"] == "PASS"
    source, inputs, reference = {"source": SHA}, {"bank": SHA}, {"reference": SHA}
    binding = sha256(
        {
            "candidate_sha256": candidate,
            "source_sha256": sha256(source),
            "input_sha256": sha256(inputs),
            "reference_sha256": sha256(reference),
        }
    )
    semantic = {
        "schema": SEMANTIC_SCHEMA,
        "verdict": "PASS",
        "qualification": "SEMANTIC_QUALIFIED",
        "bank114_sha256": SHA,
        "candidate_sha256": candidate,
        "binding_sha256": binding,
    }
    replay = {
        "schema": REPLAY_SCHEMA,
        "qualification": "REPLAY_QUALIFIED",
        "verdict": "PASS",
        "candidate_sha256": candidate,
        "binding_sha256": binding,
    }
    paced["binding_sha256"] = binding
    receipt = make_receipt(
        identity=identity,
        semantic=semantic,
        replay=replay,
        paced=paced,
        source_hashes=source,
        input_hashes=inputs,
        reference_hashes=reference,
        commands=["registered runner --frozen-manifest manifest.json"],
        journal_seals=[_seal()],
        forecast={"scope": "audited reference metadata"},
    )
    assert receipt["verdict"] == "PASS"
    assert receipt["efficacy_unblinding"] == "IDENTITY_CHECKED"
    l3_admission = make_receipt(
        identity=identity,
        semantic=semantic,
        replay=replay,
        paced={
            "schema": PACED_SCHEMA,
            "verdict": "NOT_RUN",
            "candidate_sha256": candidate,
            "binding_sha256": binding,
        },
        source_hashes=source,
        input_hashes=inputs,
        reference_hashes=reference,
        commands=["registered runner --frozen-manifest manifest.json"],
        journal_seals=[_seal()],
        forecast={"scope": "before first paced run"},
    )
    assert l3_admission["verdict"] == "INCOMPLETE"
    assert l3_admission["l3_identity_prerequisites"] == "PASS"


def test_missing_and_malformed_evidence_is_incomplete_but_divergence_is_invalid() -> None:
    identity = asdict(fixture_identity())
    receipt = make_receipt(
        identity=identity,
        semantic={"verdict": "PASS", "qualification": "SEMANTIC_QUALIFIED"},
        replay={"schema": REPLAY_SCHEMA, "verdict": "PASS"},
        paced={"schema": PACED_SCHEMA, "verdict": "PASS"},
        source_hashes={"source": "bad"},
        input_hashes={},
        reference_hashes={"reference": SHA},
        commands=[],
        journal_seals=[{"status": "SEALED"}],
        forecast={},
    )
    assert receipt["verdict"] == "INCOMPLETE"
    assert {"source_hashes_invalid", "input_hashes_missing", "commands_invalid"} <= set(
        receipt["evidence_problems"]
    )
    invalid = make_receipt(
        identity=identity,
        semantic={"verdict": "PASS", "first_divergence": {"frame": 7}},
        replay={},
        paced={},
        source_hashes={},
        input_hashes={},
        reference_hashes={},
        commands=[],
        journal_seals=[],
        forecast={},
    )
    assert invalid["verdict"] == "INVALID_IDENTITY"
    assert invalid["efficacy_unblinding"] == "BLOCKED"


def test_extra_attempt_and_mismatched_pair_profile_block_paced_qualification() -> None:
    candidate = sha256(asdict(fixture_identity()))
    runs, ids = _valid_runs(candidate)
    runs.append({**runs[0], "run_id": "extra-attempt"})
    ids.append("extra-attempt")
    for row in runs:
        row["chronological_order_sha256"] = sha256(ids)
    report = check_paced(
        runs, expected_run_ids=ids, candidate_sha256=candidate, frozen_order_sha256=sha256(ids)
    )
    assert report["verdict"] == "INCOMPLETE"
    assert "duplicate_pair_arm_attempt" in report["reason"]
    runs, ids = _valid_runs(candidate)
    runs[1]["roster_sha256"] = "b" * 64
    report = check_paced(
        runs, expected_run_ids=ids, candidate_sha256=candidate, frozen_order_sha256=sha256(ids)
    )
    assert report["verdict"] == "INCOMPLETE"
    assert "paired_world_roster_profile_mismatch" in report["reason"]
