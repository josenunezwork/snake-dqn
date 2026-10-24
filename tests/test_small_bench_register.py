from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tools import small_bench_register as register

NOW = datetime(2026, 10, 24, 12, tzinfo=timezone.utc)
SHA = "a" * 64


def init() -> dict:
    return {
        "type": "init",
        "id": "initial",
        "scopes": {
            "global": {
                "ceiling": {"units": 110, "wall_minutes": 720},
                "spent": {"units": 5, "wall_minutes": 10},
            },
            "v8": {
                "ceiling": {"units": 36, "wall_minutes": 216},
                "spent": {"units": 7, "wall_minutes": 44},
            },
        },
        "holds": [],
        "external_commitments_units": 3,
        "unsettled_debit_holds_units": 2,
        "completeness_attestation": "bounded historical carry-forward",
        "sources": [{"path": "source", "sha256": SHA}],
    }


def package(**changes: object) -> dict:
    value = {
        "job_id": "job",
        "manifest_sha256": SHA,
        "packet_id": "packet",
        "scopes": ["global", "v8"],
        "cap": {"units": 20, "wall_minutes": 120},
        "outside_tail": {"units": 0, "wall_minutes": 10},
        "deadline_utc": (NOW + timedelta(hours=3)).isoformat(),
        "capture_utc": NOW.isoformat(),
        "live_balance_units": 50,
        "bindings": {"acceptance": SHA},
        "operator": "operator",
    }
    value.update(changes)
    return value


def seeded(path: Path) -> list[dict]:
    register.append(path, init())
    return register.read(path)


def test_append_hash_chain_and_atomic_admission(tmp_path: Path) -> None:
    path = tmp_path / "register.jsonl"
    with register.reservation_lock(tmp_path / "account.lock"):
        event = register.reserve(seeded(path), package(), "r1", NOW)
        saved = register.append(path, event)
    rows = register.read(path)
    assert saved["revision"] == 1
    assert rows[-1]["prev_hash"] == rows[0]["row_hash"]
    assert register.state(rows)["outstanding"]["r1"]["pair"]["units"] == 20


def test_refresh_is_idempotent_but_changed_identity_refuses(tmp_path: Path) -> None:
    path = tmp_path / "r.jsonl"
    register.append(path, init())
    register.append(path, register.reserve(register.read(path), package(), "r1", NOW))
    refresh = register.reserve(register.read(path), package(), "r1", NOW)
    assert refresh["type"] == "refresh"
    with pytest.raises(register.Refused, match="identity"):
        register.reserve(register.read(path), package(job_id="other"), "r1", NOW)
    with pytest.raises(register.Refused, match="frozen"):
        register.reserve(
            register.read(path), package(cap={"units": 19, "wall_minutes": 120}), "r1", NOW
        )
    with pytest.raises(register.Refused, match="live account"):
        register.reserve(register.read(path), package(live_balance_units=0), "r1", NOW)


def test_scope_live_balance_and_serial_a100_refuse(tmp_path: Path) -> None:
    path = tmp_path / "r.jsonl"
    register.append(path, init())
    with pytest.raises(register.Refused, match="scope"):
        register.reserve(
            register.read(path), package(cap={"units": 40, "wall_minutes": 120}), "bad", NOW
        )
    with pytest.raises(register.Refused, match="live account"):
        register.reserve(register.read(path), package(live_balance_units=24), "bad", NOW)
    register.append(path, register.reserve(register.read(path), package(), "r1", NOW))
    with pytest.raises(register.Refused, match="another small A100"):
        register.reserve(register.read(path), package(job_id="two"), "r2", NOW)
    with pytest.raises(register.Refused, match="expired"):
        register.reserve(
            seeded(tmp_path / "deadline.jsonl"),
            package(deadline_utc=(NOW + timedelta(minutes=129)).isoformat()),
            "deadline",
            NOW,
        )
    with pytest.raises(register.Refused, match="stale"):
        register.reserve(
            seeded(tmp_path / "stale.jsonl"),
            package(capture_utc=(NOW - timedelta(minutes=11)).isoformat()),
            "stale",
            NOW,
        )


def test_audit_deadline_and_gross_limit_warn_not_refuse(tmp_path: Path) -> None:
    path = tmp_path / "r.jsonl"
    register.append(path, init())
    register.append(path, register.reserve(register.read(path), package(), "r1", NOW))
    register.append(
        path,
        {
            "type": "settlement",
            "id": "settle-r1",
            "reservation_id": "r1",
            "charged": {"units": 0, "wall_minutes": 0},
            "remaining_hold": {"units": 0, "wall_minutes": 0},
            "sources": [{"path": "receipt", "sha256": SHA}],
        },
    )
    later = NOW + timedelta(hours=25)
    with pytest.warns(register.AuditGateWarning, match="overdue"):
        late = register.reserve(
            register.read(path),
            package(
                job_id="later",
                capture_utc=later.isoformat(),
                deadline_utc=(later + timedelta(hours=3)).isoformat(),
            ),
            "r2",
            later,
        )
    gate = late["accounting_audit_gate"]
    assert gate["blocking"] is False and gate["status"] == "WARN"
    assert gate["warnings"] == ["accounting audit overdue"]
    assert gate["audit_due_utc"] == (NOW + timedelta(hours=24)).isoformat()
    rows = register.read(path)
    audit = {
        "type": "audit",
        "id": "audit-1",
        "status": "ACCEPT",
        "prefix_revision": 1,
        "prefix_hash": rows[1]["row_hash"],
        "independent_source": {"kind": "codex-v", "path": "audit", "sha256": SHA},
    }
    register.append(path, audit)
    assert register.state(register.read(path))["first_unaudited_admission"] is None


def test_settlement_preserves_remaining_hold_and_requires_evidence(tmp_path: Path) -> None:
    path = tmp_path / "r.jsonl"
    register.append(path, init())
    register.append(path, register.reserve(register.read(path), package(), "r1", NOW))
    with pytest.raises(register.Refused, match="sources"):
        register.append(
            path,
            {
                "type": "settlement",
                "id": "settle",
                "reservation_id": "r1",
                "charged": {"units": 10, "wall_minutes": 60},
                "remaining_hold": {"units": 2, "wall_minutes": 5},
            },
        )
    register.append(
        path,
        {
            "type": "settlement",
            "id": "settle",
            "reservation_id": "r1",
            "charged": {"units": 10, "wall_minutes": 60},
            "remaining_hold": {"units": 2, "wall_minutes": 5},
            "sources": [{"path": "receipt", "sha256": SHA}],
        },
    )
    state = register.state(register.read(path))
    assert state["outstanding"]["r1"]["pair"] == {"units": 2.0, "wall_minutes": 5.0}
    with pytest.raises(register.Refused, match="settled"):
        register.reserve(register.read(path), package(), "r1", NOW)


def test_lock_is_nonblocking_and_chain_tampering_refuses(tmp_path: Path) -> None:
    path = tmp_path / "r.jsonl"
    register.append(path, init())
    with register.reservation_lock(tmp_path / "shared.lock"):
        with pytest.raises(register.Refused, match="already held"):
            with register.reservation_lock(tmp_path / "shared.lock"):
                pass
    path.write_text(path.read_text().replace("initial", "tampered"))
    with pytest.raises(register.Refused, match="hash"):
        register.read(path)


def test_initial_holds_debit_parent_and_reductions_need_settlement_evidence(tmp_path: Path) -> None:
    path = tmp_path / "r.jsonl"
    initial = init()
    initial["holds"] = [
        {
            "id": "required",
            "scopes": ["global", "v8"],
            "pair": {"units": 10, "wall_minutes": 10},
            "account_units": 10,
            "kind": "required-work",
        }
    ]
    register.append(path, initial)
    with pytest.raises(register.Refused, match="scope"):
        register.reserve(register.read(path), package(), "r1", NOW)
    with pytest.raises(register.Refused, match="sources"):
        register.append(
            path,
            {
                "type": "delta",
                "id": "bad-reduce",
                "external_commitments_units_set": 2,
                "sources": [{"path": "delta", "sha256": SHA}],
            },
        )
    register.append(
        path,
        {
            "type": "delta",
            "id": "reduce",
            "holds_replace": [
                {
                    "id": "required",
                    "scopes": ["global", "v8"],
                    "pair": {"units": 0, "wall_minutes": 0},
                    "account_units": 0,
                    "kind": "required-work",
                }
            ],
            "settlement_evidence": [{"path": "release", "sha256": SHA}],
            "settlement_attestation": "operator verified settlement",
            "sources": [{"path": "delta", "sha256": SHA}],
        },
    )
    assert register.state(register.read(path))["outstanding"]["hold:required"]["account_units"] == 0


def test_initial_duplicate_holds_and_unpaid_debit_stay_conservative(tmp_path: Path) -> None:
    initial = init()
    hold = {
        "id": "same",
        "scopes": ["global"],
        "pair": {"units": 1, "wall_minutes": 1},
        "account_units": 1,
        "kind": "x",
    }
    initial["holds"] = [hold, hold]
    with pytest.raises(register.Refused, match="duplicate"):
        register.append(tmp_path / "dupe.jsonl", initial)
    path = tmp_path / "debit.jsonl"
    register.append(path, init())
    register.append(path, register.reserve(register.read(path), package(), "r1", NOW))
    register.append(
        path,
        {
            "type": "settlement",
            "id": "unpaid",
            "reservation_id": "r1",
            "charged": {"units": 20, "wall_minutes": 120},
            "remaining_hold": {"units": 0, "wall_minutes": 0},
            "remaining_account_units": 3,
            "sources": [{"path": "receipt", "sha256": SHA}],
        },
    )
    assert register.state(register.read(path))["outstanding"]["r1"]["account_units"] == 3


def test_audit_prefix_resets_batch_gross_and_sixth_reservation_only_warns(tmp_path: Path) -> None:
    path = tmp_path / "gross.jsonl"
    initial = init()
    initial["scopes"]["global"]["ceiling"] = {"units": 1000, "wall_minutes": 10000}
    register.append(path, initial)
    base = package(scopes=["global"])

    def settle(reservation_id: str, number: int) -> None:
        register.append(
            path,
            {
                "type": "settlement",
                "id": f"settle-{number}",
                "reservation_id": reservation_id,
                "charged": {"units": 0, "wall_minutes": 0},
                "remaining_hold": {"units": 0, "wall_minutes": 0},
                "sources": [{"path": f"receipt-{number}", "sha256": SHA}],
            },
        )

    register.append(path, register.reserve(register.read(path), base, "old", NOW))
    settle("old", 0)
    rows = register.read(path)
    register.append(
        path,
        {
            "type": "audit",
            "id": "audit-old",
            "status": "ACCEPT",
            "prefix_revision": rows[-1]["revision"],
            "prefix_hash": rows[-1]["row_hash"],
            "independent_source": {"kind": "codex-v", "path": "audit", "sha256": SHA},
        },
    )
    for number in range(5):
        candidate = dict(base, job_id=f"new-{number}")
        reservation_id = f"new-{number}"
        register.append(path, register.reserve(register.read(path), candidate, reservation_id, NOW))
        settle(reservation_id, number + 1)
    with pytest.warns(register.AuditGateWarning, match="gross"):
        over = register.reserve(register.read(path), dict(base, job_id="over"), "over", NOW)
    assert over["accounting_audit_gate"]["status"] == "WARN"
    assert over["accounting_audit_gate"]["unaudited_gross_units_after"] == 120
    register.append(path, over)
    assert register.read(path)[-1]["accounting_audit_gate"]["blocking"] is False


def test_release_keeps_default_debt_then_evidenced_payment_reconciles(tmp_path: Path) -> None:
    path = tmp_path / "debt.jsonl"
    register.append(path, init())
    register.append(path, register.reserve(register.read(path), package(), "r1", NOW))
    register.append(
        path,
        {
            "type": "settlement",
            "id": "released",
            "reservation_id": "r1",
            "charged": {"units": 10, "wall_minutes": 60},
            "remaining_hold": {"units": 0, "wall_minutes": 0},
            "sources": [{"path": "release", "sha256": SHA}],
        },
    )
    current = register.state(register.read(path))
    assert current["outstanding"]["r1"]["account_units"] == 10
    assert current["outstanding"]["r1"]["kind"] == "unsettled_debit"
    event = {
        "type": "reconcile",
        "id": "paid",
        "reservation_id": "r1",
        "remaining_account_units": 0,
        "sources": [{"path": "provider", "sha256": SHA}],
    }
    with pytest.raises(register.Refused, match="sources"):
        register.append(path, event)
    event.update(
        settlement_evidence=[{"path": "paid-provider", "sha256": SHA}],
        settlement_attestation="Provider debit reconciled once",
    )
    register.append(path, event)
    final = register.state(register.read(path))
    assert "r1" not in final["outstanding"]
    assert final["scopes"]["global"]["spent"]["units"] == 15


def test_payment_cannot_remove_coverage_for_remaining_runtime(tmp_path: Path) -> None:
    path = tmp_path / "exposure.jsonl"
    register.append(path, init())
    register.append(path, register.reserve(register.read(path), package(), "r1", NOW))
    close = {
        "type": "settlement",
        "id": "close",
        "reservation_id": "r1",
        "charged": {"units": 10, "wall_minutes": 60},
        "remaining_hold": {"units": 2, "wall_minutes": 5},
        "remaining_account_units": 0,
        "sources": [{"path": "paid", "sha256": SHA}],
    }
    with pytest.raises(register.Refused, match="remaining exposure"):
        register.append(path, close)
    close["remaining_account_units"] = 12
    register.append(path, close)
    with pytest.raises(register.Refused, match="remaining exposure"):
        register.append(
            path,
            {
                "type": "reconcile",
                "id": "clear",
                "reservation_id": "r1",
                "remaining_account_units": 0,
                "sources": [{"path": "paid", "sha256": SHA}],
                "settlement_evidence": [{"path": "paid", "sha256": SHA}],
                "settlement_attestation": "Past debit paid, runtime remains",
            },
        )


def raise_row(**changes: object) -> dict:
    value = {
        "type": "ceiling_raise",
        "id": "uncap",
        "ceilings": {
            "global": {"units": 100000, "wall_minutes": 600000},
            "v8": {"units": 100000, "wall_minutes": 600000},
        },
        "sources": [{"path": "owner", "sha256": SHA}],
        "attestation": "owner standing rule: no unit budgets",
        "owner_words": "remove budget",
    }
    value.update(changes)
    return value


def test_ceiling_raise_lifts_only_ceilings(tmp_path: Path) -> None:
    path = tmp_path / "raise.jsonl"
    register.append(path, init())
    register.append(path, register.reserve(register.read(path), package(), "r1", NOW))
    before = register.state(register.read(path))
    register.append(path, raise_row())
    after = register.state(register.read(path))
    assert after["scopes"]["global"]["ceiling"] == {"units": 100000.0, "wall_minutes": 600000.0}
    for name in ("global", "v8"):
        assert after["scopes"][name]["spent"] == before["scopes"][name]["spent"]
    assert after["outstanding"] == before["outstanding"]
    assert after["holds"] == before["holds"]
    for key in ("external_commitments_units", "unsettled_debit_holds_units"):
        assert after[key] == before[key]
    # The live-balance check still binds after the raise.
    fresh = tmp_path / "balance.jsonl"
    register.append(fresh, init())
    register.append(fresh, raise_row())
    with pytest.raises(register.Refused, match="live account balance"):
        register.reserve(register.read(fresh), package(live_balance_units=24), "r2", NOW)
    assert register.reserve(register.read(fresh), package(live_balance_units=25), "r2", NOW)


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"ceilings": {"global": {"units": 109, "wall_minutes": 720}}}, "cannot lower"),
        ({"ceilings": {"global": {"units": 200, "wall_minutes": 719}}}, "cannot lower"),
        ({"ceilings": {"global": {"units": 110, "wall_minutes": 720}}}, "at least one"),
        ({"ceilings": {}}, "requires ceilings"),
        ({"ceilings": {"new": {"units": 1, "wall_minutes": 1}}}, "unknown scope"),
        ({"ceilings": {"global": {"units": 200}}}, "pair"),
        ({"sources": []}, "sources"),
        ({"sources": None}, "sources"),
        ({"attestation": " "}, "attestation"),
        ({"owner_words": ""}, "owner_words"),
        ({"holds_add": []}, "refused fields"),
        ({"spent_additions": {"global": {"units": 0, "wall_minutes": 0}}}, "refused fields"),
        ({"external_commitments_units_set": 0}, "refused fields"),
        ({"scopes_add": {"x": {}}}, "refused fields"),
        ({"timeout_minutes": 999}, "refused fields"),
    ],
)
def test_ceiling_raise_refuses_decrease_other_fields_missing_sources(
    tmp_path: Path, changes: dict, match: str
) -> None:
    path = tmp_path / "bad.jsonl"
    register.append(path, init())
    with pytest.raises(register.Refused, match=match):
        register.append(path, raise_row(**changes))
    assert len(register.read(path)) == 1


def test_ceiling_raise_append_binds_sources_and_head(tmp_path: Path) -> None:
    path = tmp_path / "cli.jsonl"
    register.append(path, init())
    head = register.read(path)[-1]["row_hash"]
    evidence = tmp_path / "owner.txt"
    evidence.write_text("remove budget")
    digest = hashlib.sha256(evidence.read_bytes()).hexdigest()
    good = raise_row(sources=[{"path": str(evidence), "sha256": digest}])
    lock = tmp_path / "lock"
    with pytest.raises(register.Refused, match="mismatch"):
        register.append_ceiling_raise(
            path, lock, raise_row(sources=[{"path": str(evidence), "sha256": SHA}])
        )
    with pytest.raises(register.Refused, match="head moved"):
        register.append_ceiling_raise(path, lock, good, expect_head="b" * 64)
    assert len(register.read(path)) == 1
    row = register.append_ceiling_raise(path, lock, good, expect_head=head)
    assert row["revision"] == 1 and register.read(path)[-1]["type"] == "ceiling_raise"


LIVE = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/"
    "small-bench-sprint-20261024/register.jsonl"
)


@pytest.mark.skipif(not LIVE.exists(), reason="live sprint register not on this host")
def test_live_register_chain_still_verifies() -> None:
    rows = register.read(LIVE)
    assert rows and rows[0]["type"] == "init"
    assert register.state(rows)["initialized"] is True
