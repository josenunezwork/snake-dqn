#!/usr/bin/env python3
"""Small durable operator-task journal for the existing ops watcher.

This module deliberately has no dispatcher or transport dependency.  It records
static proof separately from live eligibility and emits at-least-once notices.
The operator still performs admission and submission under ``dispatch.sh``.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import sqlite3
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

SCHEMA = "r33-task-journal/v1"
STATIC_READY = "READY"
ELIGIBLE_READY = "ELIGIBLE_READY"
HOLD = "HOLD"
TERMINAL = "TERMINAL"
ACK_REMINDER_SECONDS = 5 * 60
ESCALATION_SECONDS = 15 * 60
RECONCILE_SECONDS = 60


class Refused(ValueError):
    """The journal cannot prove that a state transition is safe."""


@dataclass(frozen=True)
class Identity:
    """Static identity required before a task can ever be READY."""

    campaign: str
    stage: str
    attempt: str
    source_closure_sha256: str
    manifest_sha256: str
    prerequisite_receipts_sha256: str

    def __post_init__(self) -> None:
        values = asdict(self)
        if not all(isinstance(value, str) and value for value in values.values()):
            raise Refused("complete task identity is required")
        for name in ("source_closure_sha256", "manifest_sha256", "prerequisite_receipts_sha256"):
            value = values[name]
            if len(value) != 64 or set(value) - set("0123456789abcdef"):
                raise Refused("identity requires lowercase SHA-256 hashes")

    def key(self) -> str:
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    @classmethod
    def from_value(cls, value: dict[str, Any]) -> "Identity":
        fields = {name: value.get(name) for name in cls.__dataclass_fields__}
        hashes = ("source_closure_sha256", "manifest_sha256", "prerequisite_receipts_sha256")
        if not all(isinstance(item, str) and item for item in fields.values()):
            raise Refused("identity requires campaign/stage/attempt and all bound hashes")
        if any(
            len(fields[name]) != 64 or set(fields[name]) - set("0123456789abcdef")
            for name in hashes
        ):
            raise Refused("identity hash fields must be lowercase SHA-256")
        return cls(**fields)


class Journal:
    """SQLite-backed, transactional task and consumer-notification state."""

    def __init__(self, ops: Path):
        self.ops = ops
        ops.mkdir(parents=True, exist_ok=True)
        self.path = ops / "task-journal.sqlite3"
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self._schema()

    def _schema(self) -> None:
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS tasks (
              identity TEXT PRIMARY KEY, campaign TEXT NOT NULL, stage TEXT NOT NULL,
              attempt TEXT NOT NULL, source_closure_sha256 TEXT NOT NULL,
              manifest_sha256 TEXT NOT NULL, prerequisite_receipts_sha256 TEXT NOT NULL,
              lane TEXT NOT NULL, owner TEXT NOT NULL, next_action TEXT NOT NULL,
              static_verified INTEGER NOT NULL DEFAULT 0,
              authority_verified INTEGER NOT NULL DEFAULT 0,
              proof_evidence_sha256 TEXT, authority_receipt_sha256 TEXT,
              current_eligible INTEGER NOT NULL DEFAULT 0,
              source_revoked INTEGER NOT NULL DEFAULT 0,
              terminal_evidence INTEGER NOT NULL DEFAULT 0,
              allocation_released INTEGER NOT NULL DEFAULT 0,
              retained_allocation INTEGER NOT NULL DEFAULT 0,
              submission_state TEXT NOT NULL DEFAULT 'NONE',
              state TEXT NOT NULL DEFAULT 'PREPARED', blocker_code TEXT, blocker_owner TEXT,
              ready_at INTEGER, eligible_ready_at INTEGER, started_at INTEGER, terminal_at INTEGER,
              generation INTEGER NOT NULL DEFAULT 0,
              created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL
            );
            CREATE UNIQUE INDEX IF NOT EXISTS one_successor_per_lane
              ON tasks(lane) WHERE state NOT IN ('TERMINAL') AND current_eligible=1;
            CREATE TABLE IF NOT EXISTS notices (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              identity TEXT NOT NULL REFERENCES tasks(identity),
              kind TEXT NOT NULL, generation INTEGER NOT NULL, created_at INTEGER NOT NULL,
              UNIQUE(identity, kind, generation)
            );
            CREATE TABLE IF NOT EXISTS deliveries (
              consumer TEXT NOT NULL, notice_id INTEGER NOT NULL REFERENCES notices(id),
              delivered_at INTEGER NOT NULL, acked_at INTEGER, PRIMARY KEY(consumer, notice_id)
            );
            CREATE TABLE IF NOT EXISTS claims (
              identity TEXT PRIMARY KEY REFERENCES tasks(identity), consumer TEXT NOT NULL,
              generation INTEGER NOT NULL, claimed_at INTEGER NOT NULL, expires_at INTEGER NOT NULL,
              submission_intent TEXT NOT NULL DEFAULT 'NONE', confirmed_at INTEGER
            );
            CREATE TABLE IF NOT EXISTS events (
              id INTEGER PRIMARY KEY AUTOINCREMENT, identity TEXT NOT NULL, kind TEXT NOT NULL,
              detail TEXT NOT NULL, snapshot TEXT, occurred_at INTEGER NOT NULL
            );
            """)
        if "snapshot" not in {row[1] for row in self.db.execute("PRAGMA table_info(events)")}:
            self.db.execute("ALTER TABLE events ADD COLUMN snapshot TEXT")
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    @contextlib.contextmanager
    def _write(self) -> Any:
        """Acquire SQLite's write reservation before any state/generation read."""
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self.db.rollback()
            raise
        else:
            self.db.commit()

    def _event(self, identity: str, kind: str, detail: str, now: int) -> None:
        task = self.db.execute(
            "SELECT state,blocker_code,blocker_owner,current_eligible,submission_state "
            "FROM tasks WHERE identity=?",
            (identity,),
        ).fetchone()
        snapshot = (
            None if task is None else json.dumps(dict(task), sort_keys=True, separators=(",", ":"))
        )
        self.db.execute(
            "INSERT INTO events(identity, kind, detail, snapshot, occurred_at) VALUES(?,?,?,?,?)",
            (identity, kind, detail, snapshot, now),
        )

    def _notice(self, identity: str, kind: str, generation: int, now: int) -> None:
        self.db.execute(
            "INSERT OR IGNORE INTO notices(identity,kind,generation,created_at) VALUES(?,?,?,?)",
            (identity, kind, generation, now),
        )

    def upsert(
        self, identity: Identity, *, lane: str, owner: str, next_action: str, now: int
    ) -> None:
        if not all((lane, owner, next_action)):
            raise Refused("lane, owner and next_action are required")
        value = asdict(identity)
        with self._write():
            row = self.db.execute(
                "SELECT * FROM tasks WHERE identity=?", (identity.key(),)
            ).fetchone()
            if row is None:
                self.db.execute(
                    """INSERT INTO tasks(identity,lane,owner,next_action,created_at,updated_at,
                       campaign,stage,attempt,source_closure_sha256,manifest_sha256,prerequisite_receipts_sha256)
                       VALUES(:identity,:lane,:owner,:next_action,:now,:now,:campaign,:stage,:attempt,
                       :source_closure_sha256,:manifest_sha256,:prerequisite_receipts_sha256)""",
                    {
                        **value,
                        "identity": identity.key(),
                        "lane": lane,
                        "owner": owner,
                        "next_action": next_action,
                        "now": now,
                    },
                )
                self._event(identity.key(), "PREPARED", "registered", now)
            elif any(row[key] != value[key] for key in value):
                raise Refused("task identity is immutable; create a new attempt")

    def update_proofs(
        self,
        identity: str,
        *,
        static_verified: bool,
        authority_verified: bool,
        proof_evidence_sha256: str | None = None,
        authority_receipt_sha256: str | None = None,
        source_revoked: bool = False,
        blocker_code: str | None = None,
        blocker_owner: str | None = None,
        now: int,
    ) -> str:
        with self._write():
            row = self._task(identity)
            if row["state"] in {"CLAIMED", "RUNNING", TERMINAL} and not source_revoked:
                raise Refused("proof update cannot resurrect claimed, running, or terminal task")
            if static_verified and not self._sha(proof_evidence_sha256):
                raise Refused("static READY requires a checked proof-evidence SHA-256")
            if authority_verified and not self._sha(authority_receipt_sha256):
                raise Refused("READY requires a checked authority-receipt SHA-256")
            state = (
                HOLD
                if source_revoked
                else (STATIC_READY if static_verified and authority_verified else "PREPARED")
            )
            ready_at = row["ready_at"] or (now if state == STATIC_READY else None)
            self.db.execute(
                """UPDATE tasks SET static_verified=?,authority_verified=?,source_revoked=?,
                   current_eligible=0, proof_evidence_sha256=?,authority_receipt_sha256=?,
                   state=?,ready_at=?,blocker_code=?,blocker_owner=?,updated_at=?,
                   generation=generation+1 WHERE identity=?""",
                (
                    static_verified,
                    authority_verified,
                    source_revoked,
                    proof_evidence_sha256,
                    authority_receipt_sha256,
                    state,
                    ready_at,
                    blocker_code,
                    blocker_owner,
                    now,
                    identity,
                ),
            )
            generation = self._task(identity)["generation"]
            self._event(identity, state, blocker_code or "proof update", now)
            if state == STATIC_READY:
                self._notice(identity, "READY", generation, now)
            if source_revoked:
                self._invalidate(identity, "SOURCE_REVOKED", blocker_owner or "evidence", now)
            return state

    def set_eligibility(
        self,
        identity: str,
        *,
        eligible: bool,
        blocker_code: str | None = None,
        blocker_owner: str | None = None,
        now: int,
    ) -> str:
        with self._write():
            row = self._task(identity)
            if row["state"] in {"CLAIMED", "RUNNING", TERMINAL}:
                raise Refused(
                    "eligibility update cannot resurrect claimed, running, or terminal task"
                )
            if row["submission_state"] != "NONE":
                raise Refused(
                    "a possible or confirmed submission requires reconciliation, never reoffer"
                )
            if eligible and (
                not row["static_verified"] or not row["authority_verified"] or row["source_revoked"]
            ):
                raise Refused(
                    "current eligibility cannot create READY without static proof and authority"
                )
            if eligible:
                other = self.db.execute(
                    "SELECT identity FROM tasks WHERE lane=? AND identity<>? AND state<>? "
                    "AND current_eligible=1",
                    (row["lane"], identity, TERMINAL),
                ).fetchone()
                if other:
                    raise Refused(f"lane already has a reviewed successor: {other['identity']}")
            state = (
                ELIGIBLE_READY
                if eligible
                else (
                    STATIC_READY
                    if row["static_verified"] and row["authority_verified"]
                    else row["state"]
                )
            )
            eligible_at = row["eligible_ready_at"] or (now if eligible else None)
            self.db.execute(
                """UPDATE tasks SET current_eligible=?,state=?,eligible_ready_at=?,
                   blocker_code=?,blocker_owner=?,updated_at=?,generation=generation+1
                   WHERE identity=?""",
                (eligible, state, eligible_at, blocker_code, blocker_owner, now, identity),
            )
            generation = self._task(identity)["generation"]
            self._event(identity, state, blocker_code or "eligibility update", now)
            if state in {STATIC_READY, ELIGIBLE_READY}:
                self._notice(identity, state, generation, now)
            return state

    def claim(self, identity: str, consumer: str, *, ttl_seconds: int, now: int) -> int:
        if ttl_seconds <= 0:
            raise Refused("claim ttl must be positive")
        ambiguous = False
        generation = -1
        with self._write():
            task = self._task(identity)
            if task["state"] != ELIGIBLE_READY:
                raise Refused("only ELIGIBLE_READY tasks may be claimed")
            if task["submission_state"] != "NONE":
                self._hold(identity, "SUBMISSION_AMBIGUOUS", consumer, now)
                ambiguous = True
            else:
                claim = self.db.execute(
                    "SELECT * FROM claims WHERE identity=?", (identity,)
                ).fetchone()
                if claim and claim["submission_intent"] != "NONE" and claim["confirmed_at"] is None:
                    self._hold(identity, "SUBMISSION_AMBIGUOUS", consumer, now)
                    ambiguous = True
                elif claim and claim["expires_at"] > now:
                    raise Refused("active generation claim already exists")
                else:
                    generation = task["generation"] + 1
                    self.db.execute(
                        "UPDATE tasks SET generation=?,state='CLAIMED',updated_at=? "
                        "WHERE identity=?",
                        (generation, now, identity),
                    )
                    self.db.execute(
                        "INSERT OR REPLACE INTO claims(identity,consumer,generation,claimed_at,"
                        "expires_at,submission_intent) VALUES(?,?,?,?,?,?)",
                        (identity, consumer, generation, now, now + ttl_seconds, "NONE"),
                    )
                    self._event(identity, "CLAIMED", consumer, now)
        if ambiguous:
            raise Refused("possible submission is HOLD; reconciliation is required")
        return generation

    def submission_intent(self, identity: str, consumer: str, generation: int, *, now: int) -> bool:
        expired = False
        with self._write():
            claim = self._claim(identity, consumer, generation)
            if claim["expires_at"] < now:
                self._hold(identity, "SUBMISSION_AMBIGUOUS", consumer, now)
                expired = True
            elif claim["submission_intent"] != "NONE":
                raise Refused("submission intent already recorded for this generation")
            else:
                self.db.execute(
                    "UPDATE claims SET submission_intent='POSSIBLE' WHERE identity=?", (identity,)
                )
                self.db.execute(
                    "UPDATE tasks SET submission_state='POSSIBLE',updated_at=? WHERE identity=?",
                    (now, identity),
                )
                self._event(identity, "SUBMISSION_INTENT", consumer, now)
        if expired:
            raise Refused("claim expired at submission boundary; reconciliation is required")
        return True

    def confirm_submission(
        self, identity: str, consumer: str, generation: int, *, now: int
    ) -> None:
        with self._write():
            claim = self._claim(identity, consumer, generation)
            task = self._task(identity)
            if task["state"] != "CLAIMED" or claim["submission_intent"] != "POSSIBLE":
                raise Refused(
                    "confirmation requires the single recorded possible submission intent"
                )
            self.db.execute(
                "UPDATE claims SET submission_intent='CONFIRMED',confirmed_at=? WHERE identity=?",
                (now, identity),
            )
            self.db.execute(
                "UPDATE tasks SET submission_state='CONFIRMED',state='RUNNING',"
                "started_at=?,updated_at=? WHERE identity=?",
                (now, now, identity),
            )
            self._event(identity, "RUNNING", consumer, now)

    def terminal(
        self,
        identity: str,
        *,
        terminal_evidence: bool,
        allocation_released: bool,
        retained_allocation: bool,
        now: int,
    ) -> None:
        if allocation_released and retained_allocation:
            raise Refused("allocation cannot be released and retained")
        with self._write():
            if not terminal_evidence:
                self._hold(identity, "TERMINAL_EVIDENCE_MISSING", "operator", now)
                return
            if not allocation_released and not retained_allocation:
                self.db.execute(
                    "UPDATE tasks SET terminal_evidence=1,current_eligible=0,updated_at=? "
                    "WHERE identity=?",
                    (now, identity),
                )
                self._hold(identity, "ALLOCATION_STOP_UNVERIFIED", "operator", now)
                return
            self.db.execute(
                """UPDATE tasks SET terminal_evidence=1,allocation_released=?,
                   retained_allocation=?,state=?,terminal_at=?,current_eligible=0,
                   updated_at=? WHERE identity=?""",
                (allocation_released, retained_allocation, TERMINAL, now, now, identity),
            )
            self._event(identity, TERMINAL, "terminal evidence verified", now)

    def _hold(self, identity: str, code: str, owner: str, now: int) -> None:
        self.db.execute(
            "UPDATE tasks SET state=?,current_eligible=0,blocker_code=?,blocker_owner=?,"
            "updated_at=? WHERE identity=?",
            (HOLD, code, owner, now, identity),
        )
        self._event(identity, HOLD, code, now)

    def _invalidate(self, identity: str, code: str, owner: str, now: int) -> None:
        """Revoke undelivered action notices and claims when source authority changes."""
        self._hold(identity, code, owner, now)
        self.db.execute(
            "DELETE FROM claims WHERE identity=? AND submission_intent='NONE'", (identity,)
        )
        self.db.execute(
            "DELETE FROM deliveries WHERE notice_id IN (SELECT id FROM notices WHERE identity=?)",
            (identity,),
        )
        self.db.execute("DELETE FROM notices WHERE identity=?", (identity,))

    def reconcile(self, now: int) -> list[dict[str, Any]]:
        """Recover notices/expired claims; never dispatch or infer remote success."""
        emitted: list[dict[str, Any]] = []
        with self._write():
            rows = self.db.execute(
                "SELECT * FROM tasks WHERE state NOT IN (?)", (TERMINAL,)
            ).fetchall()
            for task in rows:
                identity = task["identity"]
                claim = self.db.execute(
                    "SELECT * FROM claims WHERE identity=?", (identity,)
                ).fetchone()
                if claim and claim["expires_at"] <= now:
                    valid = bool(
                        task["static_verified"]
                        and task["authority_verified"]
                        and task["current_eligible"]
                        and not task["source_revoked"]
                    )
                    if claim["submission_intent"] == "NONE" and valid:
                        self.db.execute("DELETE FROM claims WHERE identity=?", (identity,))
                        self.db.execute(
                            "UPDATE tasks SET state=?,updated_at=? WHERE identity=?",
                            (ELIGIBLE_READY, now, identity),
                        )
                        self._event(identity, "CLAIM_EXPIRED_REOFFERED", claim["consumer"], now)
                    elif claim["confirmed_at"] is None:
                        self._hold(identity, "SUBMISSION_AMBIGUOUS", claim["consumer"], now)
                task = self._task(identity)
                if task["state"] in {STATIC_READY, ELIGIBLE_READY}:
                    self._notice(identity, task["state"], task["generation"], now)
                if task["state"] == ELIGIBLE_READY and task["eligible_ready_at"] is not None:
                    age = now - task["eligible_ready_at"]
                    if age >= ESCALATION_SECONDS:
                        self._notice(identity, "OPERATOR_ESCALATION", task["generation"], now)
                    elif age >= ACK_REMINDER_SECONDS:
                        self._notice(identity, "ACK_REMINDER", task["generation"], now)
            for row in self.db.execute(
                "SELECT identity,kind,generation FROM notices ORDER BY id"
            ).fetchall():
                emitted.append(dict(row))
        return emitted

    def wake(self, source: str, detail: str, *, now: int) -> None:
        """Record a watcher wake without treating a PID/log as proof or READY."""
        if not source:
            raise Refused("watcher source is required")
        identity = "watch:" + hashlib.sha256(source.encode()).hexdigest()
        with self._write():
            self._event(identity, "WATCH_WAKE", detail, now)

    def deliver(self, consumer: str, *, now: int) -> list[dict[str, Any]]:
        """At-least-once delivery: unacked notices are deliberately re-delivered."""
        with self._write():
            rows = self.db.execute(
                """SELECT n.id,n.identity,n.kind,n.generation,n.created_at,t.owner,
                   t.next_action,t.blocker_code
                   FROM notices n JOIN tasks t ON t.identity=n.identity
                   LEFT JOIN deliveries d ON d.notice_id=n.id AND d.consumer=?
                   WHERE d.acked_at IS NULL AND n.generation=t.generation
                   AND t.static_verified=1 AND t.authority_verified=1 AND t.source_revoked=0
                   AND ((t.state='READY' AND n.kind='READY') OR
                        (t.state='ELIGIBLE_READY' AND t.current_eligible=1)) ORDER BY n.id""",
                (consumer,),
            ).fetchall()
            for row in rows:
                self.db.execute(
                    "INSERT OR IGNORE INTO deliveries(consumer,notice_id,delivered_at) "
                    "VALUES(?,?,?)",
                    (consumer, row["id"], now),
                )
            return [dict(row) for row in rows]

    def acknowledge(self, consumer: str, notice_id: int, *, now: int) -> None:
        with self._write():
            notice = self.db.execute(
                "SELECT identity FROM notices WHERE id=?", (notice_id,)
            ).fetchone()
            changed = self.db.execute(
                "UPDATE deliveries SET acked_at=? WHERE consumer=? AND notice_id=? "
                "AND acked_at IS NULL",
                (now, consumer, notice_id),
            ).rowcount
            if changed != 1:
                raise Refused("delivery was absent or already acknowledged")
            self._event(notice["identity"], "ACKNOWLEDGED", consumer, now)

    def metrics(self, now: int) -> dict[str, Any]:
        rows = self.db.execute("SELECT * FROM tasks ORDER BY created_at").fetchall()
        intervals = []
        blockers: dict[str, int] = {}
        for row in rows:
            if row["blocker_code"]:
                blockers[row["blocker_code"]] = blockers.get(row["blocker_code"], 0) + 1
            acks = self.db.execute(
                "SELECT MIN(acked_at) AS acked_at FROM deliveries d JOIN notices n "
                "ON n.id=d.notice_id WHERE n.identity=? AND n.kind='ELIGIBLE_READY' "
                "AND d.acked_at IS NOT NULL",
                (row["identity"],),
            ).fetchone()["acked_at"]
            intervals.append(
                {
                    "identity": row["identity"],
                    "state": row["state"],
                    "ready_at": row["ready_at"],
                    "eligible_ready_at": row["eligible_ready_at"],
                    "started_at": row["started_at"],
                    "terminal_at": row["terminal_at"],
                    "waiting_seconds": (
                        None
                        if row["eligible_ready_at"] is None
                        else (row["started_at"] or now) - row["eligible_ready_at"]
                    ),
                    "blocker_code": row["blocker_code"],
                    "blocker_owner": row["blocker_owner"],
                    "resource_eligible": bool(row["current_eligible"]),
                    "acknowledged_at": acks,
                    "eligible_to_ack_seconds": (
                        None
                        if acks is None or row["eligible_ready_at"] is None
                        else acks - row["eligible_ready_at"]
                    ),
                }
            )
        acknowledged = sorted(
            item["eligible_to_ack_seconds"]
            for item in intervals
            if item["eligible_to_ack_seconds"] is not None
        )
        started = sorted(
            item["waiting_seconds"] for item in intervals if item["started_at"] is not None
        )

        def summary(values: list[int]) -> dict[str, Any]:
            if len(values) < 10:
                return {
                    "n": len(values),
                    "median": None,
                    "p95": None,
                    "evidence": "INSUFFICIENT_LT10",
                }
            return {
                "n": len(values),
                "median": statistics.median(values),
                "p95": values[math.ceil(0.95 * len(values)) - 1],
                "evidence": "DESCRIPTIVE_NEAREST_RANK",
            }

        timeline = [
            {
                **dict(event),
                "snapshot": None if event["snapshot"] is None else json.loads(event["snapshot"]),
            }
            for event in self.db.execute(
                "SELECT identity,kind,detail,snapshot,occurred_at FROM events ORDER BY id"
            ).fetchall()
        ]
        return {
            "schema": SCHEMA,
            "measured_at": now,
            "intervals": intervals,
            "blockers": blockers,
            "timeline": timeline,
            "eligible_to_ack": summary(acknowledged),
            "eligible_to_start": summary(started),
            "sample_note": (
                "Operational intervals are descriptive; no population service guarantee."
            ),
        }

    def _task(self, identity: str) -> sqlite3.Row:
        row = self.db.execute("SELECT * FROM tasks WHERE identity=?", (identity,)).fetchone()
        if row is None:
            raise Refused("unknown task identity")
        return row

    def _claim(self, identity: str, consumer: str, generation: int) -> sqlite3.Row:
        row = self.db.execute(
            "SELECT * FROM claims WHERE identity=? AND consumer=? AND generation=?",
            (identity, consumer, generation),
        ).fetchone()
        if row is None:
            raise Refused("claim generation mismatch")
        return row

    @staticmethod
    def _sha(value: str | None) -> bool:
        return (
            isinstance(value, str)
            and len(value) == 64
            and not (set(value) - set("0123456789abcdef"))
        )


def _json(value: str) -> dict[str, Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise Refused("JSON must be an object")
    return parsed


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ops", type=Path, required=True)
    parser.add_argument("--now", type=int, default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    add = sub.add_parser("add")
    add.add_argument("--identity", required=True)
    add.add_argument("--lane", required=True)
    add.add_argument("--owner", required=True)
    add.add_argument("--next-action", required=True)
    proof = sub.add_parser("proof")
    proof.add_argument("identity")
    proof.add_argument("--static", action="store_true")
    proof.add_argument("--authority", action="store_true")
    proof.add_argument("--proof-evidence-sha256")
    proof.add_argument("--authority-receipt-sha256")
    proof.add_argument("--revoked", action="store_true")
    proof.add_argument("--blocker")
    proof.add_argument("--blocker-owner")
    eligible = sub.add_parser("eligible")
    eligible.add_argument("identity")
    eligible.add_argument("--yes", action="store_true")
    eligible.add_argument("--blocker")
    eligible.add_argument("--blocker-owner")
    claim = sub.add_parser("claim")
    claim.add_argument("identity")
    claim.add_argument("consumer")
    claim.add_argument("--ttl", type=int, default=300)
    intent = sub.add_parser("intent")
    intent.add_argument("identity")
    intent.add_argument("consumer")
    intent.add_argument("generation", type=int)
    confirm = sub.add_parser("confirm")
    confirm.add_argument("identity")
    confirm.add_argument("consumer")
    confirm.add_argument("generation", type=int)
    terminal = sub.add_parser("terminal")
    terminal.add_argument("identity")
    terminal.add_argument("--evidence", action="store_true")
    terminal.add_argument("--released", action="store_true")
    terminal.add_argument("--retained", action="store_true")
    deliver = sub.add_parser("deliver")
    deliver.add_argument("consumer")
    ack = sub.add_parser("ack")
    ack.add_argument("consumer")
    ack.add_argument("notice", type=int)
    wake = sub.add_parser("wake")
    wake.add_argument("--source", required=True)
    wake.add_argument("--detail", required=True)
    sub.add_parser("reconcile")
    sub.add_parser("metrics")
    args = parser.parse_args(list(argv) if argv is not None else None)
    now = args.now if args.now is not None else int(time.time())
    journal = Journal(args.ops)
    try:
        if args.command == "add":
            journal.upsert(
                Identity.from_value(_json(args.identity)),
                lane=args.lane,
                owner=args.owner,
                next_action=args.next_action,
                now=now,
            )
            value: Any = {"status": "PREPARED"}
        elif args.command == "proof":
            value = {
                "state": journal.update_proofs(
                    args.identity,
                    static_verified=args.static,
                    authority_verified=args.authority,
                    proof_evidence_sha256=args.proof_evidence_sha256,
                    authority_receipt_sha256=args.authority_receipt_sha256,
                    source_revoked=args.revoked,
                    blocker_code=args.blocker,
                    blocker_owner=args.blocker_owner,
                    now=now,
                )
            }
        elif args.command == "eligible":
            value = {
                "state": journal.set_eligibility(
                    args.identity,
                    eligible=args.yes,
                    blocker_code=args.blocker,
                    blocker_owner=args.blocker_owner,
                    now=now,
                )
            }
        elif args.command == "claim":
            value = {
                "generation": journal.claim(
                    args.identity, args.consumer, ttl_seconds=args.ttl, now=now
                )
            }
        elif args.command == "intent":
            journal.submission_intent(args.identity, args.consumer, args.generation, now=now)
            value = {"status": "POSSIBLE"}
        elif args.command == "confirm":
            journal.confirm_submission(args.identity, args.consumer, args.generation, now=now)
            value = {"status": "RUNNING"}
        elif args.command == "terminal":
            journal.terminal(
                args.identity,
                terminal_evidence=args.evidence,
                allocation_released=args.released,
                retained_allocation=args.retained,
                now=now,
            )
            value = {"status": "recorded"}
        elif args.command == "deliver":
            value = journal.deliver(args.consumer, now=now)
        elif args.command == "ack":
            journal.acknowledge(args.consumer, args.notice, now=now)
            value = {"status": "ACKED"}
        elif args.command == "wake":
            journal.wake(args.source, args.detail, now=now)
            value = {"status": "WOKE"}
        elif args.command == "reconcile":
            value = journal.reconcile(now)
        else:
            value = journal.metrics(now)
        print(json.dumps(value, sort_keys=True))
        return 0
    except Refused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    finally:
        journal.close()


if __name__ == "__main__":
    raise SystemExit(main())
