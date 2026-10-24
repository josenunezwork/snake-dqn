"""Independent, stdlib-only adversarial checks for the R33 ops contracts.

These checks intentionally live apart from the Sol-P package tests.  They encode
the rejection boundary expected by the ruling and do not use project fixtures,
Torch, NumPy, or the pytest API.
"""

from __future__ import annotations

import ast
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from research.r33_verification import lifecycle
from tools.ops.package_spec import (
    Input,
    PackageRefused,
    empty_ledger,
    publish_json,
    read_ledger,
    stage_inputs,
)
from tools.ops.proof_dependencies import Refused as DependencyRefused
from tools.ops.proof_dependencies import compare, delta_input, dependency_inventory
from tools.ops.task_journal import Identity, Journal, Refused

HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
NAMESPACE = "r33-independent"
ALLOCATION = "allocation-0001"


def _identity() -> Identity:
    return Identity("t34", "r33", "attempt-1", HASH_A, HASH_B, HASH_C)


def _frozen_identity() -> dict[str, str]:
    return {"pin": "0" * 64, "environment_sha256": "1" * 64, "packet": lifecycle.PACKET}


def _ready(journal: Journal, identity: Identity) -> str:
    journal.upsert(identity, lane="cpu", owner="claude", next_action="inspect", now=1)
    key = identity.key()
    journal.update_proofs(
        key,
        static_verified=True,
        authority_verified=True,
        proof_evidence_sha256=HASH_A,
        authority_receipt_sha256=HASH_B,
        now=2,
    )
    self_state = journal.set_eligibility(key, eligible=True, now=3)
    if self_state != "ELIGIBLE_READY":
        raise AssertionError(self_state)
    return key


class JournalAdversarialTests(unittest.TestCase):
    def test_confirmation_requires_intent_and_terminal_without_release_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            journal = Journal(Path(directory))
            try:
                key = _ready(journal, _identity())
                generation = journal.claim(key, "operator", ttl_seconds=10, now=4)
                with self.assertRaises(Refused):
                    journal.confirm_submission(key, "operator", generation, now=5)
                journal.submission_intent(key, "operator", generation, now=5)
                journal.confirm_submission(key, "operator", generation, now=6)
                try:
                    journal.terminal(
                        key,
                        terminal_evidence=True,
                        allocation_released=False,
                        retained_allocation=False,
                        now=7,
                    )
                except Refused:
                    pass
                else:
                    self.assertEqual(journal._task(key)["state"], "HOLD")
            finally:
                journal.close()

    def test_revocation_removes_ready_notice_and_terminal_cannot_be_revived(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            journal = Journal(Path(directory))
            try:
                identity = _identity()
                key = _ready(journal, identity)
                journal.update_proofs(
                    key,
                    static_verified=False,
                    authority_verified=False,
                    source_revoked=True,
                    blocker_code="STALE_PROOF",
                    now=4,
                )
                self.assertEqual(journal.deliver("operator", now=5), [])

                key = _ready(journal, Identity("t34", "r33", "attempt-2", HASH_A, HASH_B, HASH_C))
                journal.terminal(
                    key,
                    terminal_evidence=True,
                    allocation_released=True,
                    retained_allocation=False,
                    now=4,
                )
                with self.assertRaises(Refused):
                    journal.set_eligibility(key, eligible=True, now=5)
                with self.assertRaises(Refused):
                    journal.update_proofs(
                        key,
                        static_verified=True,
                        authority_verified=True,
                        proof_evidence_sha256=HASH_A,
                        authority_receipt_sha256=HASH_B,
                        now=5,
                    )
            finally:
                journal.close()

    def test_expired_possible_submission_enters_hold_without_reoffer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            journal = Journal(Path(directory))
            try:
                key = _ready(journal, _identity())
                generation = journal.claim(key, "operator", ttl_seconds=1, now=4)
                journal.submission_intent(key, "operator", generation, now=4)
                journal.reconcile(now=5)
                with self.assertRaises(Refused):
                    journal.claim(key, "replacement", ttl_seconds=10, now=6)
                self.assertEqual(journal.deliver("operator", now=6), [])
            finally:
                journal.close()


class PackageAdversarialTests(unittest.TestCase):
    def test_existing_destination_is_not_clobbered(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.json"
            source.write_text("source\n")
            staging = root / "staging"
            target = staging / "inputs/runtime.json"
            target.parent.mkdir(parents=True)
            target.write_text("sentinel\n")
            with self.assertRaises((FileExistsError, PackageRefused)):
                stage_inputs(staging, [Input(source, "inputs/runtime.json")])
            self.assertEqual(target.read_text(), "sentinel\n")

    def test_every_unit_identity_is_checked_before_terminal_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first"
            lifecycle.export_unit(first, NAMESPACE, ALLOCATION, 0)
            ledger_path = root / "ledger.json"
            lifecycle.build_verified_ledger(first, ledger_path, NAMESPACE, ALLOCATION)
            resumed = root / "resumed"
            lifecycle.consume(NAMESPACE, ALLOCATION, ledger_path, 3, resumed)
            old = first / "units/unit-000000.json"
            destination = resumed / "units/unit-000000.json"
            for suffix in ("", ".sha256", ".complete"):
                destination.parent.mkdir(parents=True, exist_ok=True)
                (destination.parent / f"{destination.name}{suffix}").write_bytes(
                    (old.parent / f"{old.name}{suffix}").read_bytes()
                )
            foreign = json.loads((resumed / "units/unit-000002.json").read_text())
            foreign["identity"]["packet"] = "foreign"
            (resumed / "units/unit-000002.json").write_text(
                json.dumps(foreign, sort_keys=True) + "\n"
            )
            payload = (resumed / "units/unit-000002.json").read_bytes()
            (resumed / "units/unit-000002.json.sha256").write_text(
                f"{hashlib.sha256(payload).hexdigest()}  unit-000002.json\n"
            )
            for path in resumed.glob("verified-ledger.json*"):
                path.unlink()
            with self.assertRaises(PackageRefused):
                lifecycle.terminal(resumed, NAMESPACE, ALLOCATION, 3)

    def test_ledger_identity_rejects_foreign_pin_and_allocation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.json"
            publish_json(path, empty_ledger(NAMESPACE, ALLOCATION, _frozen_identity()))
            with self.assertRaises(PackageRefused):
                read_ledger(path, NAMESPACE, ALLOCATION, {**_frozen_identity(), "pin": HASH_B})
            with self.assertRaises(PackageRefused):
                read_ledger(path, NAMESPACE, "foreign-allocation", _frozen_identity())


def _dependencies() -> list[dict[str, str]]:
    return [
        {"name": kind, "kind": kind, "sha256": hashlib.sha256(kind.encode()).hexdigest()}
        for kind in (
            "source",
            "input",
            "environment",
            "platform",
            "command",
            "fixture",
            "allocation",
            "contract",
        )
    ]


class DependencyAdversarialTests(unittest.TestCase):
    def test_delta_rejects_foreign_package_and_nonhex_reused_proof(self) -> None:
        previous = dependency_inventory("package-a", _dependencies())
        current = dependency_inventory(
            "package-b",
            _dependencies()[:-1] + [{"name": "contract", "kind": "contract", "sha256": HASH_B}],
        )
        with self.assertRaises(DependencyRefused):
            compare(previous, current)
        with self.assertRaises(DependencyRefused):
            delta_input(
                previous=previous,
                current=dependency_inventory(
                    "package-a",
                    _dependencies()[:-1]
                    + [{"name": "contract", "kind": "contract", "sha256": HASH_B}],
                ),
                reused_proofs=[{"path": "old.json", "sha256": "z" * 64}],
                reviewer="codex-v",
                live_obligations=["native_canary", "live_admission"],
                charter_opt_in=True,
            )


class InventoryAdversarialTests(unittest.TestCase):
    def test_required_inventory_and_process_table_are_explicit(self) -> None:
        root = Path(__file__).parents[1]
        inventory = json.loads(
            (root / "research/r33_verification/contract_inventory.json").read_text()
        )
        self.assertEqual(inventory["schema"], "r33-research-contract-inventory/v1")
        self.assertGreaterEqual(len(inventory["linux_required"]), 4)
        self.assertTrue(inventory["native_pending"])
        table = json.loads(
            (root / "research/r33_verification/process_fixture_table.json").read_text()
        )
        self.assertEqual(set(table["consumers"]), {"startup", "runtime", "recovery"})
        by_id = {case["id"]: case for case in table["cases"]}
        self.assertEqual(by_id["busy-descendant"]["expected"], "BLOCK")
        self.assertEqual(by_id["unknown-priority"]["expected"], "BLOCK")
        self.assertEqual(by_id["quoted-app-path"]["expected"], "ALLOW_PRESENCE_EXCEPTION")

    def test_required_inventory_detects_a_silent_skip_mutation(self) -> None:
        root = Path(__file__).parents[1]
        inventory = json.loads(
            (root / "research/r33_verification/contract_inventory.json").read_text()
        )
        required = set(inventory["linux_required"])

        def collected_nodes() -> set[str]:
            found: set[str] = set()
            for identifier in required:
                path_text, _, qualified = identifier.partition("::")
                tree = ast.parse((root / path_text).read_text(), filename=path_text)
                names = [
                    node.name
                    for node in tree.body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]
                for node in tree.body:
                    if isinstance(node, ast.ClassDef):
                        names.extend(
                            f"{node.name}::{child.name}"
                            for child in node.body
                            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                        )
                if qualified in names:
                    found.add(identifier)
            return found

        self.assertEqual(required, collected_nodes())
        mutated = set(required)
        mutated.remove(next(iter(required)))
        self.assertTrue(
            required - mutated, "a missing required node must remain visible as INCOMPLETE"
        )
