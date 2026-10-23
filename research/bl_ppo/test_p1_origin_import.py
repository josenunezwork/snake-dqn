"""Amendment-06 origin-import contracts and a fixture-scaled continuation.

The positive case uses the existing tiny P1 entrypoint with synthetic natural
support.  It is software evidence only: the real ruled origin pack remains
immutable and no production actor, refit, or gate run is launched here.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from research.bl_ppo import p1_origin_import, p1_runtime
from research.bl_ppo.exports import digest, export_json, verify_export
from research.bl_ppo.recovery import copy_snapshot_closure, read_snapshot, snapshot_files
from research.bl_ppo.test_p1_replay_amendment02 import synthetic_natural_support, tiny_p1
from research.bl_ppo.test_p1_y3_resume_entrypoint import _cadence_clock, _source

__all__ = ["synthetic_natural_support", "tiny_p1"]
_SOURCE_BYTES = None


def _diff_scope() -> tuple[list[str], list[dict[str, str]]]:
    paths = [
        "research/bl_ppo/run.py",
        "research/bl_ppo/p1_runtime.py",
        "research/bl_ppo/p1_source_repair.py",
        "research/bl_ppo/test_p1_source_repair.py",
        "research/bl_ppo/p1_origin_import.py",
        "research/bl_ppo/test_p1_origin_import.py",
    ]
    root = Path(__file__).parents[2]
    rows = [{"path": path, "repair_sha256": digest(root / path)} for path in paths]
    return paths, rows


def _make_import_fixture(tmp_path: Path, monkeypatch, tiny_p1):
    """Generate an exact tiny origin and flatten its immutable import set."""
    global _SOURCE_BYTES
    # Reuse immutable synthetic origin bytes for refusal cases; the positive
    # repair still executes the real collector, learner and snapshot path.
    if _SOURCE_BYTES is None:
        _cadence_clock(monkeypatch)
        _, source, argv = _source(tmp_path / "origin-run", monkeypatch, tiny_p1)
        assert p1_runtime.main(argv) == 0
        _SOURCE_BYTES = (source, argv)
    source, argv = _SOURCE_BYTES

    origin_receipt = json.loads((source / "receipt.json").read_text())
    origin = source / "origin" / origin_receipt["origin_snapshot"]["name"]
    origin_status = source / "origin" / "training-status.json"
    assert read_snapshot(origin)["counters"]["actor_transitions"] == 0
    assert json.loads(origin_status.read_text())["critic_qualified"] is True

    source_control_path = Path(argv[2])
    root = tmp_path / "import-inputs"
    root.mkdir()
    copy_snapshot_closure(origin, root)
    audit_source = source_control_path.parent / "bank-audit.json"
    for suffix in ("", ".sha256", ".complete"):
        (root / f"bank-audit.json{suffix}").write_bytes(
            Path(str(audit_source) + suffix).read_bytes()
        )
    for source_path, target_name in ((origin_status, "training-status.json"),):
        for suffix in ("", ".sha256", ".complete"):
            target = root / f"{target_name}{suffix}"
            target.write_bytes(Path(str(source_path) + suffix).read_bytes())

    evidence_sources = [
        source / "origin" / "late-body-rows-index.json",
        source / "origin" / "late-body-rows-0000.npz",
    ]
    for source_path in evidence_sources:
        target = root / source_path.name
        for suffix in ("", ".sha256", ".complete"):
            target.with_name(f"{target.name}{suffix}").write_bytes(
                Path(str(source_path) + suffix).read_bytes()
            )
    evidence_catalog = [
        {
            "name": path.name,
            "sha256": verify_export(root / path.name),
            "size_bytes": (root / path.name).stat().st_size,
        }
        for path in evidence_sources
    ]

    catalog = [
        {"name": path.name, "sha256": verify_export(path), "size_bytes": path.stat().st_size}
        for path in snapshot_files(origin)
    ]
    failed = export_json(
        root / "failed-receipt.json",
        {
            "status": "INCOMPLETE",
            "stop_verified": True,
            "cleanup_error": None,
            "exports": [
                {"path": row["name"], "sha256": row["sha256"]} for row in catalog + evidence_catalog
            ],
        },
    )
    failed_sha = failed["sha256"]
    offline = export_json(
        root / "offline-audit.json",
        {
            "status": "HOLD_MISSING_OPERATIONAL_PROOF",
            "evidence_sha256": {
                "failed-receipt.json": failed_sha,
                **{f"origin/{row['name']}": row["sha256"] for row in catalog + evidence_catalog},
            },
            "critic_qualified": True,
        },
    )

    origin_control = json.loads(source_control_path.read_text())
    old_commit = origin_control["source_commit"]
    repair_commit = "c" * 40
    paths, scope_rows = _diff_scope()
    origin_state = read_snapshot(origin)
    origin_status_value = json.loads(origin_status.read_text())
    import_block = {
        "schema": p1_origin_import.SCHEMA,
        "origin_snapshot": {"path": origin.name, "sha256": verify_export(origin)},
        "origin_status": {
            "path": "training-status.json",
            "sha256": verify_export(root / "training-status.json"),
        },
        "failed_receipt": {"path": "failed-receipt.json", "sha256": failed_sha},
        "offline_audit": {"path": "offline-audit.json", "sha256": offline["sha256"]},
        "source_commit": old_commit,
        "snapshot_catalog": catalog,
        "evidence_catalog": evidence_catalog,
        "expected_identity": {
            "arm": "F",
            "seed": 20261021,
            "target": tiny_p1.target,
            "natural_collection_world_id_offset": 3_011_000,
            "registration_sha256": origin_control["registration_sha256"],
            "evaluation_bank_sha256": origin_control["evaluation_bank_sha256"],
            "e8_critic_convergence_amendment_04_sha256": origin_control.get(
                "e8_critic_convergence_amendment_04_sha256"
            ),
        },
        "expected_counters": {
            key: value
            for key, value in origin_state["counters"].items()
            if key.startswith("critic_refit") or key.startswith("critic_fixed_holdout")
        },
        "critic_refit_sha256": p1_origin_import.canonical_sha256(
            origin_status_value["critic_refit"]
        ),
        "prefix_witness": {
            "actor_actions": 49_152,
            "full_tape_sha256": "fb3c8c7b89bbe0f347ae89799fbb66cf9bf8373e987b1d8212fe286c7e95280c",
        },
        "allowed_diff_paths": paths,
        "diff_scope": {
            "source_commit": old_commit,
            "repair_commit": repair_commit,
            "paths": scope_rows,
        },
    }
    # Bind the synthetic pack to its generated bytes while retaining the
    # validator's production checks and error paths.
    monkeypatch.setattr(p1_origin_import, "ORIGIN_COMMIT", old_commit)
    monkeypatch.setattr(p1_origin_import, "ORIGIN_SNAPSHOT_SHA256", verify_export(origin))
    monkeypatch.setattr(
        p1_origin_import, "ORIGIN_STATUS_SHA256", verify_export(root / "training-status.json")
    )
    monkeypatch.setattr(p1_origin_import, "FAILED_RECEIPT_SHA256", failed_sha)
    monkeypatch.setattr(p1_origin_import, "OFFLINE_AUDIT_SHA256", offline["sha256"])
    control = copy.deepcopy(origin_control)
    control["source_commit"] = repair_commit
    control["target"] = tiny_p1.target
    control["resume"]["allocation"] = p1_origin_import.SOURCE_ALLOCATION
    control["resume"]["namespace"] = "test-y3-pack04-repair06"
    control["natural_collection_world_id_offset"] = 3_011_000
    control["resume"]["max_outcome_blind_retries"] = 1
    control["y3_origin_import"] = import_block
    control_path = root / "p1-control.json"
    export_json(control_path, control)
    export_json(
        root / "verified-ledger.json",
        {
            "schema": "bl-p1-verified-ledger/v1",
            "namespace": control["resume"]["namespace"],
            "allocation": control["resume"]["allocation"],
            "units": [],
        },
    )
    monkeypatch.setenv("BL_PPO_COMMIT", repair_commit)
    return root, control_path, import_block, origin, origin_receipt


@pytest.mark.parametrize(
    "field", ["origin_snapshot", "origin_status", "failed_receipt", "offline_audit"]
)
def test_origin_import_refuses_digest_mismatch(tmp_path, monkeypatch, tiny_p1, field):
    root, _, block, *_ = _make_import_fixture(tmp_path, monkeypatch, tiny_p1)
    broken = copy.deepcopy(block)
    broken[field]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="digest mismatch|ruled immutable evidence"):
        p1_origin_import.validate(
            {
                "source_commit": "c" * 40,
                "registration_sha256": "",
                "evaluation_bank_sha256": "",
                "natural_collection_world_id_offset": 3_011_000,
                "y3_origin_import": broken,
            },
            root,
        )


def test_origin_import_refuses_catalog_identity_and_out_of_scope_diff(
    tmp_path, monkeypatch, tiny_p1
):
    root, _, block, *_ = _make_import_fixture(tmp_path, monkeypatch, tiny_p1)
    for kind in ("catalog", "identity", "scope"):
        broken = copy.deepcopy(block)
        if kind == "catalog":
            broken["snapshot_catalog"] = broken["snapshot_catalog"][:-1]
            message = "transitive catalog"
        elif kind == "identity":
            broken["expected_identity"]["arm"] = "G"
            message = "identity"
        else:
            broken["allowed_diff_paths"].append("src/forbidden.py")
            message = "out-of-scope"
        with pytest.raises(ValueError, match=message):
            p1_origin_import.validate(
                {
                    "source_commit": "c" * 40,
                    "registration_sha256": block["expected_identity"]["registration_sha256"],
                    "evaluation_bank_sha256": block["expected_identity"]["evaluation_bank_sha256"],
                    "natural_collection_world_id_offset": 3_011_000,
                    "y3_origin_import": broken,
                },
                root,
            )


def test_origin_import_refuses_nonempty_ledger(tmp_path, monkeypatch, tiny_p1):
    root, control_path, _, origin, _ = _make_import_fixture(tmp_path, monkeypatch, tiny_p1)
    ledger = json.loads((root / "verified-ledger.json").read_text())
    ledger["units"].append(
        {
            "name": "operational-test-y3-pack04-repair06-source_latebody_06_e8_04_pack04-1.pt",
            "unit_id": "test-y3-pack04-repair06/source_latebody_06_e8_04_pack04/1",
            "sha256": verify_export(origin),
            "size": origin.stat().st_size,
        }
    )
    for suffix in ("", ".sha256", ".complete"):
        (root / f"verified-ledger.json{suffix}").unlink()
    export_json(root / "verified-ledger.json", ledger)
    with pytest.raises(ValueError, match="empty verified ledger"):
        p1_runtime._load_control(control_path, "y3-source")


def test_origin_import_continuation_runs_actor_only_and_copies_critic_receipt(
    tmp_path, monkeypatch, tiny_p1
):
    root, control_path, block, _, origin_receipt = _make_import_fixture(
        tmp_path, monkeypatch, tiny_p1
    )
    _cadence_clock(monkeypatch, first_actor_capture=1)
    # The production declaration remains fixed at the ruled 49,152-action
    # witness.  This fixture scales that count to its already-produced tape.
    source_trace = origin_receipt["source_trace"]
    scaled_actions = source_trace["full_tape_actions"]
    assert scaled_actions > 0
    original_validate = p1_origin_import.validate

    def scaled_validate(value, artifact_root):
        # Keep the validator's fixture-only expected target aligned with the
        # same tiny shape used to create the imported pause.
        monkeypatch.setattr(p1_origin_import.p1_contract, "SHAPE", tiny_p1)
        monkeypatch.setattr(p1_origin_import, "PREFIX_ACTIONS", scaled_actions)
        monkeypatch.setattr(
            p1_origin_import, "FAILED_PREFIX_SHA256", source_trace["full_tape_sha256"]
        )
        value["y3_origin_import"]["prefix_witness"]["actor_actions"] = scaled_actions
        value["y3_origin_import"]["prefix_witness"]["full_tape_sha256"] = source_trace[
            "full_tape_sha256"
        ]
        imported = original_validate(value, artifact_root)
        imported["prefix_witness"] = {
            "actor_actions": scaled_actions,
            "full_tape_sha256": source_trace["full_tape_sha256"],
        }
        return imported

    monkeypatch.setattr(p1_origin_import, "validate", scaled_validate)
    # Install spies only after the qualified synthetic origin exists.
    from research.bl_ppo import learner as learner_module
    from research.bl_ppo.refit import CompleteLifeRefitBank

    learner_type = getattr(learner_module.PPOLearner, "func", learner_module.PPOLearner)
    original_update = learner_type.update
    calls = {"actor": 0}

    def actor_only_update(self, *args, **kwargs):
        if kwargs.get("actor_enabled") is not True:
            raise AssertionError("amendment-06 entered a refit learner update")
        calls["actor"] += 1
        return original_update(self, *args, **kwargs)

    monkeypatch.setattr(learner_type, "update", actor_only_update)
    for name in (
        "fit_complete_targets",
        "solve_e8_critic_convergence_amendment_03",
        "solve_e8_critic_convergence_amendment_04",
        "qualification",
    ):
        monkeypatch.setattr(
            CompleteLifeRefitBank,
            name,
            lambda *args, _name=name, **kwargs: (_ for _ in ()).throw(
                AssertionError(f"entered refit/gate function {_name}")
            ),
        )

    # Synthetic support is needed only to generate the origin.  Restoring a
    # qualified bank must not seed new rows while the production guard is on.
    import inspect

    original_bank_init = inspect.getclosurevars(CompleteLifeRefitBank.__init__).nonlocals[
        "original"
    ]
    monkeypatch.setattr(CompleteLifeRefitBank, "__init__", original_bank_init)

    supervisor = tmp_path / "repair-supervisor"
    supervisor.mkdir()
    (supervisor / "spec.json").write_text(
        json.dumps(
            {"release_endpoint": "TEST-ONLY-repair", "generation": "fixture-repair-generation"}
        )
    )
    monkeypatch.setenv("COLAB_CLI_ROOT", str(supervisor))
    output = tmp_path / "repair-output"
    argv = [
        "y3-source",
        "--control",
        str(control_path),
        "--output",
        str(output),
        "--checkpoint",
        str(Path(__file__).resolve().parents[3] / "snake-dqn/saved_snakes/b8_s3_t7_20261016.pth"),
    ]
    assert p1_runtime.main(argv) == 0
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["status"] == "COMPLETE_SOURCE_CONTINUATION"
    assert receipt["origin_mode"] == "IMPORTED_VERIFIED_ORIGIN"
    assert receipt["critic_refit_sha256"] == block["critic_refit_sha256"]
    assert (
        p1_origin_import.canonical_sha256(receipt["critic_refit"]) == block["critic_refit_sha256"]
    )
    assert receipt["prefix_witness"]["actor_actions"] == scaled_actions
    assert receipt["prefix_witness"]["status"] == "MATCH"
    assert calls["actor"] >= 4


def _invoke_late_body_supplement():
    from research.bl_ppo import late_body

    late_body.collect_refit_late_supplement(None, None, None)


def _invoke_critic_solve():
    from research.bl_ppo.refit import CompleteLifeRefitBank

    CompleteLifeRefitBank.solve_e8_critic_convergence_amendment_04(object(), object())


def _invoke_critic_update():
    from research.bl_ppo import learner

    learner_type = getattr(learner.PPOLearner, "func", learner.PPOLearner)
    learner_type.update(object(), object(), actor_enabled=False)


@pytest.mark.parametrize(
    "invoke",
    [_invoke_late_body_supplement, _invoke_critic_solve, _invoke_critic_update],
    ids=["late-body-supplement", "critic-solve", "critic-update"],
)
def test_import_guard_rejects_refit_entry_even_when_inner_error_is_caught(invoke):
    """The import context cannot be bypassed by swallowing its finite refusal."""
    with pytest.raises(ValueError, match="VOID: imported origin entered refit"):
        with p1_origin_import.guard_refit_entry():
            try:
                invoke()
            except ValueError:
                pass
