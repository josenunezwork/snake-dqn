"""Synthetic count, provenance and spent-claim tests; never open real inputs."""

from __future__ import annotations

import copy
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from research.bl_ppo.exports import digest, export_json
from research.bl_residual import f4_postmortem as entry
from research.bl_residual import f4_postmortem_guard as guard
from research.bl_residual import f4_postmortem_runtime as runtime
from research.bl_residual.f4_postmortem_counts import (
    COUNT_MASKS,
    AggregateObserver,
    valid_dose,
)
from research.bl_residual.fit import heldout_metrics
from research.bl_residual.model import FrozenB8Residual
from src.model.ego2s_network import Ego2sNet


def fixture_masks() -> dict:
    n = 8
    cache = {
        "features": torch.zeros(n, 256),
        "q": torch.zeros(n, 6),
        "legal": torch.ones(n, 6, dtype=torch.bool),
        "action_mask": torch.ones(n, 6, dtype=torch.bool),
        "bare": torch.zeros(n, dtype=torch.long),
        "died": torch.zeros(n, 6, dtype=torch.bool),
        "status": torch.ones(n, 6, dtype=torch.bool),
        "mass": torch.zeros(n, 6),
        "ordinary": torch.ones(n, dtype=torch.bool),
        "natural_stream": torch.tensor([True] * 7 + [False]),
        "sampling_prob": torch.ones(n),
        "life_id": torch.arange(n),
    }
    q = torch.zeros(n, 6)
    q[:, 1] = torch.tensor([0.0, 0.01, 0.1, 0.5, 0.009, 0.1, 0.5, 0.01])
    cache["died"][[1, 2, 3, 4, 6, 7], 1] = True
    cache["status"][2, 1] = False
    cache["status"][3, 0] = False
    cache["status"][4, :2] = False
    # Illegal selected death is not confirmed; fallback rejects boost Q.
    cache["legal"][6, 1] = False
    cache["action_mask"][5] = False
    q[5, 5] = 100
    received = []
    metrics = heldout_metrics(
        FrozenB8Residual(Ego2sNet(7, 60)),
        cache,
        natural_archive_complete=True,
        candidate_q=q,
        diagnostics_observer=received.append,
    )
    return {"masks": received[0], "H": metrics["harmful_override_count_natural"]}


def test_aggregates_reuse_existing_masks_and_include_zero_parent_and_bins() -> None:
    fixture = fixture_masks()
    observer = AggregateObserver(["a", "b", "zero"])
    observer(fixture["masks"], ("a", "b", "a", "b", "a", "b", "a", "zero"))
    result = {"dose": 256, "checkpoint_sha256": "a" * 64, **observer.result}
    assert valid_dose(result, ["a", "b", "zero"], natural_rows=7)
    assert result["totals"] == dict(zip(COUNT_MASKS, [7, 6, 3, 3, 1, 1, 1, 1, 3, 4]))
    assert result["totals"]["H"] == fixture["H"]
    assert [b["counts"]["N_natural"] for b in result["margin_bins"]] == [2, 1, 2, 2]
    assert set(result["parents"][-1]["totals"].values()) == {0}
    assert observer.total_rows == 8
    with pytest.raises(ValueError, match="one observer"):
        observer(fixture["masks"], ("a",) * 8)
    bad = copy.deepcopy(result)
    bad["margin_bins"][0]["counts"]["H"] += 1
    assert not valid_dose(bad, ["a", "b", "zero"], natural_rows=7)
    bad = copy.deepcopy(result)
    bad["parents"][0]["row_ids"] = ["forbidden"]
    assert not valid_dose(bad, ["a", "b", "zero"], natural_rows=7)


def test_zero_denominators_remain_counts_without_fabricated_rates() -> None:
    masks = fixture_masks()["masks"]
    masks["natural"][:] = False
    for b in masks["margin_bins"]:
        b[:] = False
    observer = AggregateObserver(["a"])
    observer(masks, ("a",) * 8)
    result = {"dose": 256, "checkpoint_sha256": "a" * 64, **observer.result}
    assert valid_dose(result, ["a"], natural_rows=0)
    assert all(v == 0 for v in result["totals"].values())
    assert set(result["totals"]) == set(COUNT_MASKS)  # no risk/rate output


def test_provenance_rejects_size_digest_partial_and_symlink(tmp_path: Path) -> None:
    path = tmp_path / "synthetic.json"
    export_json(path, {"synthetic": True})
    expected = {"path": str(path), "size_bytes": path.stat().st_size, "sha256": digest(path)}
    assert guard.identity(path, expected)["sha256"] == expected["sha256"]
    with pytest.raises(ValueError, match="size"):
        guard.identity(path, {**expected, "size_bytes": 1})
    with pytest.raises(ValueError, match="hash"):
        guard.identity(path, {**expected, "sha256": "0" * 64})
    Path(str(path) + ".complete").unlink()
    with pytest.raises(ValueError, match="missing"):
        guard.identity(path, expected)
    alias = tmp_path / "alias.json"
    alias.symlink_to(path)
    with pytest.raises(ValueError, match="unsafe"):
        guard.identity(alias, {**expected, "path": str(alias)})


def test_claim_is_spent_on_completion_partial_and_concurrent_attempt(tmp_path: Path) -> None:
    output = tmp_path / "one"
    plan = {"implementation_commit": "a" * 40, "manifest_sha256": "b" * 64}
    checksum = entry.claim_bundle(output, plan, {"id": "synthetic"})
    assert checksum == digest(output / "claim.json")
    assert json.loads((output / "claim.json").read_text())["retry_authorized"] is False
    for _ in range(2):
        with pytest.raises(FileExistsError):
            entry.claim_bundle(output, plan, {"id": "different"})


def test_partial_claim_write_never_releases_attempt(tmp_path: Path, monkeypatch) -> None:
    original = entry.export_json

    def fail_claim(path, value):
        if path.name == "claim.json":
            raise OSError("synthetic interruption")
        return original(path, value)

    monkeypatch.setattr(entry, "export_json", fail_claim)
    output = tmp_path / "partial"
    plan = {"implementation_commit": "a" * 40, "manifest_sha256": "b" * 64}
    with pytest.raises(OSError):
        entry.claim_bundle(output, plan, {})
    assert (output / "plan.json.complete").is_file()
    with pytest.raises(FileExistsError):
        entry.claim_bundle(output, plan, {})


def test_failing_real_input_gate_is_after_claim_and_never_retried(tmp_path, monkeypatch) -> None:
    manifest = {"allowed_payload_inputs": {}, "immutable_joint_claim": {}, "runtime": {}}
    metadata = {"parents": {"synthetic": True}}
    publication = {
        "manifest": manifest,
        "metadata": metadata,
        "origin_main": "a" * 40,
        "source_sha256": {},
    }
    monkeypatch.setattr(guard, "publication", lambda pin: publication)
    monkeypatch.setattr(guard, "OUTPUT", tmp_path / "one")
    monkeypatch.setattr(entry, "digest", lambda path: "a" * 64)
    accesses = []

    def refuse(_manifest):
        assert (guard.OUTPUT / "claim.json.complete").exists()
        accesses.append("claimed_then_verified")
        raise ValueError("synthetic foreign checkpoint")

    monkeypatch.setattr(guard, "verify_inputs", refuse)

    @contextmanager
    def deadline():
        yield SimpleNamespace(checkpoint=lambda: None)

    @contextmanager
    def exclusive():
        yield SimpleNamespace(deadline=deadline, lease={}, threads=1)

    monkeypatch.setattr(runtime, "exclusive_runtime", exclusive)
    from research.bl_residual import f4_postmortem_watchdog

    monkeypatch.setattr(
        f4_postmortem_watchdog, "supervise", lambda worker, **_kwargs: worker(lambda: None)
    )
    with pytest.raises(RuntimeError, match="no retry"):
        entry.run("a" * 40)
    with pytest.raises(FileExistsError):
        entry.run("a" * 40)
    assert accesses == ["claimed_then_verified"]
    failure = json.loads((guard.OUTPUT / "failure.json").read_bytes())
    assert failure["failure_class"] == "ValueError" and failure["status"] == "INCOMPLETE"
    assert "foreign checkpoint" not in json.dumps(failure)


@pytest.mark.parametrize("parents", [[], [None], ["burned"], ["fresh", None]])
def test_exclusion_rejects_exposed_or_unresolved_original_ancestry(parents, monkeypatch) -> None:
    monkeypatch.setattr(guard, "read_manifest", lambda root: {})
    monkeypatch.setattr(
        guard, "provenance", lambda *args: {"parents": {"parent_worlds": ["burned"]}}
    )
    with pytest.raises(ValueError, match="ancestry"):
        guard.require_independent_ancestry(parents)
    guard.require_independent_ancestry(["independent-original-world"])


def synthetic_provenance(tmp_path: Path) -> dict:
    """Fake every ledger and payload identity; no real-data paths are opened."""
    names = {
        f"--d{dose}-{part}": {"path": f"/synthetic/d{dose}/{part}", "sha256": str(dose)}
        for dose in (256, 1024)
        for part in ("allocation", "checkpoint", "control", "table")
    }
    for option in ("--approval", "--archive-receipt", "--b8"):
        names[option] = {"path": f"/synthetic/{option}", "sha256": option}
    names["--d1024-table"] = names["--d256-table"].copy()
    for name in names:
        names[name]["size_bytes"] = 12
    packet = {
        "input_identities": {k: {n: r[n] for n in ("path", "sha256")} for k, r in names.items()},
        "lock_path": "/synthetic/original-claim.json",
        "lock": {"reserve": {"reserve_life_ids": [f"life-{i}" for i in range(64)]}},
    }
    census = {
        "counts": {"rows": 5936, "natural_rows": 560, "eligible_lives": 71, "parents": 29},
        "parent_worlds": [f"p{i:02}" for i in range(29)],
        "table_sha256": names["--d256-table"]["sha256"],
    }
    data = {
        f"{guard.RA_LEDGER}/command-packet.json": packet,
        f"{guard.RA_LEDGER}/funding-look.json": {
            "verdict": "STOP_FUNDING",
            "harm_screen_d1024_pass": False,
        },
        "research/bl_residual/ledger/F4_R_A_ARCHIVE_CENSUS_20261024.json": census,
    }
    for dose, date in ((256, "20261023"), (1024, "20261024")):
        r = names[f"--d{dose}-checkpoint"]
        data[f"research/bl_residual/ledger/F4_D{dose}_FIT_PHASE_{date}.json"] = {
            "units": [
                {
                    "path": r["path"],
                    "sha256": r["sha256"],
                    "size": 12,
                    "index": 0 if dose == 256 else 1,
                }
            ]
        }
    for name, value in data.items():
        export_json(tmp_path / name, value)
    excluded = tmp_path / "excluded.json"
    export_json(
        excluded,
        {
            "parent_worlds": census["parent_worlds"],
            "designation": "DEVELOPMENT_FOREVER",
            "ruling_sha256": guard.RULING_SHA,
        },
    )
    return {
        "allowed_payload_inputs": names,
        "immutable_joint_claim": {"path": packet["lock_path"]},
        "development_exclusion_ledger": "excluded.json",
        "allowed_metadata_provenance": {
            name: {
                "sha256": digest(tmp_path / name),
                "size_bytes": (tmp_path / name).stat().st_size,
            }
            for name in data
        },
    }


def test_provenance_reconciles_packet_checkpoint_census_and_all_parents(tmp_path) -> None:
    manifest = synthetic_provenance(tmp_path)
    assert len(guard.provenance(manifest, tmp_path)["parents"]["parent_worlds"]) == 29
    bad = copy.deepcopy(manifest)
    bad["allowed_payload_inputs"]["--d256-checkpoint"]["sha256"] = "foreign"
    with pytest.raises(ValueError, match="allowlist"):
        guard.provenance(bad, tmp_path)
    bad = copy.deepcopy(manifest)
    bad["allowed_payload_inputs"]["--d1024-checkpoint"]["size_bytes"] += 1
    with pytest.raises(ValueError, match="fit-phase"):
        guard.provenance(bad, tmp_path)


def test_publication_failure_precedes_any_manifest_or_payload_read(monkeypatch) -> None:
    calls = []

    def absent(*args, **kwargs):
        calls.append(args)
        raise ValueError("synthetic remote unavailable")

    monkeypatch.setattr(guard, "git", absent)
    monkeypatch.setattr(
        guard, "read_manifest", lambda *args: pytest.fail("read before publication")
    )
    with pytest.raises(ValueError, match="remote"):
        guard.publication("a" * 40)
    assert calls == [("fetch", "--no-tags", "origin", "main")]


def test_detached_identity_seals_are_mandatory_before_raw_input_access(
    tmp_path, monkeypatch
) -> None:
    raw = tmp_path / "raw.pth"
    raw.write_bytes(b"synthetic raw checkpoint bytes, never a model")
    checksum = tmp_path / "raw.pth.sha256"
    checksum.write_text(f"{digest(raw)}  raw.pth\n")
    done = tmp_path / "raw.pth.complete"
    done.write_text("incomplete\n")
    record = {
        "path": str(raw),
        "sha256": digest(raw),
        "size_bytes": raw.stat().st_size,
        "detached_identity_seal": {
            "sha256_path": "raw.pth.sha256",
            "complete_path": "raw.pth.complete",
        },
    }
    manifest = {
        "allowed_payload_inputs": {"--b8": record},
        "immutable_joint_claim": {},
        "allowed_external_metadata_receipts": {},
    }
    monkeypatch.setattr(guard, "ROOT", tmp_path)
    monkeypatch.setattr(
        guard, "identity", lambda *a, **k: pytest.fail("unsealed raw input accessed")
    )
    with pytest.raises(ValueError, match="detached"):
        guard.verify_inputs(manifest)


def test_shared_table_and_census_parent_substitution_fail_closed(tmp_path) -> None:
    manifest = synthetic_provenance(tmp_path)
    bad = copy.deepcopy(manifest)
    bad["allowed_payload_inputs"]["--d1024-table"]["path"] = "/synthetic/other-table"
    with pytest.raises(ValueError, match="allowlist"):
        guard.provenance(bad, tmp_path)
    excluded = tmp_path / "excluded.json"
    value = json.loads(excluded.read_bytes())
    value["parent_worlds"].pop()
    for suffix in ("", ".sha256", ".complete"):
        Path(str(excluded) + suffix).unlink()
    export_json(excluded, value)
    with pytest.raises(ValueError, match="census"):
        guard.provenance(manifest, tmp_path)


def test_original_claim_cannot_be_replaced_to_authorize_a_second_look() -> None:
    expected = {
        "schema": "synthetic/v1",
        "lock_root": "/synthetic",
        "reserve": {},
        "d256": {"sha": "old256"},
        "d1024": {"sha": "old1024"},
    }
    guard.validate_original_claim(expected, {"lock": expected})
    with pytest.raises(ValueError, match="original R-A claim"):
        guard.validate_original_claim({**expected, "d1024": {"sha": "new"}}, {"lock": expected})


def test_two_candidate_call_order_reads_only_same_pass_H_and_preserves_invalid_disagreement(
    monkeypatch,
) -> None:
    from research.bl_residual import heldout

    parents = [f"parent-{i:02}" for i in range(29)]
    manifest = {
        "allowed_payload_inputs": {
            **{
                f"--d{dose}-{part}": {"path": f"/synthetic/d{dose}/{part}", "sha256": str(dose)}
                for dose in (256, 1024)
                for part in ("checkpoint", "table", "allocation", "control")
            },
            **{
                f"--{part}": {"path": f"/synthetic/{part}"}
                for part in ("b8", "approval", "archive-receipt")
            },
        },
        "immutable_joint_claim": {"path": "/synthetic/original-claim"},
    }
    seen = []

    class HarmOnly(dict):
        def __getitem__(self, name):
            assert name == "harmful_override_count_natural", "unpermitted metric inspected"
            return 0

    def evaluate(candidate, baseline, table, **kwargs):
        dose = 256 if "d256" in str(candidate) else 1024
        seen.append(dose)
        assert kwargs["joint_claim_path"] == Path("/synthetic/original-claim")
        natural = torch.zeros(5936, dtype=torch.bool)
        natural[:560] = True
        masks = {name: torch.zeros_like(natural) for name in COUNT_MASKS.values()}
        masks["natural"] = natural
        masks["margin_bins"] = (natural, *(torch.zeros_like(natural) for _ in range(3)))
        kwargs["diagnostics_observer"](masks, tuple(parents[i % 29] for i in range(5936)))
        return {
            "archive_eligible_life_ids": [None] * 71,
            "reserve_life_ids": [None] * 64,
            "checkpoint_sha256": str(dose),
            "metrics": HarmOnly(),
        }

    monkeypatch.setattr(heldout, "evaluate", evaluate)
    doses, checks = entry.compute_doses(
        manifest,
        {"parents": {"parent_worlds": parents}, "harm_pass": False},
        checkpoint=lambda: None,
    )
    assert seen == [256, 1024]
    assert all(row["totals"]["H"] == 0 for row in doses)
    assert checks["retained_d1024_harm_boolean_agrees"] is False
    assert all(v is True for k, v in checks.items() if k != "retained_d1024_harm_boolean_agrees")
