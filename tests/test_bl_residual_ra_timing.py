"""Synthetic publication and timing guards for the single R-A claim."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from research.bl_ppo.exports import digest, export_json
from research.bl_residual import f2, github_authority, ra_guard
from research.bl_residual.protocol import canonical_sha
from tests.test_bl_residual_f2 import git_fixture, seal_fixture


def _commit(
    root: Path,
    message: str,
    epoch: int,
    *,
    allow_empty: bool = False,
    author_epoch: int | None = None,
) -> None:
    env = os.environ | {
        "GIT_AUTHOR_DATE": datetime.fromtimestamp(
            epoch if author_epoch is None else author_epoch, timezone.utc
        ).isoformat(),
        "GIT_COMMITTER_DATE": datetime.fromtimestamp(epoch, timezone.utc).isoformat(),
    }
    args = ["git", "commit", "-qm", message]
    if allow_empty:
        args.insert(2, "--allow-empty")
    subprocess.run(args, cwd=root, check=True, env=env, capture_output=True)


def _published_binding(
    tmp_path,
    monkeypatch,
    epoch: int,
    *,
    published_epoch: int | None = None,
    author_epoch: int | None = None,
):
    root, receipt = git_fixture(tmp_path, monkeypatch)
    seal_fixture(root, receipt)
    monkeypatch.setattr(f2, "ROOT", root)
    monkeypatch.setattr(f2, "RESULT_LEDGER", "research/bl_residual/ledger/f2-result-20261024.json")
    f2.stage_publication(root=root)
    subprocess.run(["git", "add", "."], cwd=root, check=True, capture_output=True)
    _commit(root, "synthetic result publication", epoch, author_epoch=author_epoch)
    subprocess.run(
        ["git", "update-ref", "refs/remotes/origin/main", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    binding = github_authority.published_receipt_binding(receipt, root=root)
    monkeypatch.setattr(ra_guard, "f2", f2)
    commit = binding["publication_ref"]
    before = f2._git(root, "rev-parse", commit + "^").decode().strip()
    epoch = epoch if published_epoch is None else published_epoch
    timestamp = datetime.fromtimestamp(epoch, timezone.utc).isoformat().replace("+00:00", "Z")

    def request(endpoint):
        if endpoint.endswith("git/ref/heads/main"):
            tip = f2._git(root, "rev-parse", "origin/main").decode().strip()
            return (
                {"ref": github_authority.REF, "object": {"type": "commit", "sha": tip}},
                {},
                {"epoch": epoch + 1000},
            )
        return (
            [
                {
                    "id": 1,
                    "before": before,
                    "after": commit,
                    "ref": github_authority.REF,
                    "activity_type": "push",
                    "timestamp": timestamp,
                }
            ],
            {},
            {"epoch": epoch + 1000},
        )

    monkeypatch.setattr(github_authority, "_request", request)
    github_authority.capture_publication(binding)
    return root, receipt, binding


def _authoritative(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    epoch: float | None,
    *,
    overrides: dict | None = None,
) -> Path | None:
    control_path = tmp_path / "d4096-control.json"
    monkeypatch.setattr(ra_guard, "D4096_CONTROL", control_path)
    if epoch is None:
        return None
    control = {
        "schema": "bl-labels-f0-control/v1",
        "namespace": "d4096",
        "allocation": [{"allocation_index": 0}],
        "source_commit": ra_guard.D4096_SCIENCE,
        "mac_gate": {"stage": "d4096", "status": "HOLD", "source_commit": ra_guard.D4096_SCIENCE},
    }
    export_json(control_path, control)
    state = control_path.parent / "d4096-state-mac-01"
    admitted = {
        **control,
        "mac_gate": {
            "stage": "d4096",
            "status": "APPROVED",
            "source_commit": ra_guard.D4096_SCIENCE,
            "deadline_seconds": 468000,
            "package_root": str(tmp_path.resolve()),
            "state_dir": str(state.resolve()),
            "original_control_path": str(control_path.resolve()),
            "original_control_sha256": digest(control_path),
            "collector_source_commit": ra_guard.OLD_COLLECTOR,
            "bindings": {"speed_equivalence": {"status": "FAIL_OLD_FALLBACK"}},
        },
    }
    export_json(control_path.with_name("d4096-admitted.json"), admitted)
    assert canonical_sha(admitted) != canonical_sha(control)
    path = state / "attempt-01.start.json"
    export_json(
        path,
        {
            "schema": "bl-labels-f0-mac-attempt-start/v1",
            "funding_state": "PENDING_SEALED",
            "judgment_sha256": f2.JUDGMENT_SHA,
            "started_epoch": epoch,
            "namespace": "d4096",
            "allocation_sha256": canonical_sha(admitted["allocation"]),
            "control_sha256": canonical_sha(admitted),
            "source_commit": admitted["source_commit"],
            **(overrides or {}),
        },
    )
    return path


def test_publication_evidence_binds_server_push_time_and_survives_unrelated_commit(
    tmp_path, monkeypatch
):
    epoch = int(ra_guard.CUTOFF) - 100
    root, receipt, binding = _published_binding(tmp_path, monkeypatch, epoch)
    evidence = ra_guard.publication_evidence(binding)
    assert evidence["commit"] == binding["publication_ref"]
    assert evidence["published_epoch"] == epoch
    assert evidence["ref"] == "refs/heads/main"
    subprocess.run(["git", "commit", "--allow-empty", "-qm", "unrelated"], cwd=root, check=True)
    subprocess.run(["git", "update-ref", "refs/remotes/origin/main", "HEAD"], cwd=root, check=True)
    assert ra_guard.publication_evidence(binding) == evidence


def test_replaced_result_publication_is_rejected(tmp_path, monkeypatch):
    _, receipt, binding = _published_binding(tmp_path, monkeypatch, int(ra_guard.CUTOFF) - 100)
    root = f2.ROOT
    result = root / f2.RESULT_LEDGER
    result.write_text(result.read_text().replace(binding["sha256"], "0" * 64))
    assert ra_guard.publication_evidence(binding) is None


@pytest.mark.parametrize(
    "publication, expected",
    [(ra_guard.CUTOFF - 1, True), (ra_guard.CUTOFF, False), (ra_guard.CUTOFF + 1, False)],
)
def test_strict_publication_and_cutoff_boundaries(tmp_path, monkeypatch, publication, expected):
    _, _, binding = _published_binding(tmp_path, monkeypatch, int(publication))
    _authoritative(tmp_path, monkeypatch, None)
    timing = ra_guard.timing_identity(None, ra_guard.publication_evidence(binding))
    assert ra_guard.timing_eligible(timing, at=ra_guard.CUTOFF + 60) is expected


def test_publication_must_precede_r_a(tmp_path, monkeypatch):
    for offset, expected in [(-1, False), (0, False), (1, True)]:
        case = tmp_path / str(offset)
        case.mkdir()
        root, _, binding = _published_binding(case, monkeypatch, int(ra_guard.CUTOFF - 100))
        _authoritative(case, monkeypatch, None)
        timing = ra_guard.timing_identity(None, ra_guard.publication_evidence(binding))
        assert ra_guard.timing_eligible(timing, at=int(ra_guard.CUTOFF - 100) + offset) is expected


@pytest.mark.parametrize(
    ("started", "branch"),
    [
        (ra_guard.START_DEADLINE - 1, "WAIT_FOR_F2_OR_CUTOFF"),
        (ra_guard.START_DEADLINE, "WAIT_FOR_F2_OR_CUTOFF"),
        (ra_guard.START_DEADLINE + 1, "READY_WITHOUT_WAIT"),
    ],
)
def test_start_boundary_and_wait_hold_then_strikes_at_cutoff(
    tmp_path, monkeypatch, started, branch
):
    start = _authoritative(tmp_path, monkeypatch, started)
    monkeypatch.setattr(ra_guard, "now_epoch", lambda: ra_guard.CUTOFF + 60)
    timing = ra_guard.timing_identity(start, None)
    assert timing["branch"] == branch
    if branch.startswith("WAIT"):
        with pytest.raises(ValueError, match="waits"):
            ra_guard.timing_eligible(timing, at=ra_guard.CUTOFF - 1)
        assert ra_guard.timing_eligible(timing, at=ra_guard.CUTOFF) is False


@pytest.mark.parametrize(
    "field",
    [
        "started_epoch",
        "namespace",
        "allocation_sha256",
        "control_sha256",
        "source_commit",
        "funding_state",
        "judgment_sha256",
    ],
)
def test_invalid_or_future_start_metadata_refused(tmp_path, monkeypatch, field):
    monkeypatch.setattr(ra_guard, "now_epoch", lambda: ra_guard.CUTOFF + 60)
    invalid = ra_guard.CUTOFF + 10_000_000 if field == "started_epoch" else "foreign"
    start = _authoritative(
        tmp_path, monkeypatch, ra_guard.START_DEADLINE, overrides={field: invalid}
    )
    with pytest.raises(ValueError, match="sealed D4096"):
        ra_guard.timing_identity(start, None)


def test_omission_and_later_resume_cannot_hide_early_start(tmp_path, monkeypatch):
    monkeypatch.setattr(ra_guard, "now_epoch", lambda: ra_guard.CUTOFF + 60)
    start = _authoritative(tmp_path, monkeypatch, ra_guard.START_DEADLINE)
    before = ra_guard.timing_identity(None, None)
    assert before["branch"] == "WAIT_FOR_F2_OR_CUTOFF"
    assert before["d4096_start"]["path"] == str(start)
    late = start.with_name("attempt-02.start.json")
    export_json(
        late, {**json.loads(start.read_text()), "started_epoch": ra_guard.START_DEADLINE + 1}
    )
    after = ra_guard.timing_identity(None, None)
    assert before != after  # complete authoritative start set is sealed
    assert after["d4096_start"] == before["d4096_start"]
    with pytest.raises(ValueError, match="earliest authoritative"):
        ra_guard.timing_identity(late, None)


def test_new_authority_invalidates_a_prepared_no_start_branch(tmp_path, monkeypatch):
    monkeypatch.setattr(ra_guard, "now_epoch", lambda: ra_guard.CUTOFF + 60)
    _authoritative(tmp_path, monkeypatch, None)
    before = ra_guard.timing_identity(None, None)
    assert before["branch"] == "READY_WITHOUT_WAIT"
    _authoritative(tmp_path, monkeypatch, ra_guard.START_DEADLINE)
    after = ra_guard.timing_identity(None, None)
    assert after != before and after["branch"] == "WAIT_FOR_F2_OR_CUTOFF"


@pytest.mark.parametrize(
    "author_offset,committer_offset,push_offset,eligible",
    [(-100, -100, 60, False), (60, -100, 60, False), (-100, 60, 60, False), (-100, -100, -1, True)],
)
def test_delayed_push_and_independent_git_dates_cannot_authorize_f2(
    tmp_path, monkeypatch, author_offset, committer_offset, push_offset, eligible
):
    _, _, binding = _published_binding(
        tmp_path,
        monkeypatch,
        int(ra_guard.CUTOFF + committer_offset),
        author_epoch=int(ra_guard.CUTOFF + author_offset),
        published_epoch=int(ra_guard.CUTOFF + push_offset),
    )
    _authoritative(tmp_path, monkeypatch, None)
    timing = ra_guard.timing_identity(None, ra_guard.publication_evidence(binding))
    assert ra_guard.timing_eligible(timing, at=ra_guard.CUTOFF + 120) is eligible


def test_metadata_adapter_and_evidence_never_parse_sealed_result(tmp_path, monkeypatch):
    _, receipt, binding = _published_binding(tmp_path, monkeypatch, int(ra_guard.CUTOFF - 100))
    sealed = receipt.read_bytes()
    loads = json.loads

    def guarded(payload, *args, **kwargs):
        if payload == sealed or payload == sealed.decode():
            pytest.fail("sealed result parsed by metadata/evidence verification")
        return loads(payload, *args, **kwargs)

    monkeypatch.setattr(json, "loads", guarded)
    assert ra_guard.receipt_binding(receipt) == binding
    assert ra_guard.publication_evidence(binding)["receipt_sha256"] == binding["sha256"]
