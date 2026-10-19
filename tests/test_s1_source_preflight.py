"""Exercise pinned packaging and the refusal boundary without any account access."""

import argparse
import json
from pathlib import Path

import pytest

from research.s1_mature_20261018 import admission, run
from tests.test_colab_cli_source import _manifest, _run
from tests.test_colab_cli_source import repo as _source_repo


@pytest.fixture
def source_repo(tmp_path: Path) -> Path:
    return _source_repo.__wrapped__(tmp_path)


def test_packaging_builds_every_manifest_with_supervisor_builder(source_repo: Path) -> None:
    rows = [{**_manifest(source_repo), "job_id": f"job-{i}"} for i in range(10)]
    result = admission.packaging_preflight(source_repo, rows)
    assert result["status"] == "PASS"
    assert len(result["jobs"]) == 10
    assert len({entry["inventory_sha256"] for entry in result["jobs"].values()}) == 1
    assert all(entry["files"] > 0 for entry in result["jobs"].values())


@pytest.mark.parametrize("linked_tree", ["src", "research/study", "research/colab_cli"])
def test_committed_symlink_refused_before_any_ledger_or_account_access(
    source_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, linked_tree: str
) -> None:
    (source_repo / linked_tree / "bad-link").symlink_to("absent")
    _run(source_repo, "add", ".")
    _run(source_repo, "commit", "-qm", "tracked symlink")
    root = tmp_path / "production"
    manifests = []
    for i in range(10):
        path = tmp_path / f"job-{i}.json"
        admission.write_once(
            path,
            {
                **_manifest(source_repo),
                "job_id": f"job-{i}",
                "output_dir": str(root / "training" / str(i)),
            },
        )
        manifests.append(
            {"job_id": f"job-{i}", "path": str(path), "sha256": admission.sha256(path)}
        )
    plan = {"manifests": manifests}
    monkeypatch.setattr(run, "_verify_plan", lambda *_: plan)

    def forbidden(*args, **kwargs):
        pytest.fail("ledger/account boundary crossed before source refusal")

    monkeypatch.setattr(run.ledger, "Ledger", forbidden)
    monkeypatch.setattr(run, "_snapshot", forbidden)
    args = argparse.Namespace(
        plan=tmp_path / "plan.json",
        repo=source_repo,
        artifact_root=root,
        supervisor=source_repo / "research/colab_cli/launch.sh",
    )
    with pytest.raises(admission.AdmissionRefused, match="symlink prohibited"):
        run.dispatch(args)


def _continuation(tmp_path: Path) -> tuple[argparse.Namespace, dict, list]:
    paths, old_paths = [], []
    for i in range(10):
        row = {
            "job_id": f"job-{i}",
            "commit": "a" * 40,
            "output_dir": str(tmp_path / "old-output" / str(i)),
            "env": {"S1_SEED": str(i)},
        }
        old = tmp_path / f"old-{i}.json"
        admission.write_once(old, row)
        old_paths.append(
            {"job_id": row["job_id"], "path": str(old), "sha256": admission.sha256(old)}
        )
        current = tmp_path / f"new-{i}.json"
        admission.write_once(
            current,
            {**row, "commit": "b" * 40, "output_dir": str(tmp_path / "new-output" / str(i))},
        )
        paths.append((row["job_id"], current))
    mac = tmp_path / "mac.json"
    admission.write_once(mac, {"kind": "mac"})
    plan = {
        "schema": admission.SCHEMA,
        "status": "PREPARED_NOT_DISPATCHED",
        "liabilities": {"job-0": {}},
        "mac_manifest": {"path": str(mac), "sha256": admission.sha256(mac)},
    }
    old_plan = tmp_path / "previous-plan.json"
    admission.write_once(old_plan, {**plan, "manifests": old_paths})
    ledger = tmp_path / "ledger.json"
    admission.write_once(
        ledger,
        {
            "outstanding": {},
            "planned": plan["liabilities"],
            "settled": {"cli-acceptance": {}},
            "events": [
                {"kind": "preallocation_refusal", "job_id": "job-0", "manifest": old_paths[0]}
            ],
        },
    )
    return argparse.Namespace(previous_plan=old_plan, ledger=ledger), plan, paths


def test_source_repair_continues_same_ledger_with_identical_science(tmp_path: Path) -> None:
    args, plan, paths = _continuation(tmp_path)
    run._preallocation_continuation(args, plan, paths)


@pytest.mark.parametrize(
    "changed", ["seed", "liability", "outstanding", "no_event", "old_manifest", "existing_output"]
)
def test_continuation_refuses_changed_science_or_unreconciled_state(
    tmp_path: Path, changed: str
) -> None:
    args, plan, paths = _continuation(tmp_path)
    state = json.loads(args.ledger.read_text())
    if changed == "seed":
        row = json.loads(paths[0][1].read_text())
        row["env"]["S1_SEED"] = "12345"
        paths[0][1].write_text(json.dumps(row))
    elif changed == "liability":
        plan["liabilities"] = {"different": {}}
    elif changed == "outstanding":
        state["outstanding"] = {"job-0": {}}
    elif changed == "no_event":
        state["events"] = []
    elif changed == "old_manifest":
        state["events"][0]["manifest"]["sha256"] = "0" * 64
    elif changed == "existing_output":
        Path(json.loads(paths[0][1].read_text())["output_dir"]).mkdir(parents=True)
    args.ledger.write_text(json.dumps(state))
    with pytest.raises(admission.AdmissionRefused):
        run._preallocation_continuation(args, plan, paths)
