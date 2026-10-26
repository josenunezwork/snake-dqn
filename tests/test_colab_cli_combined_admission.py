"""Offline combined F5/D1024 proof reaches the real persisted registry consumer."""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from research.colab_cli import acceptance, admission, supervisor
from tests.test_colab_cli_admission import passing_result
from tests.test_colab_cli_f5_timing_transfer import _authority

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools/ops"))
import qualification  # noqa: E402
import requalify  # noqa: E402

# Reuse the base artifact/cold-kernel evidence fixture; all scoped proof below is real.
_base_result = passing_result


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


@pytest.fixture
def combined(tmp_path: Path, _base_result: dict) -> tuple[Path, Path, dict]:
    repo = tmp_path / "broker"
    shutil.copytree(
        REPO / "research/colab_cli",
        repo / "research/colab_cli",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Combined Test")
    _git(repo, "config", "user.email", "combined@example.invalid")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "current transport")
    commit, tree = _git(repo, "rev-parse", "HEAD"), _git(repo, "rev-parse", "HEAD^{tree}")
    authority = _authority(commit, tree)
    public = next(
        row
        for row in acceptance.d1024_synthetic_inventory()["inputs"]
        if row["transport"] == "https"
    )
    authority["inputs"] = [
        dict(public) if row["transport"] == "https" else row for row in authority["inputs"]
    ]
    unsigned = dict(authority)
    unsigned.pop("sha256")
    authority["sha256"] = acceptance.canonical_sha256(unsigned)
    authority_path = tmp_path / "authority.json"
    authority_path.write_text(json.dumps(authority))
    account = tmp_path / "accounts.json"
    account.write_text("{}")
    saved = requalify.plan(
        repo,
        tmp_path / "state",
        account,
        "/never/execute/colab",
        d1024_inventory=True,
        f5_timing_authority=authority_path,
    )
    plan = copy.deepcopy(json.loads(saved.read_text())["plan"])
    assert len(plan["gates"]) == 16
    plan["status"] = "PASS"
    base_gates = {g["id"]: g for g in _base_result["final_plan"]["gates"]}
    for gate in plan["gates"]:
        gate["status"] = "OBSERVED" if gate["required"] else "NOT_REQUIRED"
        if "evidence_record" in base_gates.get(gate["id"], {}):
            gate["evidence_record"] = base_gates[gate["id"]]["evidence_record"]
    plan["execution"] = []
    for job in plan["jobs"]:
        manifest_path = Path(job["manifest"])
        manifest = json.loads(manifest_path.read_text())
        output = Path(manifest["output_dir"])
        output.mkdir()
        (output / "manifest.json").write_bytes(manifest_path.read_bytes())
        transfers = [
            {
                "direction": "upload",
                "status": "verified",
                "remote": f"/content/run/inputs/{row['dest']}",
                "size_bytes": Path(row["path"]).stat().st_size,
                "sha256": row["sha256"],
            }
            for row in manifest["inputs"]
            if "path" in row
        ]
        receipt = {
            "stop_verified": True,
            "cleanup_error": None,
            "commit": commit,
            "job_id": manifest["job_id"],
            "manifest_sha256": acceptance.sha256_file(manifest_path),
            "transfers": transfers,
        }
        receipt_path = output / "receipt.json"
        receipt_path.write_text(json.dumps(receipt))
        plan["execution"].append(
            {
                "job": job["name"],
                "status": "PASS",
                "receipt": str(receipt_path),
                "receipt_sha256": acceptance.sha256_file(receipt_path),
            }
        )
        if job["name"] != "l4":
            continue
        (output / "public-input-preflight.json").write_text('{"status":"PASS"}')
        durations = {row["dest"]: 1.0 for row in manifest["inputs"] if "path" in row}
        (output / "upload-measurement.json").write_text(
            json.dumps(
                {
                    "elapsed_seconds": 100.0,
                    "transfer_seconds": 70.0,
                    "source_delivery_seconds": 16.0,
                    "staging_completed": True,
                    "input_transfer_seconds": durations,
                }
            )
        )
        for gate_id, make_evidence in (
            ("d1024-synthetic-full-inventory-staging", acceptance.d1024_synthetic_staging_evidence),
            (
                "f5-timing-synthetic-full-inventory-staging",
                acceptance.f5_timing_synthetic_staging_evidence,
            ),
        ):
            next(g for g in plan["gates"] if g["id"] == gate_id)["evidence_record"] = make_evidence(
                plan, job, receipt, output
            )
    result = {
        **_base_result,
        "source_manifest": plan["source_manifest"],
        "final_plan": plan,
        "qualified_commit": commit,
        "qualified_tree_hash": plan["tree_hash"],
        "d1024_synthetic_inventory_sha256": plan["d1024_synthetic_inventory"]["sha256"],
        "f5_timing_synthetic_inventory_sha256": authority["sha256"],
        "f5_timing_transfer_exception": plan["f5_timing_transfer_exception"],
    }
    return repo, saved, result


def _publish(combined: tuple, monkeypatch: pytest.MonkeyPatch) -> Path:
    repo, saved, result = combined
    # Only provider execution is synthetic. Keep real plan checks, final admission,
    # publication and registry consumption. Simulate outputs appearing after dispatch.
    outputs = []
    for row in result["final_plan"]["jobs"]:
        output = Path(json.loads(Path(row["manifest"]).read_text())["output_dir"])
        parked = output.with_name(output.name + "-parked")
        output.rename(parked)
        outputs.append((parked, output))

    def execute(plan: dict, **kwargs: object) -> int:
        for parked, output in outputs:
            parked.rename(output)
        saved.with_name("acceptance-result.json").write_text(json.dumps(result))
        return 0

    monkeypatch.setattr(acceptance, "execute", execute)
    registry = saved.parent / "test-qualified.json"
    requalify.execute(saved, registry, acceptance.REVIEW_TOKEN, "/never/oauth")
    assert qualification.check(repo, registry) == result["qualified_tree_hash"]
    acceptance.verify_qualified_transport_record(repo, registry)
    return registry


def test_full_combined_pass_publishes_consumable_qualification(
    combined: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _publish(combined, monkeypatch).is_file()


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "unobserved",
        "extra",
        "duplicate",
        "evidence",
        "inventory",
        "authority",
        "receipt",
        "measurement",
        "preflight",
        "upload",
    ],
)
def test_combined_publication_refuses_changed_f5_proof(
    combined: tuple, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    repo, saved, result = combined
    plan = result["final_plan"]
    gate = next(g for g in plan["gates"] if g["id"] == "f5-timing-synthetic-full-inventory-staging")
    proof = gate["evidence_record"]
    if fault == "missing":
        plan["gates"].remove(gate)
    elif fault == "unobserved":
        gate["status"] = "INCOMPLETE"
    elif fault in {"extra", "duplicate"}:
        plan["gates"].append(dict(gate, id="arbitrary-extra" if fault == "extra" else gate["id"]))
    elif fault == "evidence":
        proof["inventory_sha256"] = "0" * 64
    elif fault == "inventory":
        plan["f5_timing_synthetic_inventory"]["namespace"] = "foreign"
    elif fault == "authority":
        result["f5_timing_transfer_exception"] = {}
    elif fault == "receipt":
        Path(proof["receipt"]["path"]).write_text("{}")
    elif fault == "measurement":
        Path(proof["upload_measurement"]["path"]).write_text("{}")
    elif fault == "preflight":
        Path(proof["public_input_preflight"]).write_text("changed")
    elif fault == "upload":
        proof["inputs"][0]["status"] = "unverified"
    # A separately persisted hypothetical PASS must be refused by the downstream
    # registry consumer too, even when its outer file hash has been refreshed.
    result_path = saved.parent / "hypothetical-result.json"
    result_path.write_text(json.dumps(result))
    registry = saved.parent / "hypothetical-registry.json"
    registry.write_text(
        json.dumps(
            {
                "schema": acceptance.QUALIFIED_RECORD_SCHEMA,
                "qualified_commit": result["qualified_commit"],
                "tree_hash": result["qualified_tree_hash"],
                "acceptance_result": str(result_path),
                "acceptance_result_sha256": acceptance.sha256_file(result_path),
            }
        )
    )
    with pytest.raises((supervisor.Refused, ValueError)):
        acceptance.verify_qualified_transport_record(repo, registry)
    with pytest.raises((supervisor.Refused, ValueError)):
        _publish(combined, monkeypatch)
    assert not (saved.parent / "test-qualified.json").exists()


def test_f5_only_admission_requires_its_fifteenth_gate(combined: tuple) -> None:
    repo, _, result = combined
    plan = result["final_plan"]
    plan.pop("d1024_synthetic_inventory")
    plan["gates"] = [
        g for g in plan["gates"] if g["id"] != "d1024-synthetic-full-inventory-staging"
    ]
    assert len(plan["gates"]) == 15
    assert admission.require_final_only(result, repo) == result["final_artifact_transfer"]


def test_saved_review_cannot_omit_f5_authority(
    combined: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, saved, _ = combined
    review = json.loads(saved.read_text())
    review["f5_timing_authority"] = None
    saved.write_text(json.dumps(review))
    monkeypatch.setattr(
        acceptance, "execute", lambda *a, **kw: pytest.fail("provider execute reached")
    )
    with pytest.raises(ValueError, match="selection differs"):
        requalify.execute(
            saved, saved.parent / "qualified.json", acceptance.REVIEW_TOKEN, "/never/oauth"
        )
    assert not saved.with_name("ops-execute-started.json").exists()


@pytest.mark.parametrize(
    "d1024,f5,count", [(False, False, 14), (True, False, 15), (False, True, 15), (True, True, 16)]
)
def test_gate_contract_has_exact_conditional_inventory(d1024: bool, f5: bool, count: int) -> None:
    gates = acceptance.plan_review_gates(d1024_inventory=d1024, f5_timing_inventory=f5)
    assert len(gates) == len({g["id"] for g in gates}) == count
