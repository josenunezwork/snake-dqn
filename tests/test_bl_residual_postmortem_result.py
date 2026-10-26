"""Synthetic-only tests for outcome-blind F4 result sealing."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from research.bl_residual import f4_postmortem_result as result

PIN = "a" * 40


def _sealed(root: Path, name: str, value: object) -> None:
    payload = json.dumps(value, sort_keys=True).encode() + b"\n"
    _sealed_bytes(root, name, payload)


def _sealed_bytes(root: Path, name: str, payload: bytes) -> None:
    path = root / name
    path.write_bytes(payload)
    sha = hashlib.sha256(payload).hexdigest()
    (root / f"{name}.sha256").write_text(f"{sha}  {name}\n")
    (root / f"{name}.complete").write_text("complete\n")


def _bundle(tmp_path: Path) -> tuple[Path, Path]:
    package = tmp_path / "package"
    package.mkdir()
    manifest = package / "manifest.json"
    inputs = {
        f"--d{dose}-checkpoint": {
            "path": f"/synthetic/d{dose}.pth",
            "size_bytes": 12,
            "sha256": str(index) * 64,
        }
        for index, dose in enumerate((256, 1024), start=2)
    }
    joint = {"path": "/synthetic/original-claim.json", "size_bytes": 12, "sha256": "b" * 64}
    manifest.write_text(
        json.dumps(
            {"allowed_payload_inputs": inputs, "immutable_joint_claim": joint, "runtime": {}}
        )
        + "\n"
    )
    development = {
        "schema": "bl-residual-development-parents/v1",
        "parent_worlds": [f"p{i:02}" for i in range(29)],
    }
    (package / "development-parents.json").write_text(json.dumps(development) + "\n")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    _sealed(
        bundle,
        "plan.json",
        {
            "schema": "bl-residual-f4-postmortem-plan/v1",
            "implementation_commit": PIN,
            "ruling_sha256": result.guard.RULING_SHA,
            "output_root": str(result.guard.OUTPUT),
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "inputs": inputs,
            "original_joint_claim": joint,
            "runtime": {},
            "source_sha256": {"synthetic.py": "f" * 64},
        },
    )
    plan_sha = hashlib.sha256((bundle / "plan.json").read_bytes()).hexdigest()
    _sealed(
        bundle,
        "claim.json",
        {
            "schema": "bl-residual-f4-postmortem-claim/v1",
            "implementation_commit": PIN,
            "ruling_sha256": result.guard.RULING_SHA,
            "plan_sha256": plan_sha,
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "retry_authorized": False,
            "claimed_at": "2026-10-25T00:00:00+00:00",
        },
    )
    _sealed(bundle, "development-parents.json", development)
    return bundle, package


def test_verify_seal_accepts_malformed_outcome_bytes_without_parsing_them(tmp_path: Path) -> None:
    bundle, package = _bundle(tmp_path)
    # It is deliberately invalid JSON: pre-publication sealing must only hash it.
    path = bundle / "diagnostic.json"
    path.write_bytes(b"not-json-outcome\n")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    (bundle / "diagnostic.json.sha256").write_text(f"{sha}  diagnostic.json\n")
    (bundle / "diagnostic.json.complete").write_text("complete\n")

    sealed = result.verify_seal(bundle, publication_commit=PIN, package_root=package)
    assert sealed["outcomes"] == ["diagnostic.json"]
    assert sealed["hashes"]["diagnostic.json"] == sha


def test_verify_seal_rejects_unexpected_file_or_semantic_development_substitution(
    tmp_path: Path,
) -> None:
    bundle, package = _bundle(tmp_path)
    _sealed(bundle, "failure.json", {"opaque": True})
    (bundle / "surprise.bin").write_bytes(b"x")
    with pytest.raises(ValueError, match="unexpected bundle filename"):
        result.verify_seal(bundle, publication_commit=PIN, package_root=package)
    (bundle / "surprise.bin").unlink()
    _sealed(bundle, "development-parents.json", {"schema": "foreign", "parent_worlds": []})
    with pytest.raises(ValueError, match="development exclusion"):
        result.verify_seal(bundle, publication_commit=PIN, package_root=package)


def test_verify_seal_rejects_wrong_claim_plan_hash_and_missing_terminal_receipt(
    tmp_path: Path,
) -> None:
    bundle, package = _bundle(tmp_path)
    with pytest.raises(ValueError, match="terminal"):
        result.verify_seal(bundle, publication_commit=PIN, package_root=package)
    _sealed(bundle, "failure.json", {"opaque": True})
    _sealed(
        bundle,
        "claim.json",
        {
            "schema": "bl-residual-f4-postmortem-claim/v1",
            "implementation_commit": PIN,
            "ruling_sha256": result.guard.RULING_SHA,
            "plan_sha256": "0" * 64,
            "retry_authorized": False,
        },
    )
    with pytest.raises(ValueError, match="plan/claim"):
        result.verify_seal(bundle, publication_commit=PIN, package_root=package)


def _diagnostic(bundle: Path) -> dict:
    from research.bl_residual.f4_postmortem_counts import BOUNDS, COUNT_MASKS

    plan = json.loads((bundle / "plan.json").read_bytes())
    doses = []
    parents = json.loads((bundle / "development-parents.json").read_bytes())["parent_worlds"]
    for dose in (256, 1024):
        counts = {k: 0 for k in COUNT_MASKS}
        counts["N_natural"] = 560
        if dose == 1024:
            for k in ("N_changed", "U_selected_only", "U_unknown", "H"):
                counts[k] = 1

        def group(values):
            return {
                "totals": copy.deepcopy(values),
                "margin_bins": [
                    {
                        "lower": lo,
                        "upper": hi,
                        "counts": copy.deepcopy(values) if i == 0 else {k: 0 for k in COUNT_MASKS},
                    }
                    for i, (lo, hi) in enumerate(BOUNDS)
                ],
            }

        doses.append(
            {
                "dose": dose,
                "checkpoint_sha256": plan["inputs"][f"--d{dose}-checkpoint"]["sha256"],
                **group(counts),
                "parents": [
                    {"parent_world": p, **group(counts if i == 0 else {k: 0 for k in COUNT_MASKS})}
                    for i, p in enumerate(parents)
                ],
            }
        )
    checks = {
        f"d{dose}_{check}": True
        for dose in (256, 1024)
        for check in ("census", "bins_parents_partitions_denominators", "same_pass_existing_H")
    }
    checks["retained_d1024_harm_boolean_agrees"] = True
    all_inputs = {**plan["inputs"], "--original-joint-claim": plan["original_joint_claim"]}
    return {
        "schema": "bl-residual-f4-harm-postmortem/v1",
        "ruling_sha256": result.guard.RULING_SHA,
        "implementation_commit": PIN,
        "source_sha256": plan["source_sha256"],
        "inputs": {
            k: {"path": v["path"], "size": v["size_bytes"], "sha256": v["sha256"]}
            for k, v in all_inputs.items()
        },
        "claim_sha256": hashlib.sha256((bundle / "claim.json").read_bytes()).hexdigest(),
        "started_at": "2026-10-25T00:00:00+00:00",
        "finished_at": "2026-10-25T00:00:01+00:00",
        "platform": "macOS-synthetic",
        "nice": 15,
        "threads": 1,
        "wall_seconds": 1.0,
        "status": "COMPLETE",
        "purpose": "DEVELOPMENT_ONLY",
        "f4_disposition": "STOP_FUNDING",
        "census": {
            "total_rows": 5936,
            "natural_rows": 560,
            "eligible_lives": 71,
            "reserve_lives": 64,
            "parents": 29,
        },
        "doses": doses,
        "checks": checks,
    }


def _published(tmp_path, monkeypatch, mutate=lambda value: None, outcome="diagnostic"):
    bundle, package = _bundle(tmp_path)
    if outcome == "diagnostic":
        terminal = _diagnostic(bundle)
        mutate(terminal)
        terminal_name = "diagnostic.json"
    elif outcome == "failure":
        terminal = {"opaque": True}
        terminal_name = "failure.json"
    else:
        raise ValueError("synthetic test outcome must be diagnostic or failure")
    _sealed(bundle, terminal_name, terminal)
    root = tmp_path / "repo"
    destination = root / result.RESULT_RELATIVE
    shutil.copytree(bundle, destination)
    names = (*result.BASE_NAMES, terminal_name)
    binding = {
        "schema": "bl-residual-f4-postmortem-result-binding/v1",
        "implementation_commit": PIN,
        "ruling_sha256": result.guard.RULING_SHA,
        "outcomes": [terminal_name],
        "bundle": {
            name: {
                "sha256": hashlib.sha256((destination / name).read_bytes()).hexdigest(),
                "size": (destination / name).stat().st_size,
            }
            for name in names
        },
    }
    _sealed(destination, "result-binding.json", binding)
    original = {p.name: p.read_bytes() for p in destination.iterdir()}
    monkeypatch.setattr(result, "_git", lambda *args: "")
    monkeypatch.setattr(result, "_origin_bytes", lambda root, path: original[path.name])
    monkeypatch.setattr(result.guard, "ROOT", package.parent)
    monkeypatch.setattr(result.guard, "PACKAGE", package.name)
    return root, destination, terminal


def test_published_exporter_compatible_bundle_binds_original_joint_claim_and_all_doses(
    tmp_path, monkeypatch
):
    root, _, _ = _published(tmp_path, monkeypatch)
    assert result.verify(root, publication_commit=PIN) == "COMPLETE"


def test_invalid_harm_boolean_disagreement_remains_invalid(tmp_path, monkeypatch):
    def invalidate(value):
        value["checks"]["retained_d1024_harm_boolean_agrees"] = False
        value["status"] = "INVALID"

    root, _, _ = _published(tmp_path, monkeypatch, invalidate)
    assert result.verify(root, publication_commit=PIN) == "INVALID"


@pytest.mark.parametrize("failure_class", ["IndependentWatchdogKill", "RuntimeError"])
def test_published_finalization_kill_is_incomplete_with_honest_elapsed(
    tmp_path, monkeypatch, failure_class
):
    from research.bl_residual import f4_postmortem as entry

    root, destination, _ = _published(tmp_path, monkeypatch)
    parents = json.loads((destination / "development-parents.json").read_bytes())
    entry._record_watchdog_failure(
        PIN, {"metadata": {"parents": parents}}, destination, {"elapsed_seconds": 3600.02}
    )
    failure = json.loads((destination / "failure.json").read_bytes())
    assert failure["wall_seconds"] == 3600.02
    failure["failure_class"] = failure_class
    _sealed(destination, "failure.json", failure)
    binding = json.loads((destination / "result-binding.json").read_bytes())
    binding["outcomes"].append("failure.json")
    binding["bundle"]["failure.json"] = {
        "sha256": hashlib.sha256((destination / "failure.json").read_bytes()).hexdigest(),
        "size": (destination / "failure.json").stat().st_size,
    }
    _sealed(destination, "result-binding.json", binding)
    remote = {path.name: path.read_bytes() for path in destination.iterdir()}
    monkeypatch.setattr(result, "_origin_bytes", lambda root, path: remote[path.name])
    if failure_class == "IndependentWatchdogKill":
        assert result.verify(root, publication_commit=PIN) == "INCOMPLETE"
    else:
        with pytest.raises(ValueError, match="failure receipt schema"):
            result.verify(root, publication_commit=PIN)


@pytest.mark.parametrize(
    "field",
    [
        "claim_sha256",
        "ruling_sha256",
        "nice",
        "threads",
        "wall_seconds",
        "checkpoint",
        "hidden_input_rows",
    ],
)
def test_postpublication_verifier_rejects_foreign_provenance_or_runtime(
    tmp_path, monkeypatch, field
):
    def mutate(value):
        if field == "checkpoint":
            value["doses"][0]["checkpoint_sha256"] = "foreign"
        elif field == "hidden_input_rows":
            value["inputs"]["--d256-checkpoint"]["row_ids"] = ["forbidden"]
        else:
            value[field] = {"nice": 0, "threads": 2, "wall_seconds": 3601}.get(field, "foreign")

    root, _, _ = _published(tmp_path, monkeypatch, mutate)
    with pytest.raises(ValueError, match="schema|reconciliation"):
        result.verify(root, publication_commit=PIN)


def test_no_outcome_parsed_until_all_origin_bytes_match(tmp_path, monkeypatch):
    root, destination, value = _published(tmp_path, monkeypatch)
    value["status"] = "INVALID"
    _sealed(destination, "diagnostic.json", value)
    original = result._json_bytes
    outcome_bytes = (destination / "diagnostic.json").read_bytes()

    def parsed(payload):
        assert payload != outcome_bytes, "outcome parsed before publication proof"
        return original(payload)

    monkeypatch.setattr(result, "_json_bytes", parsed)
    with pytest.raises(ValueError, match="origin/main"):
        result.verify(root, publication_commit=PIN)


@pytest.mark.parametrize(
    ("published_outcome", "extra_outcome"),
    [("diagnostic", "failure"), ("failure", "diagnostic")],
)
def test_extra_locally_sealed_outcome_is_refused_before_any_outcome_parse(
    tmp_path, monkeypatch, published_outcome, extra_outcome
):
    root, destination, _ = _published(tmp_path, monkeypatch, outcome=published_outcome)
    _sealed_bytes(destination, f"{extra_outcome}.json", b"not-json-outcome\n")
    original = result._json_bytes
    outcome_bytes = {
        (destination / f"{name}.json").read_bytes() for name in (published_outcome, extra_outcome)
    }

    def parsed(payload):
        assert payload not in outcome_bytes, "outcome parsed before inventory proof"
        return original(payload)

    monkeypatch.setattr(result, "_json_bytes", parsed)
    with pytest.raises(ValueError, match="local sealed result inventory"):
        result.verify(root, publication_commit=PIN)


def test_parsed_snapshot_itself_must_match_binding_despite_concurrent_swap(tmp_path, monkeypatch):
    root, destination, value = _published(tmp_path, monkeypatch)
    origin = result._origin_bytes
    unbound = b"unbound-outcome-not-json\n"
    _sealed_bytes(destination, "diagnostic.json", unbound)

    def swapping_origin(root, path):
        if path.name == "diagnostic.json":
            # The first snapshot matches origin's reported bytes B, while a
            # different disk snapshot A matches the published binding. Hashing
            # the path again must not authenticate the cached B for parsing.
            _sealed(destination, "diagnostic.json", value)
            return unbound
        return origin(root, path)

    parse = result._json_bytes

    def decoding(payload):
        assert payload != unbound, "unbound outcome snapshot decoded"
        return parse(payload)

    monkeypatch.setattr(result, "_origin_bytes", swapping_origin)
    monkeypatch.setattr(result, "_json_bytes", decoding)
    with pytest.raises(ValueError, match="published result hash/size mismatch"):
        result.verify(root, publication_commit=PIN)
