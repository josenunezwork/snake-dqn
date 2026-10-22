"""Prepare/admit outcome-blind DENSITY-2 completion; never allocate Colab."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
from pathlib import Path
from typing import Any

from research.bl_density import manifests
from research.bl_density.runner import CANARY_NAMESPACE, DENSITY2_ALLOCATION, DENSITY2_NAMESPACE
from research.bl_ppo.exports import export_json, verify_export
from research.colab_cli import canary as family_canary
from research.colab_cli.schema import validate_manifest

SCIENCE_COMMIT = "f1904777aded79366703c94afbf5dc13155bb0ee"
PRIOR = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-density/"
    "density2-20261022-gatefix-f1904777/production-output-02"
)
AMENDMENT = "research/bl_density/METRICS_AMENDMENT_05.md"
TELEMETRY_BRANCH = "origin/supervisor-telemetry-20261022"
PHASES = ("skip-ledger-canary", "production-completion")


def _json(path: Path) -> Any:
    verify_export(path)
    return json.loads(path.read_text())


def check_science(repo: Path) -> None:
    """Reject science drift outside the expressly amended orchestration modules."""
    import ast

    for filename in (
        "METRICS.md",
        *(f"METRICS_AMENDMENT_0{i}.md" for i in range(1, 5)),
        "statistics.py",
    ):
        relative = "research/bl_density/" + filename
        baseline = subprocess.check_output(
            ["git", "show", f"{SCIENCE_COMMIT}:{relative}"], cwd=repo
        )
        if (repo / relative).read_bytes() != baseline:
            raise ValueError("frozen science changed: " + relative)
    # All existing analysis definitions and collector science stay identical.
    for filename, names in (
        ("analysis.py", None),
        (
            "runner.py",
            (
                "cheap_prefilter",
                "effective_q_features",
                "legal_alternatives",
                "retrospective_doom",
                "_profile_override",
                "_root_geometry",
                "_inject_p1_long",
                "_rollout_continuation",
                "_build",
                "_step",
                "run_episode",
                "planned_jobs",
            ),
        ),
    ):
        relative = "research/bl_density/" + filename
        baseline = ast.parse(
            subprocess.check_output(
                ["git", "show", f"{SCIENCE_COMMIT}:{relative}"], cwd=repo, text=True
            )
        )
        current_text = (repo / relative).read_text()
        current = ast.parse(current_text)
        if filename == "analysis.py":
            original = subprocess.check_output(
                ["git", "show", f"{SCIENCE_COMMIT}:{relative}"], cwd=repo, text=True
            )
            if not current_text.startswith(original):
                raise ValueError("frozen analysis prefix changed")
        else:
            baseline_constants = [node for node in baseline.body if isinstance(node, ast.Assign)]
            current_constants = [node for node in current.body if isinstance(node, ast.Assign)]
            for node in baseline_constants:
                if not any(
                    ast.dump(node, include_attributes=False)
                    == ast.dump(other, include_attributes=False)
                    for other in current_constants
                ):
                    raise ValueError("frozen runner constants changed")
        definitions = {n.name: n for n in current.body if isinstance(n, ast.FunctionDef)}
        for node in baseline.body:
            if isinstance(node, ast.FunctionDef) and (names is None or node.name in names):
                if node.name not in definitions or ast.dump(
                    node, include_attributes=False
                ) != ast.dump(definitions[node.name], include_attributes=False):
                    raise ValueError("frozen science changed: " + relative + ":" + node.name)


def _transport_on_main(repo: Path, commit: str) -> str:
    tip = subprocess.check_output(
        ["git", "rev-parse", TELEMETRY_BRANCH], cwd=repo, text=True
    ).strip()
    for target in ("origin/main", commit):
        if subprocess.run(["git", "merge-base", "--is-ancestor", tip, target], cwd=repo).returncode:
            raise ValueError(
                f"telemetry correction {tip} must merge before admission; repin {target}"
            )
    return tip


def prepare(
    commit: str,
    directory: Path,
    ledger: Path,
    *,
    prior: Path = PRIOR,
    repo: Path = manifests.ROOT,
    pending: bool = True,
    qualification: Path = manifests.ACCEPTANCE_RESULT,
    attempt: int = 1,
    previous_output: Path | None = None,
) -> dict[str, Any]:
    """Create two immutable same-input templates; episode payloads remain opaque."""
    from research.bl_density.ledger import build_ledger, validate_ledger

    if attempt not in (1, 2):
        raise ValueError("at most completion-01 and completion-02")
    check_science(repo)
    source = json.loads((prior / "manifest.json").read_text())
    if source["commit"] != SCIENCE_COMMIT or source["density"]["namespace"] != DENSITY2_NAMESPACE:
        raise ValueError("original production-02 science identity required")
    ledger_value = _json(ledger)
    # Runtime planning validates every identity against the same fixed allocation.
    validate_ledger(ledger_value, namespace=DENSITY2_NAMESPACE, allocation=DENSITY2_ALLOCATION)
    entries = ledger_value["entries"]
    if attempt == 1 and len(entries) != 408:
        raise ValueError("completion-01 requires exactly 408 verified ledger entries")
    if attempt == 2 and not 408 <= len(entries) < 492:
        raise ValueError("completion-02 requires a mechanically grown incomplete ledger")
    expected_ids = {
        f"{DENSITY2_NAMESPACE}/{arm}/{mix}/{index:03d}"
        for arm, cells in DENSITY2_ALLOCATION.items()
        for mix, count in cells.items()
        for index in range(count)
    }
    original = build_ledger(
        prior / "exports/density/episodes",
        prior / "poll-000015.json",
        namespace=DENSITY2_NAMESPACE,
        expected_ids=expected_ids,
    )
    expected_entries = {e["episode_id"]: e for e in original["entries"]}
    if attempt == 2:
        if previous_output is None:
            raise ValueError("completion-02 requires previous mechanical attempt evidence")
        receipt = json.loads((previous_output / "receipt.json").read_text())
        previous_manifest = json.loads((previous_output / "manifest.json").read_text())
        if (
            receipt.get("status") != "INCOMPLETE"
            or receipt.get("stop_verified") is not True
            or receipt.get("cleanup_error")
            or not (
                receipt.get("failure_class") == "RPC_EXEC_POLL_TIMEOUT"
                or receipt.get("failure_class") == "SUPERVISOR"
                and any(
                    marker in str(receipt.get("error", "")).lower()
                    for marker in ("download", "transfer", "export", "hash", "deadline", "timeout")
                )
                or receipt.get("failure_class") == "CHILD_NONZERO_EXIT"
                and "cuda"
                in str((receipt.get("terminal") or {}).get("last_traceback_line", "")).lower()
            )
            or previous_manifest.get("density", {}).get("completion_attempt") != 1
        ):
            raise ValueError("completion-02 requires a verified mechanical failure and release")
        if (
            previous_manifest.get("density", {}).get("phase") != "production-completion"
            or previous_manifest.get("density", {}).get("science_commit") != SCIENCE_COMMIT
            or receipt.get("commit") != previous_manifest.get("commit")
            or receipt.get("job_id") != previous_manifest.get("job_id")
        ):
            raise ValueError("prior completion identity differs")
        if receipt.get("manifest_sha256") != manifests.digest(previous_output / "manifest.json"):
            raise ValueError("prior completion receipt manifest binding differs")
        episode_dir = previous_output / "exports/density/episodes"
        recovered = (
            build_ledger(
                episode_dir,
                previous_output / "receipt.json",
                namespace=DENSITY2_NAMESPACE,
                expected_ids=expected_ids,
            )
            if any(episode_dir.glob("*.json"))
            else {"entries": []}
        )
        for entry in recovered["entries"]:
            if entry["episode_id"] in expected_entries:
                raise ValueError("retry ledger has duplicate original unit")
            expected_entries[entry["episode_id"]] = entry
    if {e["episode_id"]: e for e in entries} != expected_entries:
        raise ValueError("ledger must equal all verified original/authorized attempt exports")
    directory.mkdir(parents=True, exist_ok=False)
    # Synthetic A canary entries are never production data and are never scored.
    canary_ledger = directory / "canary-ledger.json"
    export_json(
        canary_ledger,
        {
            "schema": "bl-density-verified-ledger/v1",
            "namespace": CANARY_NAMESPACE,
            "entries": [
                {"episode_id": f"{CANARY_NAMESPACE}/A/{mix}/000", "sha256": "0" * 64, "size": 1}
                for mix in ("frozen", "mixed", "scripted")
            ],
        },
    )
    local_ledger = directory / "skip-ledger.json"
    export_json(local_ledger, ledger_value)
    input_hashes = directory / "input-hashes.json"
    input_paths = {
        "public-inputs.json": repo / "research/t34_exit_b8/PUBLIC_INPUTS.json",
        "skip-ledger.json": local_ledger,
        "canary-ledger.json": canary_ledger,
    }
    export_json(input_hashes, {dest: manifests.digest(path) for dest, path in input_paths.items()})
    input_paths["input-hashes.json"] = input_hashes
    paths = sorted(set(manifests.source_paths(repo)) | {AMENDMENT})
    manifests._published_and_clean(commit, paths, repo, require_main=not pending)
    source_qualification = (
        {
            "status": "PENDING_REQUALIFICATION",
            "receipt_path": str(qualification.resolve()),
            "reason": "qualified telemetry tree and main merge required before admission",
        }
        if pending
        else manifests.source_bound_transport_receipt(repo, qualification)
    )
    inputs = [manifests._input(path, dest) for dest, path in input_paths.items()]
    values = {}
    for phase in PHASES:
        probe = phase == "skip-ledger-canary"
        value = copy.deepcopy(source)
        value.pop("canary", None)
        value.update(
            commit=commit,
            job_id=(
                "bl-density2-"
                + ("skip-canary" if probe else "completion")
                + f"-{attempt:02d}-{commit[:8]}"
            ),
            output_dir=str(
                (directory / ("canary-output" if probe else "completion-output")).resolve()
            ),
            inputs=inputs,
            source_paths=paths,
            scientific_admission=False,
            source_qualification=source_qualification,
            timeout_minutes=30 if probe else 180,
            required_exports=["density/summary.json", "density/attempt.json"],
            resume={
                "unit_kind": "episode",
                "ledger_path": "inputs/canary-ledger.json" if probe else "inputs/skip-ledger.json",
                "skip_verified": True,
                "max_outcome_blind_retries": 2,
            },
        )
        value["requirements"]["est_minutes"] = 10 if probe else 35
        value["budget"].update(unit_cap=1.0 if probe else 5.0, reserve_units=0.2 if probe else 1.0)
        value["env"].pop("BL_DENSITY_CANARY_PROOF_JSON", None)
        value["env"].update(
            BL_DENSITY_COMMIT=commit,
            BL_DENSITY_SCIENCE_COMMIT=SCIENCE_COMMIT,
            BL_DENSITY_PHASE=phase,
            BL_DENSITY_SCIENTIFIC_ADMISSION="0",
            BL_DENSITY_COMPLETION_AMENDMENT_SHA256=manifests.digest(repo / AMENDMENT),
            BL_DENSITY_LEDGER_SHA256=manifests.digest(local_ledger),
            BL_DENSITY_CANARY_LEDGER_SHA256=manifests.digest(canary_ledger),
            BL_DENSITY_INPUT_HASHES_SHA256=manifests.digest(input_hashes),
        )
        value["density"].update(
            phase=phase,
            status="NOT_READY",
            scientific_admission=False,
            science_commit=SCIENCE_COMMIT,
            source_qualification_status=source_qualification["status"],
            prior_spend_units=4.50,
            total_envelope_unit_cap=16.50,
            effective_production_unit_cap=5.0,
            completion_attempt=attempt,
            base_envelope={
                "unit_cap": 0.5 if probe else 2.5,
                "reserve_units": 0.1 if probe else 0.5,
                "timeout_minutes": 15 if probe else 90,
            },
            effective_envelope={
                "unit_cap": 1.0 if probe else 5.0,
                "reserve_units": 0.2 if probe else 1.0,
                "timeout_minutes": 30 if probe else 180,
            },
            maximum_completion_attempts=2,
            skip_ledger_sha256=manifests.digest(local_ledger),
            canary_ledger_sha256=manifests.digest(canary_ledger),
            canary_skipped_episode_ids=[
                row["episode_id"] for row in json.loads(canary_ledger.read_text())["entries"]
            ],
            input_hashes_sha256=manifests.digest(input_hashes),
            completion_amendment_sha256=manifests.digest(repo / AMENDMENT),
            ledger_entries=len(entries),
            missing_registered_episodes=492 - len(entries),
            identity_replay_episodes=0 if probe else 12,
            attempted_episodes=9 if probe else 492 - len(entries) + 12,
            total_episodes=12 if probe else 492,
            namespace=CANARY_NAMESPACE if probe else DENSITY2_NAMESPACE,
            episodes_per_mix_by_arm=(
                {
                    arm: {mix: 1 for mix in ("frozen", "mixed", "scripted")}
                    for arm in DENSITY2_ALLOCATION
                }
                if probe
                else DENSITY2_ALLOCATION
            ),
            workers=1 if probe else 10,
            horizon=8 if probe else 5000,
            work_wall_seconds=1200 if probe else 10800,
            one_terminal_look=not probe,
            production_gate="REQUALIFIED_TELEMETRY_AND_SKIP_CANARY_REQUIRED",
            original_manifest_sha256=manifests.digest(prior / "manifest.json"),
            original_receipt_sha256=manifests.digest(prior / "receipt.json"),
        )
        if probe:
            value["env"].update(
                SNAKE_COLAB_CANARY="1", SNAKE_COLAB_CANARY_PACKAGE=manifests.PACKAGE
            )
            value["canary"] = {
                "schema": family_canary.SCHEMA,
                "mode": "probe",
                "family": family_canary.family(value),
            }
        manifests.assert_upload_budget(value, repo=repo)
        validate_manifest(value)
        target = directory / ("canary-template.json" if probe else "completion-template.json")
        export_json(target, value)
        values[phase] = str(target)
    export_json(
        directory / "preparation.json",
        {
            "schema": "bl-density-completion-preparation/v1",
            "commit": commit,
            "science_commit": SCIENCE_COMMIT,
            "templates": values,
            "outcomes_read": False,
            "ledger_entries": len(entries),
            "missing_episodes": 492 - len(entries),
            "replay_episodes": 12,
            "source_qualification": source_qualification,
        },
    )
    return values


def admit(
    template: dict[str, Any],
    *,
    qualification: Path,
    repo: Path = manifests.ROOT,
    canary_proof: Path | None = None,
    canary_receipt: Path | None = None,
    other_assignments: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Require actual requalification and recovered canary evidence, offline only."""
    from research.bl_density.canary import verify_skip_proof

    if template.get("scientific_admission") is not False:
        raise ValueError("non-admitted template required")
    phase = template.get("density", {}).get("phase")
    if phase not in PHASES:
        raise ValueError("completion or skip-ledger-canary required")
    manifests._published_and_clean(template["commit"], template["source_paths"], repo)
    _transport_on_main(repo, template["commit"])
    check_science(repo)
    value = copy.deepcopy(template)
    value["source_qualification"] = manifests.source_bound_transport_receipt(repo, qualification)
    if phase == "production-completion":
        if canary_proof is None or canary_receipt is None:
            raise ValueError("recovered skip-ledger canary proof and family receipt required")
        d = value["density"]
        proof = verify_skip_proof(
            canary_proof,
            value["commit"],
            d["round14_amendment_sha256"],
            d["skip_ledger_sha256"],
            d["canary_ledger_sha256"],
            d["input_hashes_sha256"],
        )
        if proof["receipt_sha256"] != manifests.digest(canary_receipt):
            raise ValueError("skip-ledger proof does not bind family receipt")
        value["canary"] = {
            "receipt": {
                "path": str(canary_receipt.resolve()),
                "sha256": manifests.digest(canary_receipt),
            }
        }
        value["env"]["BL_DENSITY_CANARY_PROOF_JSON"] = json.dumps(proof, sort_keys=True)
    elif canary_proof or canary_receipt:
        raise ValueError("canary admission does not accept production proof")
    if other_assignments is not None:
        value["budget"]["other_assignments"] = other_assignments
    value["scientific_admission"] = True
    value["env"]["BL_DENSITY_SCIENTIFIC_ADMISSION"] = "1"
    value["density"].update(
        scientific_admission=True,
        status="ADMITTED_OFFLINE_PENDING_DISPATCH",
        source_qualification_status="PASS_FINAL_ONLY",
        production_gate="ADMITTED_OFFLINE_PENDING_DISPATCH",
    )
    manifests.assert_upload_budget(value, repo=repo)
    validate_manifest(value)
    manifests.validate(value, repo=repo)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("prepare")
    build.add_argument("--commit", required=True)
    build.add_argument("--directory", type=Path, required=True)
    build.add_argument("--ledger", type=Path, required=True)
    build.add_argument("--prior", type=Path, default=PRIOR)
    build.add_argument("--attempt", type=int, default=1)
    build.add_argument("--previous-output", type=Path)
    build.add_argument("--qualification", type=Path, default=manifests.ACCEPTANCE_RESULT)
    build.add_argument("--qualified", action="store_true")
    admit_parser = commands.add_parser("admit")
    admit_parser.add_argument("--template", type=Path, required=True)
    admit_parser.add_argument("--qualification", type=Path, required=True)
    admit_parser.add_argument("--manifest", type=Path, required=True)
    admit_parser.add_argument("--canary-proof", type=Path)
    admit_parser.add_argument("--canary-receipt", type=Path)
    admit_parser.add_argument("--other-assignments", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(
            args.commit,
            args.directory,
            args.ledger,
            prior=args.prior,
            qualification=args.qualification,
            pending=not args.qualified,
            attempt=args.attempt,
            previous_output=args.previous_output,
        )
    else:
        value = admit(
            _json(args.template),
            qualification=args.qualification,
            canary_proof=args.canary_proof,
            canary_receipt=args.canary_receipt,
            other_assignments=json.loads(args.other_assignments.read_text()),
        )
        result = export_json(args.manifest, value)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
