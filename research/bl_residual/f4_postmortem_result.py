"""Outcome-blind seal, publication, and post-publication verification for F4."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import stat
import subprocess
from pathlib import Path
from typing import Any

from research.bl_ppo.exports import digest, export_json, verify_export
from research.bl_residual import f4_postmortem_guard as guard

RESULT_RELATIVE = Path("research/bl_residual/ledger/F4_POSTMORTEM_20261025")
MAX_FILE_BYTES = 2 * 1024 * 1024
BASE_NAMES = ("plan.json", "claim.json", "development-parents.json")
OUTCOME_NAMES = ("diagnostic.json", "failure.json")
RESULT_WORKTREE = guard.ROOT.parent / "snake-dqn-f4pm-result-20261025.noindex"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")
    return value


def _safe_files(root: Path, *, allow_binding: bool = False) -> set[str]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("unsafe bundle root")
    names: set[str] = set()
    for path in root.iterdir():
        if path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode):
            raise ValueError("bundle contains symlink or non-regular file")
        if path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError("bundle component exceeds 2 MiB limit")
        names.add(path.name)
    allowed = {
        f"{name}{suffix}"
        for name in (*BASE_NAMES, *OUTCOME_NAMES)
        for suffix in ("", ".sha256", ".complete")
    }
    if allow_binding:
        allowed.update(
            {"result-binding.json", "result-binding.json.sha256", "result-binding.json.complete"}
        )
    if not names <= allowed:
        raise ValueError("unexpected bundle filename")
    return names


def _sealed(root: Path, name: str) -> str:
    if not all((root / f"{name}{suffix}").is_file() for suffix in ("", ".sha256", ".complete")):
        raise ValueError(f"missing sealed {name}")
    return verify_export(root / name)


def verify_seal(
    bundle: Path,
    *,
    publication_commit: str,
    package_root: Path | None = None,
    allow_binding: bool = False,
) -> dict[str, Any]:
    """Check a terminal bundle without deserializing diagnostic/failure bytes."""
    bundle = Path(bundle)
    if len(publication_commit) != 40 or any(
        c not in "0123456789abcdef" for c in publication_commit
    ):
        raise ValueError("full implementation publication commit required")
    names = _safe_files(bundle, allow_binding=allow_binding)
    hashes = {name: _sealed(bundle, name) for name in BASE_NAMES}
    outcomes = [name for name in OUTCOME_NAMES if name in names]
    if not outcomes:
        raise ValueError("terminal diagnostic or failure receipt required")
    hashes.update({name: _sealed(bundle, name) for name in outcomes})
    plan, claim, development = (_json(bundle / name) for name in BASE_NAMES)
    static_root = package_root or (guard.ROOT / guard.PACKAGE)
    manifest = _json(static_root / "manifest.json")
    if (
        plan.get("schema") != "bl-residual-f4-postmortem-plan/v1"
        or plan.get("implementation_commit") != publication_commit
        or plan.get("ruling_sha256") != guard.RULING_SHA
        or plan.get("output_root") != str(guard.OUTPUT)
        or plan.get("manifest_sha256") != digest(static_root / "manifest.json")
        or plan.get("inputs") != manifest.get("allowed_payload_inputs")
        or plan.get("original_joint_claim") != manifest.get("immutable_joint_claim")
        or plan.get("runtime") != manifest.get("runtime")
        or claim.get("schema") != "bl-residual-f4-postmortem-claim/v1"
        or claim.get("implementation_commit") != publication_commit
        or claim.get("ruling_sha256") != guard.RULING_SHA
        or claim.get("plan_sha256") != hashes["plan.json"]
        or claim.get("manifest_sha256") != plan.get("manifest_sha256")
        or claim.get("retry_authorized") is not False
    ):
        raise ValueError("plan/claim binding mismatch")
    expected_development = _json(static_root / "development-parents.json")
    if development != expected_development:
        raise ValueError("development exclusion ledger differs from published static ledger")
    return {
        "bundle": str(bundle),
        "hashes": hashes,
        "outcomes": outcomes,
        "plan_sha256": hashes["plan.json"],
        "claim_sha256": hashes["claim.json"],
        "sizes": {name: (bundle / name).stat().st_size for name in hashes},
    }


def _git(root: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(root), *args], text=True, capture_output=True, timeout=60, check=False
    )
    if done.returncode:
        raise ValueError("result publication git operation failed")
    return done.stdout.strip()


def publish(bundle: Path, *, publication_commit: str, source_root: Path = guard.ROOT) -> str:
    """Copy sealed bytes into a fresh result worktree, then push without reading outcomes."""
    if Path(bundle) != guard.OUTPUT:
        raise ValueError("publish accepts only the fixed F4 output bundle")
    # Prove source and static implementation publication before inspecting even
    # metadata in the claimed bundle.  This remains outcome-blind.
    guard.publication(publication_commit, root=source_root)
    seal = verify_seal(bundle, publication_commit=publication_commit)
    _git(source_root, "fetch", "--no-tags", "origin", "main")
    _git(source_root, "merge-base", "--is-ancestor", publication_commit, "origin/main")
    branch = "f4-postmortem-result-20261025"
    worktree = (
        RESULT_WORKTREE
        if source_root == guard.ROOT
        else source_root.parent / "snake-dqn-f4pm-result-20261025.noindex"
    )
    if worktree.exists() or _git(source_root, "branch", "--list", branch):
        raise FileExistsError("result publication worktree or branch already exists; no retry")
    _git(source_root, "worktree", "add", "--detach", str(worktree), "origin/main")
    try:
        _git(worktree, "checkout", "-b", branch)
        destination = worktree / RESULT_RELATIVE
        destination.mkdir(parents=True, exist_ok=False)
        for name in (*BASE_NAMES, *seal["outcomes"]):
            for suffix in ("", ".sha256", ".complete"):
                source, copied = Path(bundle) / f"{name}{suffix}", destination / f"{name}{suffix}"
                shutil.copyfile(source, copied)
                if source.read_bytes() != copied.read_bytes():
                    raise ValueError("copy proof differs from presealed source bytes")
            if (
                verify_export(destination / name) != seal["hashes"][name]
                or (destination / name).stat().st_size != seal["sizes"][name]
            ):
                raise ValueError("copied bytes differ from original immutable seal")
        binding = {
            "schema": "bl-residual-f4-postmortem-result-binding/v1",
            "implementation_commit": publication_commit,
            "ruling_sha256": guard.RULING_SHA,
            "bundle": {
                name: {"sha256": value, "size": seal["sizes"][name]}
                for name, value in seal["hashes"].items()
            },
            "outcomes": seal["outcomes"],
        }
        export_json(destination / "result-binding.json", binding)
        venv = worktree / "venv"
        if not venv.exists():
            venv.symlink_to(source_root / "venv")
        _git(worktree, "add", str(RESULT_RELATIVE))
        _git(worktree, "commit", "-m", "Publish F4 harm post-mortem aggregate bundle")
        _git(worktree, "fetch", "--no-tags", "origin", "main")
        _git(worktree, "merge", "--no-edit", "origin/main")
        commit = _git(worktree, "rev-parse", "HEAD")
        _git(worktree, "push", "origin", "HEAD:main", f"HEAD:refs/heads/{branch}")
        print(
            json.dumps(
                {"publication_commit": commit, "claim_sha256": seal["claim_sha256"]}, sort_keys=True
            )
        )
        return commit
    finally:
        # Preserve a failed worktree for forensic inspection; successful worktrees
        # are intentionally retained too, so this never destructively cleans up.
        pass


def _origin_bytes(root: Path, relative: Path) -> bytes:
    done = subprocess.run(
        ["git", "-C", str(root), "show", f"origin/main:{relative}"], capture_output=True, timeout=60
    )
    if done.returncode:
        raise ValueError("published result component absent from origin/main")
    return done.stdout


def verify(root: Path = RESULT_WORKTREE, *, publication_commit: str) -> str:
    """Verify published bytes first, then parse the aggregate-only terminal result."""
    _git(root, "fetch", "--no-tags", "origin", "main")
    destination = Path(root) / RESULT_RELATIVE
    binding_path = destination / "result-binding.json"
    if binding_path.read_bytes() != _origin_bytes(root, RESULT_RELATIVE / "result-binding.json"):
        raise ValueError("local result binding differs from origin/main")
    for suffix in (".sha256", ".complete"):
        if Path(str(binding_path) + suffix).read_bytes() != _origin_bytes(
            root, RESULT_RELATIVE / ("result-binding.json" + suffix)
        ):
            raise ValueError("result binding sidecar differs from origin/main")
    verify_export(binding_path)
    binding = _json(binding_path)
    allowed = set(BASE_NAMES) | set(OUTCOME_NAMES)
    if set(binding) != {
        "schema",
        "implementation_commit",
        "ruling_sha256",
        "bundle",
        "outcomes",
    } or (
        binding.get("schema") != "bl-residual-f4-postmortem-result-binding/v1"
        or not isinstance(binding.get("bundle"), dict)
        or not isinstance(binding.get("outcomes"), list)
        or set(binding["bundle"]) != set(BASE_NAMES) | set(binding["outcomes"])
        or not set(binding["outcomes"]).issubset(set(OUTCOME_NAMES))
        or not set(binding["bundle"]).issubset(allowed)
    ):
        raise ValueError("unsafe result binding")
    if (
        binding.get("implementation_commit") != publication_commit
        or binding.get("ruling_sha256") != guard.RULING_SHA
    ):
        raise ValueError("publication binding mismatch")
    for name, info in binding.get("bundle", {}).items():
        if Path(name).name != name or not isinstance(info, dict):
            raise ValueError("unsafe result binding path")
        path = destination / name
        if path.read_bytes() != _origin_bytes(root, RESULT_RELATIVE / name):
            raise ValueError("local outcome bytes differ from origin/main")
        if digest(path) != info.get("sha256") or path.stat().st_size != info.get("size"):
            raise ValueError("published result hash/size mismatch")
        for suffix in (".sha256", ".complete"):
            sidecar = Path(str(path) + suffix)
            if sidecar.read_bytes() != _origin_bytes(root, RESULT_RELATIVE / sidecar.name):
                raise ValueError("local result sidecar differs from origin/main")
        verify_export(path)
    # Only now is static plan/claim metadata parsed; diagnostic/failure remains unread.
    verify_seal(
        destination,
        publication_commit=publication_commit,
        package_root=guard.ROOT / guard.PACKAGE,
        allow_binding=True,
    )
    diagnostic = destination / "diagnostic.json"
    if (destination / "failure.json").exists():
        failure = _json(destination / "failure.json")
        if (
            set(failure)
            != {
                "schema",
                "status",
                "purpose",
                "f4_disposition",
                "ruling_sha256",
                "implementation_commit",
                "claim_sha256",
                "started_at",
                "finished_at",
                "wall_seconds",
                "failure_class",
                "retry_authorized",
            }
            or failure.get("schema") != "bl-residual-f4-postmortem-failure/v1"
            or failure.get("status") != "INCOMPLETE"
            or failure.get("purpose") != "DEVELOPMENT_ONLY"
            or failure.get("f4_disposition") != "STOP_FUNDING"
            or failure.get("ruling_sha256") != guard.RULING_SHA
            or failure.get("implementation_commit") != publication_commit
            or failure.get("claim_sha256") != binding["bundle"]["claim.json"]["sha256"]
            or failure.get("retry_authorized") is not False
            or type(failure.get("wall_seconds")) not in (float, int)
            or not math.isfinite(failure["wall_seconds"])
            or not 0 <= failure["wall_seconds"] <= 3600
            or not isinstance(failure.get("failure_class"), str)
            or not failure["failure_class"]
        ):
            raise ValueError("failure receipt schema invalid")
    if not diagnostic.exists():
        print("VERIFIED_INCOMPLETE")
        return "INCOMPLETE"
    value = _json(diagnostic)
    # Import is deliberately after origin/main bytes are proved; this module imports torch.
    from research.bl_residual.f4_postmortem_counts import dose_shape_valid, valid_dose

    parents = _json(destination / "development-parents.json")["parent_worlds"]
    plan = _json(destination / "plan.json")
    claim_sha = binding["bundle"]["claim.json"]["sha256"]
    expected_inputs = {**plan["inputs"], "--original-joint-claim": plan["original_joint_claim"]}

    def normalized_inputs(rows: object, *, plan_form: bool) -> dict[str, dict[str, object]]:
        if not isinstance(rows, dict):
            return {}
        return {
            str(key): {
                "path": value.get("path"),
                "size": value.get("size_bytes" if plan_form else "size"),
                "sha256": value.get("sha256"),
            }
            for key, value in rows.items()
            if isinstance(value, dict)
        }

    census = value.get("census")
    expected_keys = {
        "schema",
        "ruling_sha256",
        "implementation_commit",
        "source_sha256",
        "inputs",
        "claim_sha256",
        "started_at",
        "finished_at",
        "platform",
        "nice",
        "threads",
        "wall_seconds",
        "status",
        "purpose",
        "f4_disposition",
        "census",
        "doses",
        "checks",
    }
    if (
        set(value) != expected_keys
        or value.get("schema") != "bl-residual-f4-harm-postmortem/v1"
        or value.get("status") not in ("COMPLETE", "INVALID")
        or value.get("purpose") != "DEVELOPMENT_ONLY"
        or value.get("f4_disposition") != "STOP_FUNDING"
        or value.get("ruling_sha256") != guard.RULING_SHA
        or value.get("implementation_commit") != publication_commit
        or value.get("claim_sha256") != claim_sha
        or value.get("source_sha256") != plan.get("source_sha256")
        or normalized_inputs(value.get("inputs"), plan_form=False)
        != normalized_inputs(expected_inputs, plan_form=True)
        or not all(
            isinstance(r, dict) and set(r) == {"path", "size", "sha256"}
            for r in value["inputs"].values()
        )
        or type(value.get("nice")) is not int
        or value["nice"] != 15
        or type(value.get("threads")) is not int
        or value["threads"] != 1
        or not isinstance(value.get("platform"), str)
        or not value["platform"].startswith(("macOS", "Darwin"))
        or not isinstance(value.get("wall_seconds"), (int, float))
        or isinstance(value.get("wall_seconds"), bool)
        or not math.isfinite(float(value["wall_seconds"]))
        or not 0 <= float(value["wall_seconds"]) <= 3600
        or census
        != {
            "total_rows": 5936,
            "natural_rows": 560,
            "eligible_lives": 71,
            "reserve_lives": 64,
            "parents": 29,
        }
        or not isinstance(value.get("checks"), dict)
        or set(value["checks"])
        != {
            "d256_census",
            "d256_bins_parents_partitions_denominators",
            "d256_same_pass_existing_H",
            "d1024_census",
            "d1024_bins_parents_partitions_denominators",
            "d1024_same_pass_existing_H",
            "retained_d1024_harm_boolean_agrees",
        }
        or not all(type(flag) is bool for flag in value["checks"].values())
        or len(value.get("doses", [])) != 2
        or [row.get("dose") for row in value["doses"]] != [256, 1024]
        or not all(dose_shape_valid(row, parents) for row in value["doses"])
        or any(
            row["checkpoint_sha256"] != expected_inputs[f"--d{row['dose']}-checkpoint"]["sha256"]
            for row in value["doses"]
        )
    ):
        raise ValueError("aggregate-only diagnostic schema or reconciliation invalid")
    expected_status = "COMPLETE" if all(value["checks"].values()) else "INVALID"
    for row in value["doses"]:
        if (
            valid_dose(row, parents)
            is not value["checks"][f"d{row['dose']}_bins_parents_partitions_denominators"]
        ):
            raise ValueError("diagnostic counters disagree with recorded arithmetic check")
    if value["status"] != expected_status:
        raise ValueError("diagnostic status does not match fixed checks")
    if (destination / "failure.json").exists():
        expected_status = "INCOMPLETE"
    print(f"VERIFIED_{expected_status}")
    return expected_status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("verify-seal", "publish", "verify"))
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--result-worktree", type=Path, default=RESULT_WORKTREE)
    parser.add_argument("--publication-commit", required=True)
    args = parser.parse_args()
    if args.action in ("verify-seal", "publish") and args.bundle != guard.OUTPUT:
        parser.error("--bundle must be the single ruling-bound F4 output directory")
    if args.action == "verify-seal":
        if args.bundle is None:
            parser.error("--bundle required")
        print(
            json.dumps(
                verify_seal(args.bundle, publication_commit=args.publication_commit), sort_keys=True
            )
        )
    elif args.action == "publish":
        if args.bundle is None:
            parser.error("--bundle required")
        publish(args.bundle, publication_commit=args.publication_commit)
    else:
        verify(args.result_worktree, publication_commit=args.publication_commit)


if __name__ == "__main__":
    main()
