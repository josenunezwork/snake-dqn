#!/usr/bin/env python3
"""Check a documentation-only agent plan; never dispatch or run product tests."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

PLAN_DIR = Path(__file__).resolve().parent
REPO_ROOT = PLAN_DIR.parents[2]


def overlap(left: str, right: str) -> bool:
    """Match exact files and explicit directory ownership prefixes."""
    return (
        left.rstrip("/") == right.rstrip("/")
        or (left.endswith("/") and right.startswith(left))
        or (right.endswith("/") and left.startswith(right))
    )


def validate(plan: dict, *, check_cards: bool = True) -> tuple[list[str], dict]:
    """Validate dependency closure, file handoffs, tests and planning status."""
    errors: list[str] = []
    nodes = plan["packages"]
    by_id = {node["id"]: node for node in nodes}
    if len(by_id) != len(nodes):
        errors.append("Duplicate package IDs")
    dependencies = {}
    for node in nodes:
        conditional = [
            package
            for condition in node.get("conditional_dependencies", [])
            for package in condition["requires"]
        ]
        dependencies[node["id"]] = node["depends_on"] + conditional
        if node["status"] != "PLANNED":
            errors.append(f"{node['id']}: this delivered plan must remain PLANNED")
        for dep in dependencies[node["id"]]:
            if dep not in by_id:
                errors.append(f"{node['id']}: unknown dependency {dep}")
            elif by_id[dep]["wave"] >= node["wave"]:
                errors.append(f"{node['id']}: {dep} must be in an earlier wave")
        for path in node["owned_files"]:
            if Path(path).is_absolute() or ".." in Path(path).parts:
                errors.append(f"{node['id']}: ownership must be repository-relative: {path}")
            if any(character in path for character in "*?["):
                errors.append(f"{node['id']}: use exact files or explicit directories: {path}")
            if not (REPO_ROOT / path).exists() and path not in node["new_files"]:
                errors.append(f"{node['id']}: missing owned path not declared new: {path}")
        if not set(node["new_files"]).issubset(node["owned_files"]):
            errors.append(f"{node['id']}: new files outside ownership")
        if check_cards:
            card = PLAN_DIR / node["card"]
            if not card.exists():
                errors.append(f"{node['id']}: missing card")
            else:
                content = card.read_text()
                if not content.startswith(f"# {node['id']} — {node['title']}"):
                    errors.append(f"{node['id']}: card title/ID mismatch")
                if any(f"`{path}`" not in content for path in node["owned_files"]):
                    errors.append(f"{node['id']}: card omits an owned file")
                if f"**Earliest wave:** {node['wave']}." not in content:
                    errors.append(f"{node['id']}: stale card wave")
                dependency_text = ", ".join(node["depends_on"]) or (
                    "None; can start in the first parallel group."
                )
                if f"**Dependencies:** {dependency_text}" not in content:
                    errors.append(f"{node['id']}: stale card dependencies")
                required_text = (
                    node["steps"]
                    + node["gates"]
                    + [node["goal"], node["rollback"], node["artifact_directory"]]
                )
                if any(value not in content for value in required_text):
                    errors.append(f"{node['id']}: card differs from manifest requirements")
    ancestors: dict[str, set[str]] = {}
    visiting: set[str] = set()

    def visit(package: str) -> set[str]:
        if package in visiting:
            errors.append(f"Dependency cycle through {package}")
            return set()
        if package in ancestors:
            return ancestors[package]
        visiting.add(package)
        result = set()
        for dependency in dependencies.get(package, []):
            if dependency in by_id:
                result.add(dependency)
                result.update(visit(dependency))
        visiting.remove(package)
        ancestors[package] = result
        return result

    for package in by_id:
        visit(package)
    handoffs = []
    for index, left in enumerate(nodes):
        for right in nodes[index + 1 :]:
            shared = sorted(
                {a for a in left["owned_files"] for b in right["owned_files"] if overlap(a, b)}
            )
            if not shared:
                continue
            if left["id"] in ancestors.get(right["id"], set()):
                handoffs.append([left["id"], right["id"], shared])
            elif right["id"] in ancestors.get(left["id"], set()):
                handoffs.append([right["id"], left["id"], shared])
            else:
                errors.append(f"Unordered writers {left['id']} / {right['id']}: {shared}")
    for node in nodes:
        for target in node["test_targets"]:
            if (REPO_ROOT / target).exists():
                continue
            owners = {candidate["id"] for candidate in nodes if target in candidate["new_files"]}
            permitted = ancestors.get(node["id"], set()) | {node["id"]}
            if not owners.intersection(permitted):
                errors.append(f"{node['id']}: nonexistent test has no upstream owner: {target}")
    ready = [node["id"] for node in nodes if not dependencies[node["id"]]]
    if len(ready) > plan["resource_limits"]["max_write_packages"]:
        errors.append("Initial group exceeds declared writer limit")
    return errors, {"ancestors": ancestors, "handoffs": handoffs, "initial_ready": ready}


def self_test(plan: dict) -> int:
    """Verify the checker rejects realistic planning failures."""
    mutations = []
    candidate = copy.deepcopy(plan)
    candidate["packages"][0]["depends_on"].append("DOES_NOT_EXIST")
    mutations.append(candidate)
    candidate = copy.deepcopy(plan)
    by_id = {node["id"]: node for node in candidate["packages"]}
    by_id["C0"]["depends_on"].append("P2")
    mutations.append(candidate)
    candidate = copy.deepcopy(plan)
    by_id = {node["id"]: node for node in candidate["packages"]}
    by_id["S0"]["owned_files"].append("src/training/pqn_trainer.py")
    mutations.append(candidate)
    candidate = copy.deepcopy(plan)
    candidate["packages"][0]["test_targets"].append("tests/not_an_actual_test.py")
    mutations.append(candidate)
    candidate = copy.deepcopy(plan)
    by_id = {node["id"]: node for node in candidate["packages"]}
    by_id["L1"]["depends_on"] = ["C0"]
    mutations.append(candidate)
    for index, changed in enumerate(mutations):
        errors, _ = validate(changed, check_cards=False)
        if not errors:
            raise AssertionError(f"Mutated invalid plan {index} was accepted")
    return len(mutations)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--batch", nargs="*", help="Check a proposed simultaneous writer group")
    args = parser.parse_args()
    plan = json.loads((PLAN_DIR / "dispatch.json").read_text())
    errors, details = validate(plan)
    if args.batch:
        by_id = {node["id"]: node for node in plan["packages"]}
        if len(args.batch) > plan["resource_limits"]["max_write_packages"]:
            errors.append("Proposed batch exceeds writer limit")
        if len(args.batch) != len(set(args.batch)):
            errors.append("Proposed batch repeats a package")
        for package in args.batch:
            if package not in by_id:
                errors.append(f"Unknown batch package: {package}")
        for index, left in enumerate(args.batch):
            for right in args.batch[index + 1 :]:
                if right in details["ancestors"].get(left, set()) or left in details[
                    "ancestors"
                ].get(right, set()):
                    errors.append(f"Dependent packages cannot share a launch batch: {left}/{right}")
    tests = self_test(plan) if args.self_test and not errors else 0
    print(
        json.dumps(
            {
                "status": "FAIL" if errors else "PASS",
                "packages": len(plan["packages"]),
                "initial_ready": details["initial_ready"],
                "ordered_shared_file_pairs": len(details["handoffs"]),
                "invalid_plan_self_tests": tests,
                "errors": errors,
                "scope": (
                    "Static planning validation; not dependency acceptance "
                    "or compute authorization"
                ),
            },
            indent=2,
        )
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
