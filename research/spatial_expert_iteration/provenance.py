"""Mac-side resolution of all ancestor bytes and effective seed domains."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from research.spatial_student_pilot.config import PilotConfig
from research.spatial_student_pilot.prepare import digest

from .config import RoundConfig, identity


def resolve(ref: dict[str, Any]) -> Path:
    """Require a concrete file and exact sha256, never a truthy placeholder."""
    if not isinstance(ref, dict) or not {"path", "sha256"}.issubset(ref):
        raise ValueError("provenance requires path/sha256 bindings")
    path = Path(ref.get("path", ""))
    if not path.is_absolute() or not path.is_file() or digest(path) != ref.get("sha256"):
        raise ValueError("provenance file missing or hash mismatch")
    return path.resolve()


def audit_ancestors(cfg: RoundConfig, inventory: dict[str, Any]) -> dict[str, Any]:
    """Walk a hash-bound DAG, validating config/input/seed/round relationships.

    Pilot roots bind the actual supervisor manifest plus every input. Round
    roots bind FREEZE and resolve all sibling inputs, configs and inventories.
    The report is copied into the current packet, so remote runtimes do not
    dereference Mac paths. Historical source hashes remain commit identities.
    """
    refs = inventory.get("prior_round_freezes")
    if not isinstance(refs, list) or not refs:
        raise ValueError("resolved pilot and prior round freezes required")
    active, visited, rounds, seeds, nodes = set(), {}, {}, set(), []
    pilot_count = 0
    closures = {}

    def walk(ref: dict[str, Any]) -> dict[str, Any]:
        nonlocal pilot_count
        path = resolve(ref)
        key = str(path)
        if key in active:
            raise ValueError("provenance DAG cycle")
        if key in visited:
            if visited[key] != identity(ref):
                raise ValueError("conflicting provenance binding")
            return closures[key]
        active.add(key)
        value = json.loads(path.read_text())
        files = {key: ref["sha256"]}
        if ref.get("kind") == "pilot":
            # An operator supplies relocated Mac copies of manifest inputs;
            # every supplied binding must match the original manifest hash.
            bindings = ref.get("inputs")
            if not isinstance(bindings, dict) or not isinstance(value.get("inputs"), list):
                raise ValueError("pilot manifest input bindings required")
            expected = {v["dest"]: v["sha256"] for v in value["inputs"]}
            if len(expected) != len(value["inputs"]) or set(expected) != set(bindings):
                raise ValueError("pilot manifest input population mismatch")
            for name, binding in bindings.items():
                file = resolve(binding)
                if digest(file) != expected[name]:
                    raise ValueError("pilot manifest input hash mismatch")
                files[str(file)] = digest(file)
            pcfg = PilotConfig.from_dict(json.loads(resolve(bindings["config.json"]).read_text()))
            if pcfg.tiny or pcfg.mode != "pilot":
                raise ValueError("scientific ancestry requires registered pilot")
            keys = [
                fn(i, phase)
                for phase, n in (("collection", pcfg.worlds), ("gameplay", pcfg.gameplay_worlds))
                for i in range(n)
                for fn in (pcfg.world_seed, pcfg.opp_seed)
            ]
            pilot_count += 1
            node = dict(
                kind="pilot",
                namespace=pcfg.namespace,
                files=files,
                training_parents={
                    f"{pcfg.namespace}/collection/w{i}": [pcfg.world_seed(i), pcfg.opp_seed(i)]
                    for i, part in enumerate(pcfg.partitions())
                    if part == "train"
                },
            )
            closure = dict(rounds=set(), pilots={key}, seeds=set(keys))
        elif ref.get("kind") == "round":
            if value.get("schema") != "spatial-exit-freeze/v1" or value.get("tiny") is not False:
                raise ValueError("foreign prior round freeze")
            config_path = path.parent / "config.json"
            prior = RoundConfig.from_dict(json.loads(config_path.read_text()))
            index = prior.round_index
            if (
                prior.program != cfg.program
                or not 1 <= index < cfg.round_index
                or value.get("round_index") != index
                or value.get("config_sha256") != prior.sha256
                or value.get("namespace") != prior.namespace
                or value.get("allocation_index") != prior.allocation_index
            ):
                raise ValueError("prior round identity mismatch")
            if index in rounds and rounds[index] != ref["sha256"]:
                raise ValueError("conflicting prior round allocation")
            source_hashes = value.get("source_sha256")
            if not isinstance(source_hashes, dict) or prior.implementation_sha256 != identity(
                source_hashes
            ):
                raise ValueError("ancestor source closure mismatch")
            for name, expected_hash in source_hashes.items():
                source = subprocess.check_output(
                    ["git", "show", f"{value['source_commit']}:{name}"],
                    cwd=Path(__file__).resolve().parents[2],
                )
                if hashlib.sha256(source).hexdigest() != expected_hash:
                    raise ValueError("ancestor source hash mismatch")
            rounds[index] = ref["sha256"]
            files[str(config_path)] = digest(config_path)
            for name, expected_hash in value["inputs"].items():
                if Path(name).name != name:
                    raise ValueError("unsafe ancestor input path")
                file = resolve(dict(path=str(path.parent / name), sha256=expected_hash))
                files[str(file)] = expected_hash
            for name, field in (
                ("verified-ledger.json", "ledger_sha256"),
                ("seed-audit.json", "seed_audit_sha256"),
            ):
                file = resolve(dict(path=str(path.parent / name), sha256=value[field]))
                files[str(file)] = digest(file)
            ancestor_inventory = json.loads((path.parent / "seed-inventory.json").read_text())
            parents = ancestor_inventory.get("prior_round_freezes")
            if not isinstance(parents, list) or not parents:
                raise ValueError("missing ancestor provenance DAG")
            parent_rounds, parent_pilots, parent_keys = set(), set(), set()
            for parent in parents:
                closure = walk(parent)
                parent_rounds.update(closure["rounds"])
                parent_pilots.update(closure["pilots"])
                parent_keys.update(closure["seeds"])
            if parent_rounds != set(range(1, index)) or len(parent_pilots) != 1:
                raise ValueError("ancestor DAG omits or misorders prior rounds")
            if not parent_keys.issubset(set(ancestor_inventory.get("excluded_seeds", []))):
                raise ValueError("ancestor inventory omits predecessor effective seeds")
            from .prepare import seed_keys

            keys = seed_keys(prior)
            audited = json.loads((path.parent / "seed-audit.json").read_text())
            if (
                audited.get("config_sha256") != prior.sha256
                or audited.get("effective_keys") != keys
                or audited.get("collision_count") != 0
                or audited.get("inventory_sha256") != identity(ancestor_inventory)
            ):
                raise ValueError("ancestor seed audit mismatch")
            node = dict(
                kind="round",
                round_index=index,
                namespace=prior.namespace,
                files=files,
                training_parents={
                    f"{prior.namespace}/collection/w{i}": [prior.world_seed(i), prior.opp_seed(i)]
                    for i in range(prior.anchor_worlds, prior.worlds)
                },
            )
            closure = dict(
                rounds=parent_rounds | {index}, pilots=parent_pilots, seeds=parent_keys | set(keys)
            )
        else:
            raise ValueError("unknown provenance node kind")
        if len(keys) != len(set(keys)) or seeds.intersection(keys):
            raise ValueError("ancestor seed collision")
        seeds.update(keys)
        nodes.append(node)
        visited[key] = identity(ref)
        active.remove(key)
        closures[key] = closure
        return closure

    for ref in refs:
        walk(ref)
    if pilot_count != 1 or set(rounds) != set(range(1, cfg.round_index)):
        raise ValueError("provenance must include one pilot and every preceding round")
    if not seeds.issubset(set(inventory.get("excluded_seeds", []))):
        raise ValueError("ancestor effective seeds missing from exclusion inventory")
    return dict(
        schema="spatial-exit-provenance-audit/v1",
        nodes=nodes,
        prior_rounds=rounds,
        effective_keys=sorted(seeds),
        collision_count=0,
    )
