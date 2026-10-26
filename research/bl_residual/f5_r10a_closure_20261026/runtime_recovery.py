"""Bounded F5 runtime-identity recovery from contemporary metadata only.

This tool intentionally reads neither labels nor scientific outputs.  It projects
only the identity-bearing fields from an explicit source list, and uses pinned
historical source text solely to describe construction and continuation semantics.
It is not a replay, fit, environment probe, or a current-environment substitute.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path("/Users/josenunez/Projects/ml/snake-dqn-f5annex")
ARTIFACTS = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts")
ORIGINS = {
    "d256": {
        "dose": 256,
        "control": ARTIFACTS / "bl-labels-f0/mac-pack-02/d256-control.json",
        "pack": ARTIFACTS / "big-lever/exp/f4/d256-metadata-pack-02/pack.json",
        "pack_control": ARTIFACTS / "big-lever/exp/f4/d256-metadata-pack-02/inputs/control.json",
        "manifest": ARTIFACTS
        / "big-lever/exp/f4/d256-metadata-pack-02/d256-output-l4-01/manifest.json",
        "receipt": ARTIFACTS
        / "big-lever/exp/f4/d256-metadata-pack-02/d256-output-l4-01/receipt.json",
        "commit": "61900af28932e47e22e332840fc4406559adb062",
    },
    "d1024": {
        "dose": 1024,
        "control": ARTIFACTS
        / "bl-labels-f0-speed/d1024-old-w6-equiv-02-pack-01/d1024-control.json",
        "pack": ARTIFACTS / "big-lever/exp/f4/d1024-pack-04/pack.json",
        "pack_control": ARTIFACTS / "big-lever/exp/f4/d1024-pack-04/inputs/control.json",
        "manifest": ARTIFACTS / "big-lever/exp/f4/d1024-pack-04/d1024-output-l4-04/manifest.json",
        "receipt": ARTIFACTS / "big-lever/exp/f4/d1024-pack-04/d1024-output-l4-04/receipt.json",
        "commit": "111c47818a097c57451386ef4f8a7009616f8f77",
    },
}
FORBIDDEN = ("d4096", "t34-round0-study-01", "outcome", "payload", "snapshot", ".npz")
MAX_BYTES = 16 * 1024 * 1024


def canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def safe(path: Path) -> None:
    """Reject protected paths, links, and non-metadata source sizes before a read."""
    resolved = path.absolute()
    lowered = str(resolved).lower()
    if any(token in lowered for token in FORBIDDEN):
        raise ValueError("forbidden path")
    if not resolved.is_file() or resolved.stat().st_size > MAX_BYTES:
        raise ValueError("missing or unbounded metadata")
    if any(part.is_symlink() for part in (resolved, *resolved.parents)):
        raise ValueError("symlink source refused")


def load_json(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    safe(path)
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("metadata must be an object")
    return value, {"path": str(path), "sha256": digest(raw), "bytes": len(raw)}


def receipt_binding(path: Path) -> dict[str, Any]:
    """Bind a generic receipt through its pre-existing SHA sidecar only.

    Receipt JSON may contain scientific export fields, so this function never
    opens, hashes, parses, or deserializes receipt bytes.
    """
    sidecar = Path(str(path) + ".sha256")
    if not sidecar.exists():
        return {
            "path": str(path),
            "sha256_from_sidecar": None,
            "receipt_decoded": False,
            "disposition": "UNBOUND_NO_SIDECAR",
        }
    safe(sidecar)
    fields = sidecar.read_text(encoding="utf-8").split()
    if (
        len(fields) < 1
        or len(fields[0]) != 64
        or any(c not in "0123456789abcdef" for c in fields[0])
    ):
        raise ValueError("malformed receipt sidecar")
    return {"path": str(path), "sha256_from_sidecar": fields[0], "receipt_decoded": False}


def source_text(commit: str, path: str) -> tuple[str, str]:
    """Read source at the historical commit; never execute or import it."""
    result = subprocess.run(
        ["git", "show", f"{commit}:{path}"], cwd=ROOT, check=True, capture_output=True
    )
    raw = result.stdout
    return digest(raw), raw.decode("utf-8")


def roster_rows(
    control: dict[str, Any], watch6: list[dict[str, str]]
) -> tuple[list[dict[str, Any]], str]:
    """Project each contemporary roster without opening checkpoint bytes."""
    rows = []
    for allocation in control["allocation"]:
        plan = allocation.get("plan")
        hero, watch6_opponents = watch6[0], watch6[1:]
        if not isinstance(plan, dict):
            if allocation.get("stream") != "watch6":
                raise ValueError("allocation lacks roster identity")
            specs = [[member["kind"], member["reference"]] for member in watch6_opponents]
            members = [member["sha256"] for member in watch6_opponents]
        else:
            specs, members = plan.get("specs"), plan.get("member_sha256s")
        if (
            not isinstance(specs, list)
            or not isinstance(members, list)
            or len(specs) != len(members)
        ):
            raise ValueError("missing or contradictory ordered roster")
        projected = []
        for spec, member in zip(specs, members):
            if not (
                isinstance(spec, list)
                and len(spec) == 2
                and isinstance(spec[0], str)
                and isinstance(spec[1], str)
                and isinstance(member, str)
                and len(member) == 64
            ):
                raise ValueError("malformed roster identity")
            projected.append({"kind": spec[0], "reference": spec[1], "sha256": member})
        rows.append(
            {
                "allocation_id": allocation["allocation_id"],
                "allocation_index": allocation["allocation_index"],
                "seed": allocation["seed"],
                "stream": allocation["stream"],
                "hero": hero,
                "ordered_opponents": projected,
            }
        )
    rows.sort(key=lambda row: row["allocation_index"])
    return rows, digest(canonical(rows))


def validate_links(
    name: str,
    refs: dict[str, Any],
    pack_record: dict[str, Any],
    pack: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Require all directly represented pack/control/manifest origin links."""
    if pack_record["commit"] != manifest["commit"]:
        raise ValueError(f"{name}: pack and manifest commit differ")
    if pack["doses"] != [refs["dose"]] or manifest["f4"]["dose"] != refs["dose"]:
        raise ValueError(f"{name}: dose linkage differs")
    if "F4_DOSE" in manifest["env"] and manifest["env"]["F4_DOSE"] != str(refs["dose"]):
        raise ValueError(f"{name}: manifest environment dose differs")
    inputs = {row["dest"]: row["sha256"] for row in manifest["inputs"]}
    if inputs.get("b8.pth") != pack["input_hashes"].get("b8.pth"):
        raise ValueError(f"{name}: B8 input linkage differs")
    metadata = inputs.get("metadata.tar")
    if metadata != manifest["env"].get("METADATA_ARCHIVE_SHA256"):
        raise ValueError(f"{name}: metadata archive environment linkage differs")
    ledger = inputs.get("factory-verified-ledger.json")
    if ledger is not None and ledger != pack["input_hashes"].get("factory-verified-ledger.json"):
        raise ValueError(f"{name}: factory ledger linkage differs")
    pack_metadata = pack_record.get("metadata_archive_sha256")
    if pack_metadata is not None and pack_metadata != metadata:
        raise ValueError(f"{name}: pack metadata archive linkage differs")
    return {
        "pack_manifest_commit": manifest["commit"],
        "dose": refs["dose"],
        "b8_sha256": inputs["b8.pth"],
        "metadata_archive_sha256": metadata,
        "pack_metadata_archive_sha256": pack_metadata,
        "pack_metadata_archive_link": (
            "VERIFIED" if pack_metadata == metadata else "NOT_REPRESENTED_IN_PACK_RECORD"
        ),
        "factory_ledger_link": (
            "VERIFIED" if ledger is not None else "NOT_DIRECTLY_IN_RUNTIME_MANIFEST"
        ),
    }


def source_semantics(commit: str) -> dict[str, Any]:
    """State only deterministic behavior visible in pinned constructor source."""
    factory_sha, factory = source_text(commit, "research/bl_labels/factory.py")
    replay_sha, replay = source_text(commit, "research/t1_commitment_20261015/replay.py")
    residual_requirements_sha, residual_requirements = source_text(
        commit, "research/bl_residual/requirements.txt"
    )
    project_requirements_sha, project_requirements = source_text(commit, "requirements.txt")
    replay_audit_sha, replay_audit = source_text(commit, "research/bl_labels/replay_audit.py")
    b2_engine_sha, b2_engine = source_text(commit, "research/bl_served_search/b2_engine.py")
    bank_engine_sha, _ = source_text(commit, "research/t3_rollout_labels_20261015/bank_engine.py")
    collector_sha, _ = source_text(commit, "research/bl_ppo/collector.py")
    ceiling_sha, ceiling = source_text(commit, "research/bl_served_search/ceiling.py")
    frp_match = re.search(r'^FRP3 = "([0-9a-f]{64})"$', b2_engine, re.M)
    b8_match = re.search(r'^B8_SHA256 = "([0-9a-f]{64})"$', ceiling, re.M)
    if not frp_match or not b8_match:
        raise ValueError("pinned source lacks Watch6 checkpoint identities")
    required_factory = (
        "random.seed(seed)",
        "np.random.seed(seed)",
        "torch.manual_seed(seed)",
        'backend="numba"',
    )
    required_replay = ("torch.set_num_threads(1)", "torch.set_num_interop_threads(1)")
    if not all(text in factory for text in required_factory) or not all(
        text in replay for text in required_replay
    ):
        raise ValueError("pinned source no longer establishes declared semantics")
    required_audit = (
        "get_advisory_action_mask",
        "gap <= 4 * ulp",
        "np.finfo(np.float32).tiny",
        "gid.env_digest",
        "len(keyed) < 64",
    )
    if not all(text in replay_audit for text in required_audit):
        raise ValueError("pinned source no longer establishes historical replay tie rule")
    if not all(
        text in b2_engine
        for text in ('"bl-b2-state/v2"', '"game_config"', '"sim"', '"frozen"', '"hero"')
    ):
        raise ValueError("pinned source no longer establishes B2 continuation contract")
    return {
        "constructor_sources": {
            "research/bl_labels/factory.py": factory_sha,
            "research/t1_commitment_20261015/replay.py": replay_sha,
            "research/bl_residual/requirements.txt": residual_requirements_sha,
            "requirements.txt": project_requirements_sha,
            "research/bl_labels/replay_audit.py": replay_audit_sha,
            "research/bl_served_search/b2_engine.py": b2_engine_sha,
            "research/t3_rollout_labels_20261015/bank_engine.py": bank_engine_sha,
            "research/bl_ppo/collector.py": collector_sha,
            "research/bl_served_search/ceiling.py": ceiling_sha,
        },
        "declared_dependency_pins_not_runtime_install_receipts": {
            "collection_requirements": [
                line for line in residual_requirements.splitlines() if line
            ],
            "project_numeric_pins": [
                line
                for line in project_requirements.splitlines()
                if line.startswith(("numba==", "llvmlite=="))
            ],
        },
        "evidenced": {
            "rng_initialization": [
                "python random.seed(allocation seed)",
                "numpy np.random.seed(allocation seed)",
                "torch torch.manual_seed(allocation seed)",
            ],
            "backend": "numba",
            "cpu_thread_contract": "torch intra-op 1; inter-op 1 when permitted",
            "continuation": (
                "continuation restores trusted B2 decoder state; no serialized RNG state "
                "identity is present in admitted metadata"
            ),
            "continuation_source_contract": {
                "source": b2_engine_sha,
                "schema": "bl-b2-state/v2",
                "restored_components": [
                    "game_config",
                    "sim",
                    "frozen",
                    "hero",
                    "decision",
                    "life",
                    "awaiting_respawn",
                ],
                "limitation": (
                    "The original serialized B2 payload/SHA and runtime RNG state "
                    "were not admitted metadata."
                ),
            },
            "source_driver_config_construction_pins": {
                "factory": factory_sha,
                "b2_engine_gate_driver": b2_engine_sha,
                "bank_engine": bank_engine_sha,
                "collector_sim_config": collector_sha,
            },
            "watch6_homogeneous_roster": [
                {"kind": "checkpoint", "reference": "B8 hero", "sha256": b8_match.group(1)},
                *[
                    {
                        "kind": "checkpoint",
                        "reference": "FRP3-s12+v8 opponent",
                        "sha256": frp_match.group(1),
                    }
                    for _ in range(5)
                ],
            ],
            "historical_recorded_action_replay_tie_rule": {
                "scope": "Mac replay of recorded collection actions; not an F5 discrepancy waiver",
                "membership": "source action must be in the Mac advisory action mask",
                "q_gap": (
                    "absolute Q gap <= 4 * max(abs(float32 spacing(max(abs(Q_mac), "
                    "abs(Q_source)))), float32 tiny)"
                ),
                "integrity": (
                    "action tape decision continuity, simulator env digest after every "
                    "decision, and at least 64 distinct roots"
                ),
            },
        },
    }


def origin_identity(name: str, refs: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    control, control_bind = load_json(refs["control"])
    pack_record, pack_record_bind = load_json(refs["pack"])
    pack, pack_bind = load_json(refs["pack_control"])
    manifest, manifest_bind = load_json(refs["manifest"])
    receipt_bind = receipt_binding(refs["receipt"])
    required = {"namespace", "source_commit", "platform", "workers", "allocation"}
    if not required <= set(control) or control["source_commit"] != refs["commit"]:
        raise ValueError(f"{name}: control lacks origin identity")
    if not {"schema", "commit"} <= set(pack_record):
        raise ValueError(f"{name}: pack record lacks identity")
    # Only stable supervisor identity fields are projected from non-receipt metadata.
    for obj, keys, label in (
        (pack, {"namespace", "recipe_sha256", "registration_sha256"}, "pack control"),
        (
            manifest,
            {"commit", "device", "env", "requirements", "script", "f4", "inputs"},
            "manifest",
        ),
    ):
        if not keys <= set(obj):
            raise ValueError(f"{name}: {label} lacks required identity fields")
    linkage = validate_links(name, refs, pack_record, pack, manifest)
    semantics = source_semantics(refs["commit"])
    rows, roster_sha = roster_rows(control, semantics["evidenced"]["watch6_homogeneous_roster"])
    f4 = manifest["f4"]
    fit_config = {key: f4[key] for key in ("dose", "registered_updates", "qualified_tree")}
    safe_inputs = [
        {"dest": row["dest"], "sha256": row["sha256"]}
        for row in manifest["inputs"]
        if row.get("dest") in {"b8.pth", "metadata.tar", "factory-verified-ledger.json"}
    ]
    evidenced = {
        "origin_source_commit": control["source_commit"],
        "collection_namespace": control["namespace"],
        "collection_platform": control["platform"],
        "collection_workers": control["workers"],
        "runtime_device": manifest["device"],
        "runtime_command_environment": manifest["env"],
        "runtime_requirements": manifest["requirements"],
        "runtime_script": manifest["script"],
        "runtime_commit": manifest["commit"],
        "fit_runtime_target": {
            "device": manifest["device"],
            "allowed_gpus": manifest["requirements"]["allowed_gpus"],
        },
        "fit_config_identity": fit_config,
        "fit_safe_input_identities": safe_inputs,
        "terminal_receipt_binding": receipt_bind,
        "pack_record": {"schema": pack_record["schema"], "commit": pack_record["commit"]},
        "verified_pack_control_manifest_links": linkage,
        "ordered_rosters": rows,
        "ordered_rosters_sha256": roster_sha,
        **semantics,
    }
    missing = [
        "CPU model and instruction-set identity",
        "Python version",
        "NumPy version and BLAS/LAPACK backend/version",
        "PyTorch version/build and CUDA/cuDNN version",
        "numba version",
        "serialized original AppConfig / resolved simulator configuration",
        "serialized Python/NumPy/PyTorch RNG state at continuation boundary",
        "exact continuation snapshot identity and restoration state",
    ]
    return {
        "origin": name,
        "disposition": "HOLD_RUNTIME_IDENTITIES_INCOMPLETE",
        "evidenced": evidenced,
        "missing_required_identities": missing,
        "no_current_environment_substitution": True,
    }, [control_bind, pack_record_bind, pack_bind, manifest_bind, receipt_bind]


def publish(path: Path, raw: bytes) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--publish", type=Path, required=True)
    args = parser.parse_args()
    if args.publish.exists():
        raise SystemExit("publish directory already exists")
    identities, bindings = [], []
    roster_shards: dict[str, list[tuple[str, bytes]]] = {}
    for name, refs in ORIGINS.items():
        identity, origin_bindings = origin_identity(name, refs)
        rosters = identity["evidenced"].pop("ordered_rosters")
        chunks = [rosters[index : index + 1000] for index in range(0, len(rosters), 1000)]
        names = (
            [f"{name}-ordered-rosters.json"]
            if len(chunks) == 1
            else [f"{name}-ordered-rosters-{index:03d}.json" for index in range(len(chunks))]
        )
        roster_shards[name] = [
            (filename, canonical(chunk)) for filename, chunk in zip(names, chunks)
        ]
        identity["evidenced"]["ordered_roster_files"] = [
            {"path": filename, "sha256": digest(raw)} for filename, raw in roster_shards[name]
        ]
        identities.append(identity)
        bindings.extend(origin_bindings)
    report = {
        "schema": "f5-r10a-runtime-identities/v1",
        "disposition": "HOLD_RUNTIME_IDENTITIES_INCOMPLETE",
        "access_attestation": {
            "outcome_payloads_opened_or_hashed": False,
            "d4096_accessed": False,
            "t34_round0_study_01_accessed": False,
            "current_environment_used_as_evidence": False,
            "constructors_executed": False,
            "scientific_work_executed": False,
        },
        "source_bindings": bindings,
        "safe_search_surfaces": {
            "examined": [
                "collection control",
                "F4 pack record",
                "F4 pack control",
                "F4 supervisor manifest",
                "F4 supervisor terminal receipt SHA sidecar only (receipt bytes never decoded)",
                "historical pinned constructor and setup requirements source",
            ],
            "excluded_without_opening": [
                (
                    "life/session records, root caches, labels, NPZs, checkpoints, "
                    "snapshots, exports, outcomes, D4096, t34-round0-study-01"
                ),
            ],
            "result": (
                "No contemporaneous installed-version capture was found. The original "
                "recorded-action replay tie rule was recovered from both pinned source origins."
            ),
        },
        "origins": identities,
        "required_consequence": (
            "No restore/parity/canary or adoption follows from this metadata recovery."
        ),
    }
    body = canonical(report)
    summary = "\n".join(
        [
            "# F5 R10a runtime identity recovery",
            "",
            "Disposition: **HOLD_RUNTIME_IDENTITIES_INCOMPLETE**.",
            "",
            "Both origins bind source commits, collection platform and runtime manifest fields,",
            "plus ordered hero/opponent identities and the historical replay tie rule.",
            "Installed versions, CPU identity and serialized continuation state remain absent.",
            "No current environment was substituted. No payload or protected source was opened;",
            "no constructor, fit or replay was executed.",
            "",
            f"JSON SHA256: `{digest(body)}`",
            "",
        ]
    ).encode()
    for entries in roster_shards.values():
        for filename, raw in entries:
            publish(args.publish / filename, raw)
            publish(args.publish / f"{filename}.sha256", (digest(raw) + f"  {filename}\n").encode())
    publish(args.publish / "runtime-identities.json", body)
    publish(
        args.publish / "runtime-identities.json.sha256",
        (digest(body) + "  runtime-identities.json\n").encode(),
    )
    publish(args.publish / "runtime-identities.md", summary)
    publish(
        args.publish / "runtime-identities.md.sha256",
        (digest(summary) + "  runtime-identities.md\n").encode(),
    )
    publish(args.publish / "runtime-identities.complete", b"complete\n")


if __name__ == "__main__":
    main()
