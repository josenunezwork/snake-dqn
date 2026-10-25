"""Mac rowwise fused-checkpoint verification and the single R30 terminal look."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from research.bl_ppo.exports import digest, export_json, verify_export
from research.bl_residual.data import validate_rows
from research.bl_round30.metrics import (
    conversion_build_trigger,
    heldout_reducer,
    served_selection,
    summarize_training_rows,
)
from research.bl_round30.protocol import (
    AMENDMENT_SHA,
    JUDGMENT_SHA,
    require_amendment_binding,
    require_successor_gate,
)
from src.model.ego2s_network import Ego2sNet


def rows_from_table(table: dict) -> list[dict]:
    """Preserve original row identities and sentinel values for the pure reducer."""
    rows = []
    for index in range(len(table["q"])):
        row = {
            key: value[index].tolist()
            for key, value in table.items()
            if key not in ("local", "global", "scalars")
        }
        row.update(horizon=300, base_action=int(table["bare"][index]))
        rows.append(row)
    return rows


def rowwise_q(endpoint: dict, table: dict) -> np.ndarray:
    """Reload exported fused fp32 weights; never infer actions from float64 fits."""
    net = Ego2sNet(**endpoint["config"]).cpu().eval()
    net.load_state_dict(endpoint["state_dict"])
    outputs = []
    with torch.no_grad():
        for index in range(len(table["q"])):
            inputs = [
                torch.as_tensor(table[key][index : index + 1])
                for key in ("local", "global", "scalars")
            ]
            outputs.append(net(*inputs).numpy()[0])
    q = np.asarray(outputs)
    if not np.isfinite(q).all():
        raise ValueError("INVALID: nonfinite fused Q")
    return q


def endpoint_sha256(endpoint: dict) -> str:
    """Hash one exported endpoint independent of its enclosing worker unit."""
    state = endpoint.get("state_dict")
    config = endpoint.get("config")
    if not isinstance(state, dict) or not isinstance(config, dict):
        raise ValueError("INVALID: endpoint lacks config/state_dict")
    result = hashlib.sha256()
    result.update(json.dumps(config, sort_keys=True, allow_nan=False).encode())
    for name in sorted(state):
        value = state[name]
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise ValueError("INVALID: malformed endpoint state")
        tensor = value.detach().cpu().contiguous()
        result.update(name.encode() + b"\0")
        result.update(str(tensor.dtype).encode() + b"\0")
        result.update(json.dumps(list(tensor.shape)).encode() + b"\0")
        result.update(tensor.numpy().tobytes())
    return result.hexdigest()


def parameters_sha256(endpoint: dict) -> str:
    """Bind the unfused residual fit used to initialize a converged refit."""
    parameters = endpoint.get("parameters")
    if not isinstance(parameters, torch.Tensor):
        raise ValueError("INVALID: endpoint lacks residual parameters")
    tensor = parameters.detach().cpu().contiguous()
    result = hashlib.sha256()
    result.update(str(tensor.dtype).encode() + b"\0")
    result.update(json.dumps(list(tensor.shape)).encode() + b"\0")
    result.update(tensor.numpy().tobytes())
    return result.hexdigest()


def require_global_life_identity(table: dict, life_id_map: dict | None = None) -> None:
    """Refuse tranche-local life IDs before a reducer can aggregate them."""
    source_life_ids = table.get("source_life_id")
    global_life_indices = table.get("global_life_index")
    life_ids = table.get("life_id")
    parents = table.get("parent_world")
    if (
        source_life_ids is None
        or global_life_indices is None
        or life_ids is None
        or parents is None
    ):
        raise ValueError("INVALID: globally unique source life IDs/map missing")
    row_count = len(table["q"])
    if any(
        len(values) != row_count
        for values in (source_life_ids, global_life_indices, life_ids, parents)
    ):
        raise ValueError("INVALID: global life identity arrays have unequal lengths")
    if (
        not isinstance(source_life_ids, np.ndarray)
        or source_life_ids.dtype.kind not in "US"
        or not isinstance(life_ids, np.ndarray)
        or life_ids.dtype.kind not in "US"
        or not isinstance(global_life_indices, np.ndarray)
        or global_life_indices.dtype.kind not in "iu"
        or global_life_indices.dtype.kind == "b"
        or np.any(global_life_indices < 0)
    ):
        raise ValueError("INVALID: global life identity arrays have invalid dtypes")
    frozen_ids = None if life_id_map is None else life_id_map.get("ids")
    if frozen_ids is not None and not isinstance(frozen_ids, dict):
        raise ValueError("INVALID: life ID map is malformed")
    source_to_identity: dict[object, tuple[object, object, int]] = {}
    for source_life, global_index, parent, life in zip(
        source_life_ids.tolist(),
        global_life_indices.tolist(),
        table["parent_world"].tolist(),
        table["life_id"].tolist(),
    ):
        if source_life != life or not isinstance(global_index, (int, np.integer)):
            raise ValueError(
                "INVALID: global life identity/map schema differs from materialization"
            )
        identity = (parent, life, int(global_index))
        if source_life in source_to_identity and source_to_identity[source_life] != identity:
            raise ValueError("INVALID: source life ID maps to multiple identities")
        source_to_identity[source_life] = identity
        if frozen_ids is not None and frozen_ids.get(source_life) != int(global_index):
            raise ValueError("INVALID: global life ID differs from frozen map")
    if len(source_to_identity) != len(
        {(parent, life) for parent, life, _ in source_to_identity.values()}
    ):
        raise ValueError("INVALID: tranche-local parent/life identity collision")
    if len(source_to_identity) != len({index for _, _, index in source_to_identity.values()}):
        raise ValueError("INVALID: global life integer map collision")


def resolved_conversion(
    rows: list[dict], optimizer_label: str, endpoint_hashes: dict[str, str]
) -> dict:
    """Run the decision-bearing conversion audit against the resolved control only."""
    control_name = {
        "OPTIMIZER_ADEQUATE": "adam",
        "OPTIMIZER_MATERIAL": "lbfgs",
    }.get(optimizer_label)
    if control_name is None:
        return {"label": "HOLD", "reason": "optimizer_unresolved", "control": None}
    endpoint_sha = endpoint_hashes.get(control_name)
    if not isinstance(endpoint_sha, str) or len(endpoint_sha) != 64:
        raise ValueError("INVALID: resolved optimizer endpoint digest missing")
    controlled_rows = [{**row, "control_action": row[control_name + "_action"]} for row in rows]
    conversion = conversion_build_trigger(controlled_rows)
    conversion["control"] = {
        "name": control_name,
        "procedure": control_name,
        "endpoint_sha256": endpoint_sha,
    }
    return conversion


def analyze(
    unit_path: Path, table_path: Path, binding_path: Path, output: Path, *, stage: str
) -> dict:
    """Bind endpoints/analyzer/split before opening outcomes; create the look once."""
    if torch.get_num_threads() != 1:
        torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    binding = json.loads(binding_path.read_text())
    repo = Path(__file__).resolve().parents[2]
    source_paths = [
        "research/bl_round30",
        "research/bl_residual/data.py",
        "research/bl_residual/model.py",
        "src/model/ego2s_network.py",
    ]
    source_diff = subprocess.check_output(
        ["git", "diff", binding["source_commit"], "--", *source_paths], cwd=repo, timeout=10
    )
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "--", *source_paths],
        cwd=repo,
        timeout=10,
    )
    if source_diff or untracked:
        raise ValueError("INVALID: current reducer/source closure differs from bound pin")
    from research.bl_round30.admission import freeze_inventory, require_codex_v_approval

    inventory_path, approval_path = Path(binding["inventory_path"]), Path(binding["approval_path"])
    if (
        digest(inventory_path) != binding["inventory_sha256"]
        or digest(approval_path) != binding["approval_sha256"]
    ):
        raise ValueError("INVALID: stale corpus approval")
    inventory = json.loads(inventory_path.read_text())
    admitted = freeze_inventory(inventory)
    require_codex_v_approval(admitted, json.loads(approval_path.read_text()))
    life_id_map_path = Path(binding["life_id_map_path"])
    if digest(life_id_map_path) != binding.get("life_id_map_sha256"):
        raise ValueError("INVALID: stale global life ID map")
    life_id_map = json.loads(life_id_map_path.read_text())
    if (
        life_id_map.get("schema") != "r30-global-life-map/v1"
        or life_id_map.get("inventory_sha256") != admitted.inventory_sha256
    ):
        raise ValueError("INVALID: foreign global life ID map")
    partition_manifest_path = Path(binding["partition_manifest_path"])
    coverage_approval_path = Path(binding["coverage_approval_path"])
    if digest(partition_manifest_path) != binding.get("partition_manifest_sha256") or digest(
        coverage_approval_path
    ) != binding.get("coverage_approval_sha256"):
        raise ValueError("INVALID: stale partition manifest/coverage approval")
    partition_manifest = json.loads(partition_manifest_path.read_text())
    coverage_approval = json.loads(coverage_approval_path.read_text())
    if (
        partition_manifest.get("schema") != "r30-partition-manifest/v1"
        or partition_manifest.get("inventory_sha256") != admitted.inventory_sha256
        or partition_manifest.get("allocation_sha256") != admitted.allocation_sha256
        or partition_manifest.get("life_id_map_sha256") != binding.get("life_id_map_sha256")
        or coverage_approval.get("schema") != "r30-partition-coverage-approval/v1"
        or coverage_approval.get("approved") is not True
        or coverage_approval.get("reviewer_role") != "Codex-V"
        or coverage_approval.get("partition_manifest_sha256")
        != binding.get("partition_manifest_sha256")
    ):
        raise ValueError("INVALID: foreign partition manifest/coverage approval")
    if admitted.allocation_sha256 != binding["corpus_allocation"]:
        raise ValueError("INVALID: foreign corpus allocation")
    if binding.get("salt") != admitted.salt:
        raise ValueError("INVALID: foreign split salt")
    try:
        require_amendment_binding(binding)
    except ValueError as exc:
        raise ValueError("INVALID: foreign R30 authority/amendment binding") from exc
    if (
        binding.get("ruling_sha256") != JUDGMENT_SHA
        or binding.get("amendment_sha256") != AMENDMENT_SHA
        or binding.get("analyzer_sha256") != digest(Path(__file__))
        or binding.get("unit_sha256") != verify_export(unit_path)
        or binding.get("table_sha256") != digest(table_path)
        or binding.get("stage") != stage
        or binding.get("partition") != ("train-audit" if stage == "r30-1" else "validation")
        or binding.get("mac_rowwise_admission") is not True
    ):
        raise ValueError("INVALID: stale analyzer/endpoints/partition binding")
    unit = torch.load(unit_path, map_location="cpu", weights_only=False)
    if (
        unit.get("synthetic") is not False
        or unit.get("ruling_sha256") != JUDGMENT_SHA
        or unit.get("amendment_sha256") != AMENDMENT_SHA
        or unit.get("heldout_outcomes_opened") is not False
        or unit.get("allocation") != binding.get("allocation")
        or unit.get("source_commit") != binding.get("source_commit")
        or unit.get("stage") != stage
    ):
        raise ValueError("INVALID: foreign endpoint")
    if stage == "r30-2":
        diagnosis_path = Path(binding["diagnosis_path"])
        from research.bl_round30.protocol import bound_json

        diagnosis = bound_json(diagnosis_path, binding["successor_gate"]["diagnosis_sha256"])
        require_successor_gate(binding["successor_gate"], binding["corpus_allocation"], diagnosis)
    # This create-only record is the decision-bearing look fence. No second tune/look.
    # Global canonical ID depends on the frozen corpus/stage, never the caller's output name.
    look_root = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-30/looks")
    look_path = look_root / (admitted.allocation_sha256 + "-" + stage + ".json")
    if binding.get("look_path") != str(look_path):
        raise ValueError("INVALID: noncanonical look binding")
    export_json(
        look_path,
        {
            "schema": "r30-one-look/v1",
            "binding_sha256": digest(binding_path),
            "unit_sha256": digest(unit_path),
            "stage": stage,
        },
    )
    with np.load(table_path, allow_pickle=False) as archive:
        table = {key: archive[key] for key in archive.files}
    validate_rows(table)
    require_global_life_identity(table, life_id_map)
    from research.bl_round30.admission import split_parent

    partition = "train" if stage == "r30-1" else "validation"
    if any(
        split_parent(str(parent), binding["salt"]) != partition for parent in table["parent_world"]
    ):
        raise ValueError("INVALID: split leakage")
    expected_parents = set(
        admitted.train_parents if stage == "r30-1" else admitted.validation_parents
    )
    actual_parents = set(table["parent_world"].tolist())
    if not actual_parents <= expected_parents:
        raise ValueError("INVALID: foreign parent")
    if actual_parents != expected_parents:
        raise ValueError("INCOMPLETE: allocated parents missing")
    if partition_manifest.get("partitions", {}).get(partition, {}).get("sha256") != digest(
        table_path
    ):
        raise ValueError("INVALID: outcomes outside frozen inventory")
    if stage == "r30-2" and (
        len(admitted.validation_parents) < 32 or len(admitted.validation_rescue_parents) < 16
    ):
        raise ValueError("INSUFFICIENT: validation parent support")
    rows = rows_from_table(table)
    q_values = {}
    for name, endpoint in unit["endpoints"].items():
        q = rowwise_q(endpoint, table)
        q_values[name] = q
        for row, values in zip(rows, q):
            row[name + "_q"] = values.tolist()
            row[name + "_action"] = served_selection(row, name + "_q")
    if stage == "r30-1":
        diagnostics = unit["diagnostics"]
        audit_rows = [rows[i] for i in diagnostics["audit_indices"]]
        audit_multiplicities = Counter(diagnostics["audit_indices"])
        unique_audit_rows = [rows[i] for i in audit_multiplicities]
        result = {
            "stage": stage,
            "diagnostics": diagnostics,
            "audit_multiplicities": dict(audit_multiplicities),
            "training": {
                name: summarize_training_rows(unique_audit_rows, name + "_q") for name in q_values
            },
        }
        # Optimizer resolution comes only from the certified objective + actual fp32 actions.
        result["served_action_disagreement"] = float(
            np.mean([row["adam_action"] != row["lbfgs_action"] for row in audit_rows])
        )
        result["label"] = "HOLD"  # filled by the explicit certificate/retention reducer below
        from research.bl_round30.objective import optimizer_decision

        result.update(optimizer_decision(diagnostics, audit_rows))
        result["conversion"] = resolved_conversion(
            rows,
            result["label"],
            {name: endpoint_sha256(endpoint) for name, endpoint in unit["endpoints"].items()},
        )
        control = result["conversion"].get("control")
        if control is not None:
            name = control["name"]
            binding = diagnostics.get("convergence_bindings", {}).get(name)
            if not isinstance(binding, dict):
                raise ValueError("INVALID: resolved convergence binding is missing")
            control["parameters_sha256"] = parameters_sha256(unit["endpoints"][name])
            control["convergence_binding"] = binding
    else:
        # Archive membership is frozen independently and includes nonrescue parents.
        archive = [row for row in rows if row.get("ordinary") and row.get("natural_stream")]
        archive_meta = inventory.get("natural_ordinary_archive", {})
        permitted_archive = set(archive_meta.get("ultimate_parent_ids", []))
        probabilities = archive_meta.get("sampling_probabilities", {})
        for row in archive:
            parent = row["parent_world"]
            catalog = {p["ultimate_parent_id"]: p for p in inventory["parents"]}
            row["declared_nonrescue"] = not catalog[parent]["rescue_bearing"]
            if parent not in permitted_archive or row.get("sampling_prob") != float(
                np.float32(probabilities[parent])
            ):
                raise ValueError("INVALID: archive/probability differs from frozen allocation")
        actual_archive = {row["parent_world"] for row in archive}
        if actual_archive != permitted_archive:
            raise ValueError("INCOMPLETE: frozen natural archive parents missing")
        rescue = [row for row in rows if not row.get("ordinary")]
        result = heldout_reducer(rescue, natural_rows=archive)
    result.update(
        schema="r30-mac-result/v1",
        ruling_sha256=JUDGMENT_SHA,
        amendment_sha256=AMENDMENT_SHA,
        corpus_allocation=admitted.allocation_sha256,
        binding_sha256=digest(binding_path),
        development_only=True,
        gameplay_funded=False,
        release_authorized=False,
    )
    export_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("r30-1", "r30-2"), required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.unit, args.table, args.binding, args.output, stage=args.stage)))


if __name__ == "__main__":
    main()
