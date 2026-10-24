"""Mac rowwise fused-checkpoint verification and the single R30 terminal look."""

from __future__ import annotations

import argparse
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
from research.bl_round30.protocol import JUDGMENT_SHA, require_successor_gate
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
    if admitted.allocation_sha256 != binding["corpus_allocation"]:
        raise ValueError("INVALID: foreign corpus allocation")
    if binding.get("salt") != admitted.salt:
        raise ValueError("INVALID: foreign split salt")
    if (
        binding.get("ruling_sha256") != JUDGMENT_SHA
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
    if inventory.get("partition_sha256", {}).get(partition) != digest(table_path):
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
        for row in rows:
            row["control_action"] = row["lbfgs_action"]
        result["conversion"] = conversion_build_trigger(rows)
        # Optimizer resolution comes only from the certified objective + actual fp32 actions.
        result["served_action_disagreement"] = float(
            np.mean([row["adam_action"] != row["lbfgs_action"] for row in audit_rows])
        )
        result["label"] = "HOLD"  # filled by the explicit certificate/retention reducer below
        from research.bl_round30.objective import optimizer_decision

        result.update(optimizer_decision(diagnostics, audit_rows))
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
