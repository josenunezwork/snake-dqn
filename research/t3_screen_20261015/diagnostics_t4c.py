"""CPU-only final-bank and T4c drift readouts.

The evaluator never trains or writes checkpoints.  It evaluates final models on
the immutable registered banks with raw six-action argmax, and it treats the
already-published T4c drift summary as contextual evidence rather than trying
to follow its historical source paths.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

for _key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_key] = "1"
os.environ["SNAKE_DQN_DEVICE"] = "cpu"

import numpy as np  # noqa: E402

from research.t2_screen_20261015.analyze_t2 import _mean  # noqa: E402
from research.t3_screen_20261015.analyze_t4c import ARM, SEEDS  # noqa: E402

ARTIFACTS = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts")
DATA_ROOT = ARTIFACTS / "t4c-data-20261016"
DRIFT_SUMMARY = ARTIFACTS / "t4c-drift-20261016/drift-summary-verified.json"
TRACKED_DRIFT_SUMMARY = (
    Path(__file__).resolve().parents[1] / "t4c_20261016/results/drift_summary_verified.json"
)
LAUNCH_INPUTS = Path(__file__).resolve().parents[1] / "t4c_20261016/launch_inputs.json"


def _records(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text().splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("transitions"), (int, float)):
            rows.append(value)
    return rows


def json_safe(value: Any) -> Any:
    """Replace nonfinite telemetry values with null for strict result artifacts."""
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def read_telemetry(roots: Mapping[str, Path]) -> dict[str, Any]:
    """Retain final post-7.9M telemetry records without inventing missing values."""
    out: dict[str, Any] = {}
    for name, root in roots.items():
        path = Path(root) / "log.jsonl"
        if not path.exists():
            out[name] = {"final_100k": [], "final": None}
            continue
        records = _records(path)
        final_window = [row for row in records if float(row["transitions"]) >= 7_900_000]
        out[name] = {
            "source": str(path),
            "final_100k": final_window,
            "final": records[-1] if records else None,
        }
    return json_safe(out)


def _obs(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    from research.redesign_m3c_20261008.obs_pack import train_layout, unpack_ref

    local, global_ = unpack_ref(np.asarray(arrays["packed"]), train_layout())
    return json_safe(
        {
            "local": local,
            "global": global_,
            "scalars": np.asarray(arrays["scalars"], dtype=np.float32),
        }
    )


def _actions(checkpoint: Path, arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    """Compute raw six-action first-argmax on one registered immutable bank."""
    import torch

    from src.model.ego2s_network import load_ego2s_checkpoint
    from src.simd_env.ego2s_policy import forward_q

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    net = load_ego2s_checkpoint(str(checkpoint), "cpu").eval()
    obs = _obs(arrays)
    # Raw first-argmax must preserve the registered rowwise convention, so do
    # not permit batch-sensitive float ties to alter a target readout.
    chunks = []
    for start in range(0, len(obs["local"]), 512):
        piece = {key: value[start : start + 512] for key, value in obs.items()}
        chunks.append(forward_q(net, piece, "rowwise").numpy())
    return np.concatenate(chunks).astype(np.float64).argmax(axis=1)


def _bank(path: Path) -> dict[str, np.ndarray]:
    from research.t4c_20261016.data import bound_files, load_arrays, sha256

    bound = bound_files(path)
    approved = json.loads(LAUNCH_INPUTS.read_text())["files"]
    for name, digest in bound.items():
        full = str(path / name)
        if approved.get(full) != digest or sha256(path / name) != digest:
            raise ValueError(f"T4c immutable bank binding mismatch: {full}")
    return load_arrays(path / "bank-00000.npz")


def _family_agreement(checkpoint: Path) -> dict[str, Any]:
    retained, outside = _bank(DATA_ROOT / "bank-v3"), _bank(DATA_ROOT / "outside-bank")
    action = _actions(checkpoint, retained)
    row_family, row_weight = retained["row_family"], retained["row_weight"]
    decisive = row_family == 0
    target = retained["surviving"][np.arange(len(action)), action]
    groups = {
        "retained_decisive_all": decisive,
        "retained_decisive_gap_lt100": decisive & (row_weight == 1.0),
        "retained_decisive_gap_100_300": decisive & (row_weight == 0.3),
        "in_window_counter": row_family == 1,
    }
    result = {}
    for name, take in groups.items():
        hits = (
            target[take]
            if name.startswith("retained_decisive")
            else action[take] == retained["action"][take]
        )
        result[name] = {
            "rows": int(take.sum()),
            "agreement": None if not len(hits) else float(hits.mean()),
        }
    outside_action = _actions(checkpoint, outside)
    result["outside_counter"] = {
        "rows": int(len(outside_action)),
        "agreement": float((outside_action == outside["action"]).mean()),
    }
    return result


def _family_means(per_seed: Mapping[str, Any]) -> dict[str, Any]:
    """Report unweighted five-seed family means while preserving null support."""
    names = (
        "retained_decisive_all",
        "retained_decisive_gap_lt100",
        "retained_decisive_gap_100_300",
        "in_window_counter",
        "outside_counter",
    )
    output = {}
    for name in names:
        values = [item[name]["agreement"] for item in per_seed.values() if item is not None]
        values = [value for value in values if value is not None]
        output[name] = {"mean": _mean(values) if values else None, "n": len(values)}
    return output


def _telemetry_summary(telemetry: Mapping[str, Any]) -> dict[str, Any]:
    keys = ("bank_agree_decisive", "bank_agree_counter", "bank_agree_outside")
    per_run, means = {}, {}
    for name, item in telemetry.items():
        if not name.startswith(ARM + "_s"):
            continue
        rows = item.get("final_100k", [])
        per_run[name] = {}
        for key in keys:
            values = [float(row[key]) for row in rows if row.get(key) is not None]
            per_run[name][key] = _mean(values) if values else None
    for key in keys:
        values = [item[key] for item in per_run.values() if item[key] is not None]
        means[key] = {"mean": _mean(values) if values else None, "n": len(values)}
    return {"per_run": per_run, "five_seed_means": means}


def load_drift_summary(path: Path = DRIFT_SUMMARY) -> dict[str, Any] | None:
    """Load the verified historical summary without chasing bound source paths."""
    if not Path(path).exists():
        return None
    from research.t4c_20261016.data import sha256

    if sha256(path) != sha256(TRACKED_DRIFT_SUMMARY):
        raise ValueError("T4c drift summary differs from the tracked verified result")
    value = json.loads(Path(path).read_text())
    if value.get("schema") != "t4c-drift-report/v2":
        raise ValueError("unexpected T4c drift summary schema")
    return value


def drift_context(
    summary: Mapping[str, Any] | None, candidate: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    if not summary:
        return {
            "baseline": None,
            "candidate": None,
            "within_ten_percentage_points": None,
            "note": "T4c drift baseline summary is unavailable.",
        }
    baseline = summary["rounds"]["62"]["arms"]["B8"]["five_seed_mean"]["consequential_change_rate"]
    candidate_rate = (
        None if candidate is None else candidate.get("candidate_consequential_change_rate")
    )
    return {
        "baseline": baseline,
        "candidate": candidate_rate,
        "within_ten_percentage_points": (
            None if candidate_rate is None else abs(float(candidate_rate) - float(baseline)) <= 0.10
        ),
        "note": "Baseline is B8 versus same-init CTRL on historical r62 states.",
    }


def declared_fork(gameplay: Mapping[str, Any], drift: Mapping[str, Any]) -> str:
    data = gameplay["arms"][ARM]
    if data["decision"] == "KILL" and data["failed_runs"]:
        return "No declared fork assigned: a nonzero training exit is a registered KILL."
    survival = data["vs_ctrl"]["scripted"]["episode_survival"]["mean"]
    mass_ok = all(
        data["vs_ctrl"][mix]["mass"]["mean"] >= -30 for mix in ("scripted", "frozen", "mixed")
    )
    fell = drift.get("within_ten_percentage_points")
    if fell is None:
        return (
            "No declared fork assigned: T4c-final consequential-change rate is not yet available."
        )
    if fell and data["decision"] == "GO" and survival > 0 and mass_ok:
        return (
            "(c) falls to the drift baseline and round-65 survival moves without a mass fail: "
            "expert iteration with the tightened recipe is next, as planned."
        )
    if fell:
        if survival > 0 and not mass_ok:
            return "No declared fork assigned: survival moved but a registered mass guard failed."
        if survival > 0:
            return (
                "UNRESOLVED: (c) is within the drift band, but the final survival screen is "
                "inconclusive. The preregistration does not define 'survival moves' for an "
                "intermediate outcome, so neither survival-dependent fork is established. "
                "The conditional branches remain:\n\n"
                "(c) falls to the drift baseline and round-65 survival moves without a mass fail: "
                "expert iteration with the tightened recipe is next, as planned.\n\n"
                "(c) falls but survival does not move: return regression next, with outside-bank "
                "branches rolled out too, which is the principled form of item 2 and supplies "
                '"greedy is fine here" targets with magnitudes.'
            )
        return (
            "(c) falls but survival does not move: return regression next, with outside-bank "
            "branches rolled out too, which is the principled form of item 2 and supplies "
            '"greedy is fine here" targets with magnitudes.'
        )
    return (
        "(c) does not fall even with outside-bank counters: the boundary is not expressible "
        "from bv8 at this capacity, and the campaign moves to inputs."
    )


def diagnose(
    rows: list[Mapping[str, Any]],
    telemetry_roots: Mapping[str, Path],
    drift_summary: Mapping[str, Any] | None = None,
    candidate_drift: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run permitted CPU final-bank forwards and summarize final telemetry."""
    finals = {}
    for seed in SEEDS:
        name = f"{ARM}_s{seed}"
        root = Path(telemetry_roots[name])
        checkpoint = root / "final.pth"
        finals[name] = None if not checkpoint.exists() else _family_agreement(checkpoint)
    telemetry = read_telemetry(telemetry_roots)
    return json_safe(
        {
            "bank_agreement_raw_six_action": finals,
            "bank_agreement_five_seed_means": _family_means(finals),
            "telemetry_final_100k": _telemetry_summary(telemetry),
            "drift_candidate_readout": dict(candidate_drift or {}),
            "drift": drift_context(
                drift_summary if drift_summary is not None else load_drift_summary(),
                candidate_drift,
            ),
        }
    )
