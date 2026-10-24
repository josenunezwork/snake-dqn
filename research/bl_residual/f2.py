"""One sealed F2 look under D4096 start judgment section 4, items 1--6.

Preparing and publishing ledgers never deserialize an outcome. The runner writes
the statistic only to its create-only sealed receipt. A separate hash-only result
publication must reach origin/main before any consumer opens that receipt.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from research.bl_ppo.exports import digest, export_json, verify_export

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts")
F2_ROOT = ARTIFACTS / "big-lever/exp/f2-i2-20261024"
RECEIPT = F2_ROOT / "sealed-receipt.json"
INPUT_LEDGER = "research/bl_residual/ledger/f2-inputs-20261024.json"
RESULT_LEDGER = "research/bl_residual/ledger/f2-result-20261024.json"
JUDGMENT = ARTIFACTS / "big-lever/exp/f4/JUDGMENT_D4096_START_20261024.md"
JUDGMENT_SHA = "ccf1e0cd8d60a2019c983ab2714d58663ca42e2a228a0ddf78c16216c2ef83fb"
SCHEMA = "bl-residual-f2-disposition/v1"
EXPECTED_DECISIVE = (1481, 269)
THREAD_ENV = (
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
SOURCE_FILES = (
    "research/bl_residual/f2.py",
    "research/bl_residual/f2_inputs.py",
    "research/bl_residual/f2_c0.py",
    "research/a100_a2/PREREGISTRATION.md",
    "research/a100_a2/SEMANTICS_ERRATUM.md",
    "research/a100_a2/model.py",
    "research/a100_a2/data.py",
    "research/a100_a2/run.py",
    "research/a100_a2/protocol.py",
    "research/bl_ppo/exports.py",
    "src/model/ego2s_network.py",
    "research/redesign_m3c_20261008/obs_pack.py",
)


def _git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=root, timeout=30, stderr=subprocess.DEVNULL)


def _published(path: Path, relative: str, root: Path) -> str:
    """Verify payload and sidecars locally and on origin/main, without parsing it."""
    if path.resolve() != (root / relative).resolve():
        raise ValueError("publication must use the canonical checkout ledger")
    verify_export(path)
    for suffix in ("", ".sha256", ".complete"):
        local = Path(str(path) + suffix)
        if local.read_bytes() != _git(root, "show", f"origin/main:{relative}{suffix}"):
            raise ValueError("local publication differs from origin/main")
    # Stable across unrelated later main commits, so a prepared R-A packet does
    # not become foreign merely because another implementation landed.
    return _git(root, "log", "-1", "--format=%H", "origin/main", "--", relative).decode().strip()


def _commit_contains(root: Path, commit: str, relative: str, payload: bytes) -> None:
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("published ref must be a full commit")
    _git(root, "merge-base", "--is-ancestor", commit, "origin/main")
    if _git(root, "show", f"{commit}:{relative}") != payload:
        raise ValueError("published ref does not contain the bound ledger")


def prepare(*, root: Path = ROOT) -> dict[str, Any]:
    """Freeze metadata/sidecars only; never hash or deserialize a label payload."""
    from research.bl_residual.f2_inputs import metadata_inputs

    if digest(JUDGMENT) != JUDGMENT_SHA:
        raise ValueError("judgment hash differs")
    value = {
        "schema": "bl-residual-f2-inputs/v1",
        "judgment_sha256": JUDGMENT_SHA,
        "inputs": metadata_inputs(),
        "sources": {name: digest(root / name) for name in SOURCE_FILES},
        "receipt_path": str(RECEIPT),
        "one_look_lock": str(F2_ROOT / "one-look.lock.json"),
        "analysis": {
            "decisive_rows": 1481,
            "decisive_worlds": 269,
            "doses": [25, 100],
            "encoder": "c0",
            "target": "d300",
            "epochs": 20,
            "seeds": [20261022, 20261023],
            "bootstrap_seed": 3403,
            "resamples": 10000,
            "lower_quantile": 0.1,
            "selection": "argmin legal predicted D300 hazard; lowest action index ties",
            "endpoint": "EXACT H300 survivor; NOT_EVALUATED/UNKNOWN are not survived",
        },
        "outcome_reporting": "sealed receipt only",
    }
    export_json(root / INPUT_LEDGER, value)
    return {"ledger": INPUT_LEDGER, "sha256": digest(root / INPUT_LEDGER)}


def selected_survival(table: dict[str, np.ndarray], probabilities: np.ndarray) -> np.ndarray:
    """Use all legal actions, including ones without a recorded exact endpoint."""
    legal = np.asarray(table["legal"])
    if (
        legal.dtype != np.bool_
        or probabilities.shape != legal.shape
        or legal.ndim != 2
        or legal.shape[1] != 6
        or not legal.any(axis=1).all()
        or not np.isfinite(probabilities).all()
        or np.any((probabilities < 0) | (probabilities > 1))
    ):
        raise ValueError("invalid legal actions or D300 predictions")
    choice = np.where(legal, probabilities, np.inf).argmin(axis=1)
    exact = (
        legal
        & ~np.asarray(table["died"], dtype=bool)
        & (np.asarray(table["horizon"]) == 300)
        & np.asarray(table["truncated_at_300"], dtype=bool)
    )
    # Explicit uncertainty always defeats an otherwise survivor-shaped record.
    if "status" in table:
        exact &= np.asarray(table["status"]).astype(str) == "EXACT"
    if "observed" in table:
        exact &= np.asarray(table["observed"], dtype=bool)
    return exact[np.arange(len(choice)), choice]


def statistic(table: dict[str, np.ndarray], predictions: dict[int, np.ndarray]) -> dict[str, Any]:
    """Paired row mean, resampling whole decisive-world clusters with replacement."""
    decisive = np.asarray(table["boost_only_escape"])
    worlds = np.asarray(table["world"]).astype(str)
    if decisive.dtype != np.bool_ or decisive.shape != worlds.shape:
        raise ValueError("invalid decisive row mask")
    if (int(decisive.sum()), len(np.unique(worlds[decisive]))) != EXPECTED_DECISIVE:
        raise ValueError("INVALID decisive rows/worlds")
    if set(predictions) != {25, 100}:
        raise ValueError("only the two pinned doses are admitted")
    delta = (
        selected_survival(table, predictions[100]).astype(np.int64)
        - selected_survival(table, predictions[25]).astype(np.int64)
    )[decisive]
    parents, inverse = np.unique(worlds[decisive], return_inverse=True)
    sums = np.bincount(inverse, weights=delta).astype(np.int64)
    sizes = np.bincount(inverse)
    rng = np.random.default_rng(3403)
    # Integer sufficient statistics preserve every row in each sampled cluster.
    picked = rng.integers(0, len(parents), size=(10000, len(parents)))
    draws = sums[picked].sum(axis=1) / sizes[picked].sum(axis=1)
    lower = float(np.quantile(draws, 0.1, method="linear"))
    return {
        "paired_delta": float(delta.mean()),
        "decisive_rows": int(len(delta)),
        "decisive_worlds": len(parents),
        "world_cluster_bootstrap": {
            "resamples": 10000,
            "seed": 3403,
            "quantile": 0.1,
            "method": "linear",
            "lower90": lower,
        },
        "volume_struck": lower <= 0,
    }


def _validate_plan(plan: dict[str, Any], root: Path) -> None:
    if (
        plan.get("schema") != "bl-residual-f2-inputs/v1"
        or plan.get("judgment_sha256") != JUDGMENT_SHA
        or plan.get("receipt_path") != str(RECEIPT)
        or plan.get("one_look_lock") != str(F2_ROOT / "one-look.lock.json")
        or set(plan.get("sources", {})) != set(SOURCE_FILES)
        or digest(JUDGMENT) != JUDGMENT_SHA
    ):
        raise ValueError("foreign F2 plan")
    for name, expected in plan["sources"].items():
        if digest(root / name) != expected:
            raise ValueError("F2 implementation changed after publication")


def run_sealed(*, root: Path = ROOT) -> dict[str, str]:
    """Claim the single look, compute in memory, and seal without disclosing it."""
    from research.bl_residual.f2_c0 import fit_c0
    from research.bl_residual.f2_inputs import load_inputs, load_raw, safe_path

    if (
        any(os.environ.get(name) != "1" for name in THREAD_ENV)
        or os.getpriority(os.PRIO_PROCESS, 0) < 15
    ):
        raise ValueError("F2 requires nice >=15 and single-thread environment")
    published_ref = _published(root / INPUT_LEDGER, INPUT_LEDGER, root)
    plan = json.loads((root / INPUT_LEDGER).read_text())
    _validate_plan(plan, root)
    safe_path(RECEIPT)
    if any(Path(str(RECEIPT) + suffix).exists() for suffix in ("", ".sha256", ".complete")):
        raise FileExistsError("F2 receipt already exists")
    os.umask(0o077)
    export_json(
        F2_ROOT / "one-look.lock.json",
        {"source_sha256": digest(root / INPUT_LEDGER), "published_ref": published_ref},
    )
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "judgment_sha256": JUDGMENT_SHA,
        "published": True,
        "published_ref": published_ref,
        "source_sha256": digest(root / INPUT_LEDGER),
        "status": "INVALID",
        "volume_struck": True,
    }
    # Even accidental diagnostics from imported code cannot escape the seal.
    with (
        open(os.devnull, "w") as sink,
        contextlib.redirect_stdout(sink),
        contextlib.redirect_stderr(sink),
    ):
        try:
            table, folds, predictions = load_inputs(plan["inputs"])
            # Counts must fail before an expensive fallback is attempted.
            decisive = table["boost_only_escape"]
            if (int(decisive.sum()), len(np.unique(table["world"][decisive]))) != EXPECTED_DECISIVE:
                raise ValueError("INVALID decisive rows/worlds")
            mode = "a2-sha-verified-fold-predictions"
            if predictions is None:
                mode = "refit-c0-only"
                raw = load_raw(plan["inputs"], table, folds)
                predictions = fit_c0(raw, Path(plan["inputs"]["checkpoint"]["path"]))
            receipt.update(statistic(table, predictions))
            receipt.update(status="COMPLETE", prediction_source=mode)
        except Exception as error:
            # Fixed diagnostic codes only: never serialize exception text/arrays.
            receipt["status"] = "NOT_RUN" if type(error).__name__ == "LineageError" else "INVALID"
            receipt["reason"] = (
                "FORBIDDEN_LINEAGE"
                if receipt["status"] == "NOT_RUN"
                else "INPUT_OR_ANALYSIS_INVALID"
            )
        export_json(RECEIPT, receipt)
    return {"sealed_receipt": str(RECEIPT), "sha256": digest(RECEIPT)}


def stage_publication(*, root: Path = ROOT) -> dict[str, str]:
    """Publish only receipt path/hash/size; never parse the sealed receipt."""
    input_ref = _published(root / INPUT_LEDGER, INPUT_LEDGER, root)
    sha = verify_export(RECEIPT)
    value = {
        "schema": "bl-residual-f2-publication/v1",
        "receipt_path": str(RECEIPT),
        "receipt_sha256": sha,
        "receipt_bytes": RECEIPT.stat().st_size,
        "source_sha256": digest(root / INPUT_LEDGER),
        "input_published_ref": input_ref,
        "judgment_sha256": JUDGMENT_SHA,
    }
    export_json(root / RESULT_LEDGER, value)
    return {"publication": RESULT_LEDGER, "sha256": digest(root / RESULT_LEDGER)}


def verify_published_receipt(receipt: Path, *, root: Path | None = None) -> dict[str, str]:
    """Return only binding metadata; the result ledger must be published first."""
    from research.bl_residual.f2_inputs import safe_path

    root = ROOT if root is None else Path(root)
    receipt = safe_path(receipt)
    if receipt != RECEIPT.resolve():
        raise ValueError("F2 requires the canonical one-look receipt")
    input_ref = _published(root / INPUT_LEDGER, INPUT_LEDGER, root)
    _validate_plan(json.loads((root / INPUT_LEDGER).read_text()), root)
    publication_ref = _published(root / RESULT_LEDGER, RESULT_LEDGER, root)
    publication = json.loads((root / RESULT_LEDGER).read_text())
    if (
        publication.get("schema") != "bl-residual-f2-publication/v1"
        or publication.get("receipt_path") != str(receipt)
        or publication.get("receipt_sha256") != verify_export(receipt)
        or publication.get("receipt_bytes") != receipt.stat().st_size
        or publication.get("source_sha256") != digest(root / INPUT_LEDGER)
        or publication.get("judgment_sha256") != JUDGMENT_SHA
    ):
        raise ValueError("F2 receipt differs from its published ledger")
    _commit_contains(
        root,
        publication["input_published_ref"],
        INPUT_LEDGER,
        (root / INPUT_LEDGER).read_bytes(),
    )
    # FIRST semantic read of the result, only after all publication checks above.
    sealed_bytes = receipt.read_bytes()
    if hashlib.sha256(sealed_bytes).hexdigest() != publication["receipt_sha256"]:
        raise ValueError("F2 receipt changed after publication verification")
    value = json.loads(sealed_bytes)
    if (
        value.get("schema") != SCHEMA
        or value.get("published") is not True
        or type(value.get("volume_struck")) is not bool
        or value.get("source_sha256") != publication["source_sha256"]
        or value.get("judgment_sha256") != JUDGMENT_SHA
        or value.get("status") not in ("COMPLETE", "INVALID", "NOT_RUN")
        or (value.get("status") != "COMPLETE" and value.get("volume_struck") is not True)
    ):
        raise ValueError("foreign F2 receipt")
    if value["status"] == "COMPLETE":
        bootstrap = value.get("world_cluster_bootstrap", {})
        lower = bootstrap.get("lower90")
        point = value.get("paired_delta")
        if (
            type(lower) not in (int, float)
            or not np.isfinite(lower)
            or not -1 <= lower <= 1
            or type(point) not in (int, float)
            or not np.isfinite(point)
            or not -1 <= point <= 1
            or bootstrap
            != {
                "resamples": 10000,
                "seed": 3403,
                "quantile": 0.1,
                "method": "linear",
                "lower90": lower,
            }
            or (value.get("decisive_rows"), value.get("decisive_worlds")) != EXPECTED_DECISIVE
            or value["volume_struck"] != (lower <= 0)
        ):
            raise ValueError("F2 disposition differs from the pinned statistic")
    _commit_contains(root, value["published_ref"], INPUT_LEDGER, (root / INPUT_LEDGER).read_bytes())
    # input_ref may advance without changing the published ledger bytes.
    del input_ref
    return {
        "path": str(receipt),
        "sha256": publication["receipt_sha256"],
        "published_ref": value["published_ref"],
        "source_sha256": value["source_sha256"],
        "publication_ref": publication_ref,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "step", choices=("metadata-check", "prepare", "run-sealed", "stage-publication", "verify")
    )
    args = parser.parse_args()
    try:
        if args.step == "metadata-check":
            from research.bl_residual.f2_inputs import metadata_inputs

            inputs = metadata_inputs()
            result = {
                "status": "METADATA_ONLY",
                "prediction_candidates": len(inputs["prediction_units"]),
            }
        elif args.step == "prepare":
            result = prepare()
        elif args.step == "run-sealed":
            result = run_sealed()
        elif args.step == "stage-publication":
            result = stage_publication()
        else:
            result = verify_published_receipt(RECEIPT)
        print(json.dumps(result, sort_keys=True))
    except Exception:
        parser.exit(2, "F2 refused; no statistic disclosed.\n")


if __name__ == "__main__":
    main()
