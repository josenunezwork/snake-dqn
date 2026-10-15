"""Post-hoc T3 label-survivor reroll under a final checkpoint.

The fixed sample contains only decisive exported bank rows.  Each selected row
is bound to the bank manifest and its source-world provenance; the source
world is reconstructed and advanced with its original B8 policy to the stored
pre-action decision.  Every action in the *B8-labelled surviving set* is then
forced once and continued for 300 decisions by the supplied final policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from research.t3_rollout_labels_20261015 import bank_engine

DEPTH = 300
SAMPLE_SEED = 20261015
SAMPLE_SIZE = 200


def sha256(path: Path) -> str:
    """Return a file digest without making its bytes part of the report API."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _row_digest(row: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    obs_keys = ("packed",) if "packed" in row else ("local", "global")
    for key in (*obs_keys, "scalars", "mask", "surviving", "action"):
        value = np.asarray(row[key])
        digest.update(key.encode())
        digest.update(str((value.dtype.str, value.shape)).encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _bank_rows(bank: Path) -> list[dict[str, Any]]:
    """Read decisive exported rows, retaining their source-world identity."""
    manifest = bank / "manifest.json"
    if not manifest.is_file():
        raise ValueError(f"missing bank manifest: {manifest}")
    meta = json.loads(manifest.read_text())
    if meta.get("schema") != "t3-label-bank/v1":
        raise ValueError("bank manifest schema must be t3-label-bank/v1")
    files, shards = meta.get("files"), meta.get("shards")
    if not isinstance(files, dict) or not isinstance(shards, dict):
        raise ValueError("bank manifest must bind files and train shards")
    for name, digest in {**files, **shards}.items():
        path = bank / str(name)
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"bank manifest hash mismatch: {name}")
    rows: list[dict[str, Any]] = []
    for shard_name in sorted(shards):
        shard = bank / shard_name
        provenance = shard.with_suffix("").with_suffix(".provenance.jsonl")
        if not provenance.is_file():
            raise ValueError(f"missing provenance: {provenance}")
        entries = [json.loads(line) for line in provenance.read_text().splitlines() if line]
        with np.load(shard) as data:
            required = ("scalars", "mask", "surviving", "action")
            arrays = {key: data[key].copy() for key in required}
            if "packed" in data.files:
                arrays["packed"] = data["packed"].copy()
            else:
                arrays.update({key: data[key].copy() for key in ("local", "global")})
            if len(entries) != len(arrays["action"]):
                raise ValueError(f"provenance count mismatch: {shard}")
            for index, entry in enumerate(entries):
                if int(entry.get("row", -1)) != index:
                    raise ValueError(f"provenance row mismatch: {shard}:{index}")
                if not entry.get("decisive", False):
                    continue
                if "packed" in arrays:
                    row = {"packed": arrays["packed"][index].copy()}
                else:
                    row = {key: arrays[key][index].copy() for key in ("local", "global")}
                row.update(
                    {
                        key: arrays[key][index].copy()
                        for key in ("scalars", "mask", "surviving", "action")
                    }
                )
                if not bool(row["surviving"].any()):
                    raise ValueError("decisive exported row has no surviving action")
                rows.append(
                    {
                        "shard": shard.name,
                        "row": index,
                        "source_key": str(entry["source_key"]),
                        "decision": int(entry["decision"]),
                        "frame": int(entry["frame"]),
                        "digest": _row_digest(row),
                        "data": row,
                    }
                )
    return rows


def _unpack_row(row: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Materialize one selected packed row only; sample preparation stays cheap."""
    if "packed" not in row:
        return dict(row)
    from research.redesign_m3c_20261008.obs_pack import train_layout, unpack_ref

    local, global_ = unpack_ref(np.asarray(row["packed"])[None, ...], train_layout())
    return {
        **{key: value for key, value in row.items() if key != "packed"},
        "local": local[0],
        "global": global_[0],
    }


def prepare_sample(
    bank: Path, sample: Path, *, size: int = SAMPLE_SIZE, seed: int = SAMPLE_SEED
) -> dict[str, Any]:
    """Create once or validate a reusable deterministic decisive-row sample."""
    rows = _bank_rows(bank)
    if len(rows) < size:
        raise ValueError(f"bank has only {len(rows)} decisive exported rows; need {size}")
    manifest_digest = sha256(bank / "manifest.json")
    if sample.exists():
        value = json.loads(sample.read_text())
        if (
            value.get("schema") != "t3-reroll-sample/v1"
            or value.get("bank_manifest_sha256") != manifest_digest
        ):
            raise ValueError("existing sample is not bound to this exact bank manifest")
        selected = {
            (r["shard"], int(r["row"]), r["source_key"], r["decision"], r["frame"], r["digest"])
            for r in rows
        }
        seen = set()
        for entry in value.get("entries", []):
            identity = (
                entry.get("shard"),
                int(entry.get("row", -1)),
                entry.get("source_key"),
                entry.get("decision"),
                entry.get("frame"),
                entry.get("digest"),
            )
            if identity not in selected:
                raise ValueError("sample entry is absent from the current decisive export")
            if identity in seen:
                raise ValueError("sample contains a duplicate decisive row")
            seen.add(identity)
        if len(value.get("entries", [])) != size:
            raise ValueError("existing sample size differs from requested size")
        return value
    indices = np.random.default_rng(seed).choice(len(rows), size=size, replace=False)
    entries = [
        {
            key: rows[int(index)][key]
            for key in ("shard", "row", "source_key", "decision", "frame", "digest")
        }
        for index in sorted(indices)
    ]
    value = {
        "schema": "t3-reroll-sample/v1",
        "bank": str(bank),
        "bank_manifest_sha256": manifest_digest,
        "seed": seed,
        "size": size,
        "entries": entries,
    }
    _atomic_json(sample, value)
    return value


def _verify_lineage(receipt: Mapping[str, Any], cache: dict[Path, str]) -> None:
    """Refuse reconstruction if the receipt's B8 or roster checkpoint drifted."""
    from research.b8_death_census_20261014 import census

    identity = receipt.get("run_identity", {})
    expected_b8 = identity.get("b8_final_sha256", {})
    expected_roster = identity.get("roster_checkpoint_sha256", {})
    if not isinstance(expected_b8, dict) or not isinstance(expected_roster, dict):
        raise RuntimeError("world receipt lacks checkpoint lineage bindings")
    paths = {census.b8_path(name): digest for name, digest in expected_b8.items()}
    paths.update({Path(name): digest for name, digest in expected_roster.items()})
    for path, expected in paths.items():
        actual = cache.setdefault(path, sha256(path))
        if actual != expected:
            raise RuntimeError(f"receipt checkpoint lineage mismatch: {path}")


def _prepared_state(
    entry: Mapping[str, Any], row: Mapping[str, np.ndarray], bank: Path, lineage: dict[Path, str]
):
    """Rebuild and stop at the provenance-bound pre-action selector state."""
    receipt = json.loads((bank / "worlds" / f"{entry['source_key']}.json").read_text())
    _verify_lineage(receipt, lineage)
    arm, plan = str(receipt["arm"]), receipt["roster"]
    world = bank_engine.build_world(arm, plan)
    target, captured = int(entry["decision"]), {}

    class Captured(Exception):
        pass

    for decision in range(target + 1):
        original = world.runtime.prepare

        def prepare(sim, decision=decision):
            original(sim)
            if decision == target:
                from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster

                obs = build_ego_raster(
                    sim, EgoRasterConfig(version="bv8"), backend="numba", rows=np.array([[0, 0]])
                )
                captured["world"], captured["obs"] = world.clone(), obs
                raise Captured

        world.runtime.prepare = prepare
        try:
            bank_engine.choose_and_step(world, capture_obs=False)
        except Captured:
            break
        finally:
            world.runtime.prepare = original
    if "world" not in captured:
        raise RuntimeError("source world ended before the sampled decision")
    for key in ("local", "global", "scalars", "mask"):
        if not np.array_equal(captured["obs"][key][0], row[key]):
            raise RuntimeError(
                f"reconstructed state mismatch for {entry['source_key']} decision {target}: {key}"
            )
    return captured["world"]


def reroll(
    checkpoint: Path,
    bank: Path,
    sample: Path,
    out: Path,
    *,
    size: int = SAMPLE_SIZE,
    seed: int = SAMPLE_SEED,
) -> dict[str, Any]:
    """Evaluate all sampled labelled survivors under ``checkpoint`` continuations."""
    selected = prepare_sample(bank, sample, size=size, seed=seed)
    by_identity = {(row["shard"], row["row"], row["digest"]): row for row in _bank_rows(bank)}
    from src.model.ego2s_network import load_ego2s_checkpoint
    from src.simd_env.ego2s_policy import forward_q

    final = load_ego2s_checkpoint(str(checkpoint), "cpu").eval()
    details, survivor_total, survivor_kept, raw_agree, advisory_agree = [], 0, 0, 0, 0
    lineage: dict[Path, str] = {}
    for index, entry in enumerate(selected["entries"], start=1):
        item = by_identity[(entry["shard"], int(entry["row"]), entry["digest"])]
        row = _unpack_row(item["data"])
        labelled = np.flatnonzero(row["surviving"]).tolist()
        q = forward_q(
            final, {key: row[key][None, ...] for key in ("local", "global", "scalars")}, "rowwise"
        ).numpy()[0]
        raw = int(np.argmax(q))
        mask = np.asarray(row["mask"], bool)
        advisory = int(np.where(mask, q, -np.inf).argmax())
        world = _prepared_state(entry, row, bank, lineage)
        world.hero.net = final
        outcomes = bank_engine.rollout_all(world, labelled, DEPTH, int(entry["decision"]))
        kept = sum(bool(outcomes[action]["survives"]) for action in labelled)
        survivor_total += len(labelled)
        survivor_kept += kept
        raw_agree += int(bool(row["surviving"][raw]))
        advisory_agree += int(bool(row["surviving"][advisory]))
        details.append(
            {
                "source_key": entry["source_key"],
                "decision": entry["decision"],
                "labelled_surviving_actions": labelled,
                "still_surviving_actions": [a for a in labelled if outcomes[a]["survives"]],
                "final_raw_argmax": raw,
                "final_raw_argmax_in_labelled_surviving_set": bool(row["surviving"][raw]),
                "final_advisory_argmax": advisory,
                "final_advisory_argmax_in_labelled_surviving_set": bool(row["surviving"][advisory]),
            }
        )
        if index % 10 == 0 or index == len(selected["entries"]):
            print(f"reroll progress states={index}/{len(selected['entries'])}", flush=True)
    result = {
        "schema": "t3-reroll-result/v1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "bank_manifest_sha256": selected["bank_manifest_sha256"],
        "sample": str(sample),
        "sample_sha256": sha256(sample),
        "depth": DEPTH,
        "continuation": (
            "forced labelled-surviving first action; final checkpoint then "
            "greedy advisory-masked policy"
        ),
        "argmax_definition": (
            "raw six-action, unmasked np.argmax with NumPy first-maximum tie break; "
            "advisory value is reported separately"
        ),
        "sample_states": len(details),
        "labelled_survivor_actions": survivor_total,
        "labelled_survivor_actions_still_survive": survivor_kept,
        "labelled_survivor_action_survival_fraction": survivor_kept / survivor_total,
        "final_raw_argmax_in_labelled_surviving_set_fraction": raw_agree / len(details),
        "final_advisory_argmax_in_labelled_surviving_set_fraction": advisory_agree / len(details),
        "states": details,
    }
    _atomic_json(out, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument(
        "--sample", type=Path, required=True, help="Created once, then reused for every final"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--sample-seed", type=int, default=SAMPLE_SEED)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    import torch

    torch.set_num_threads(1)
    result = reroll(
        args.checkpoint,
        args.bank,
        args.sample,
        args.out,
        size=args.sample_size,
        seed=args.sample_seed,
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "sample_states",
                    "labelled_survivor_action_survival_fraction",
                    "final_raw_argmax_in_labelled_surviving_set_fraction",
                )
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
