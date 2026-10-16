"""Descriptive T4a re-roll of derived-bank states under final checkpoints.

The T4a bank is a derived v2 label layer.  It retains the source observations,
but keeps source-world provenance and reconstruction receipts in the immutable
v1 source bank named by its manifest.  A sampled state therefore binds both
layers before it is reconstructed.  The continuation target mirrors T4a's
two-sided loss: a decisive row forces every preferred action once; a counter
row forces only its stored B8 action.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from research.t3_screen_20261015 import reroll_t3

SAMPLE_SEED = 20261016
SAMPLE_SIZE = 200


def sha256(path: Path) -> str:
    """Return a file digest using the established re-roll implementation."""
    return reroll_t3.sha256(path)


def _source_meta(bank: Path, meta: Mapping[str, Any]) -> tuple[Path, Mapping[str, str]]:
    """Resolve and validate the immutable source-bank binding from v2 metadata."""
    source = meta.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("bank v2 manifest lacks source binding")
    path, files = source.get("path"), source.get("files")
    if not isinstance(path, str) or not isinstance(files, Mapping):
        raise ValueError("bank v2 source binding lacks path or files")
    source_bank = Path(path)
    if not source_bank.is_dir():
        raise ValueError(f"source bank is unavailable: {source_bank}")
    for name in ("manifest.json", "bank-00000.npz", "bank-00000.provenance.jsonl"):
        expected = files.get(name)
        actual = source_bank / name
        if not isinstance(expected, str) or not actual.is_file() or sha256(actual) != expected:
            raise ValueError(f"source bank hash mismatch: {name}")
    return source_bank, files


def _row_digest(row: Mapping[str, np.ndarray]) -> str:
    """Bind labels and observations that determine a v2 re-roll target."""
    import hashlib

    digest = hashlib.sha256()
    obs_keys = ("packed",) if "packed" in row else ("local", "global")
    for key in (*obs_keys, "scalars", "mask", "surviving", "surviving_original", "action"):
        value = np.asarray(row[key])
        digest.update(key.encode())
        digest.update(str((value.dtype.str, value.shape)).encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _bank_rows(bank: Path) -> tuple[list[dict[str, Any]], Path, Mapping[str, str]]:
    """Read v2 rows and pair each one with its source-world provenance."""
    manifest = bank / "manifest.json"
    if not manifest.is_file():
        raise ValueError(f"missing bank manifest: {manifest}")
    meta = json.loads(manifest.read_text())
    if meta.get("schema") != "t4-derived-label-bank/v2":
        raise ValueError("bank manifest schema must be t4-derived-label-bank/v2")
    if meta.get("label_policy") != "prefer-nonboost":
        raise ValueError("T4a reroll requires prefer-nonboost v2 labels")
    shards = meta.get("shards")
    if not isinstance(shards, Mapping):
        raise ValueError("bank v2 manifest must bind shards")
    source_bank, source_files = _source_meta(bank, meta)
    rows: list[dict[str, Any]] = []
    for shard_name, expected in sorted(shards.items()):
        shard = bank / str(shard_name)
        if not isinstance(expected, str) or not shard.is_file() or sha256(shard) != expected:
            raise ValueError(f"bank v2 shard hash mismatch: {shard_name}")
        source_shard = source_bank / str(shard_name)
        provenance = source_shard.with_suffix("").with_suffix(".provenance.jsonl")
        with (
            np.load(shard, allow_pickle=False) as data,
            np.load(source_shard, allow_pickle=False) as source,
        ):
            required = ("scalars", "mask", "surviving", "surviving_original", "action")
            if any(key not in data.files for key in required):
                raise ValueError(f"bank v2 shard lacks required row data: {shard}")
            if any(key not in source.files for key in ("scalars", "mask", "surviving", "action")):
                raise ValueError(f"source bank shard lacks required row data: {source_shard}")
            entries = [json.loads(line) for line in provenance.read_text().splitlines() if line]
            if len(entries) != len(data["action"]) or len(source["action"]) != len(entries):
                raise ValueError(f"v2/source provenance count mismatch: {shard_name}")
            obs_keys = ("packed",) if "packed" in data.files else ("local", "global")
            if any(key not in source.files for key in obs_keys):
                raise ValueError(f"source observation layout differs: {shard_name}")
            # NpzFile decompresses a member on each subscript.  The actual packed
            # bank is 50 MB, so cache every member before walking its 16k rows.
            derived_arrays = {key: data[key].copy() for key in (*obs_keys, *required)}
            source_arrays = {
                key: source[key].copy()
                for key in (*obs_keys, "scalars", "mask", "surviving", "action")
            }
            for index, entry in enumerate(entries):
                if int(entry.get("row", -1)) != index:
                    raise ValueError(f"provenance row mismatch: {shard_name}:{index}")
                row = {key: derived_arrays[key][index].copy() for key in (*obs_keys, *required)}
                original = source_arrays["surviving"][index]
                if not np.array_equal(row["surviving_original"], original):
                    raise ValueError(f"v2 original-survival drift: {shard_name}:{index}")
                for key in (*obs_keys, "scalars", "mask", "action"):
                    if not np.array_equal(row[key], source_arrays[key][index]):
                        raise ValueError(f"v2/source state drift for {key}: {shard_name}:{index}")
                action = int(row["action"])
                counter = bool(row["surviving_original"][action])
                target_actions = [action] if counter else np.flatnonzero(row["surviving"]).tolist()
                if not target_actions:
                    raise ValueError(f"decisive row has no preferred actions: {shard_name}:{index}")
                rows.append(
                    {
                        "shard": str(shard_name),
                        "row": index,
                        "source_key": str(entry["source_key"]),
                        "decision": int(entry["decision"]),
                        "frame": int(entry["frame"]),
                        "side": "counter" if counter else "decisive",
                        "target_actions": target_actions,
                        "digest": _row_digest(row),
                        "data": row,
                    }
                )
    return rows, source_bank, source_files


def prepare_sample(
    bank: Path, sample: Path, *, size: int = SAMPLE_SIZE, seed: int = SAMPLE_SEED
) -> dict[str, Any]:
    """Create once or validate a deterministic v2 sample shared by all finals."""
    rows, source_bank, _ = _bank_rows(bank)
    if len(rows) < size:
        raise ValueError(f"bank has only {len(rows)} v2 rows; need {size}")
    manifest_digest, source_digest = sha256(bank / "manifest.json"), sha256(
        source_bank / "manifest.json"
    )
    if sample.exists():
        value = json.loads(sample.read_text())
        if (
            value.get("schema") != "t4a-reroll-sample/v1"
            or value.get("bank_manifest_sha256") != manifest_digest
            or value.get("source_manifest_sha256") != source_digest
        ):
            raise ValueError("existing sample is not bound to these exact v2/source manifests")
        known = {
            (row["shard"], row["row"], row["source_key"], row["decision"], row["digest"])
            for row in rows
        }
        selected = set()
        for entry in value.get("entries", []):
            identity = (
                entry.get("shard"),
                int(entry.get("row", -1)),
                entry.get("source_key"),
                entry.get("decision"),
                entry.get("digest"),
            )
            if identity not in known or identity in selected:
                raise ValueError("sample entry is absent from the current v2 bank or duplicated")
            selected.add(identity)
        if len(value.get("entries", [])) != size:
            raise ValueError("existing sample size differs from requested size")
        expected = {
            (rows[int(index)]["shard"], rows[int(index)]["row"], rows[int(index)]["digest"])
            for index in np.random.default_rng(seed).choice(len(rows), size=size, replace=False)
        }
        actual = {
            (entry["shard"], int(entry["row"]), entry["digest"]) for entry in value["entries"]
        }
        if actual != expected:
            raise ValueError("existing sample differs from the deterministic v2 selection")
        return value
    indices = np.random.default_rng(seed).choice(len(rows), size=size, replace=False)
    entries = [
        {
            key: rows[int(index)][key]
            for key in (
                "shard",
                "row",
                "source_key",
                "decision",
                "frame",
                "side",
                "target_actions",
                "digest",
            )
        }
        for index in sorted(indices)
    ]
    value = {
        "schema": "t4a-reroll-sample/v1",
        "bank": str(bank),
        "bank_manifest_sha256": manifest_digest,
        "source_bank": str(source_bank),
        "source_manifest_sha256": source_digest,
        "seed": seed,
        "size": size,
        "entries": entries,
    }
    reroll_t3._atomic_json(sample, value)
    return value


def _verify_receipt(source_bank: Path, source_files: Mapping[str, str], source_key: str) -> None:
    """Bind each selected reconstruction receipt to v2's source-tree inventory."""
    relative = f"worlds/{source_key}.json"
    receipt = source_bank / relative
    expected = source_files.get(relative)
    if not isinstance(expected, str) or not receipt.is_file() or sha256(receipt) != expected:
        raise ValueError(f"source world receipt hash mismatch: {relative}")


def reroll(
    checkpoint: Path,
    bank: Path,
    sample: Path,
    out: Path,
    *,
    size: int = SAMPLE_SIZE,
    seed: int = SAMPLE_SEED,
) -> dict[str, Any]:
    """Evaluate the T4a target action(s) of every sampled v2 state."""
    selected = prepare_sample(bank, sample, size=size, seed=seed)
    rows, source_bank, source_files = _bank_rows(bank)
    by_identity = {(row["shard"], row["row"], row["digest"]): row for row in rows}
    from src.model.ego2s_network import load_ego2s_checkpoint
    from src.simd_env.ego2s_policy import forward_q

    final = load_ego2s_checkpoint(str(checkpoint), "cpu").eval()
    details, target_total, target_kept = [], 0, 0
    raw_ok = {"decisive": 0, "counter": 0}
    advisory_ok = {"decisive": 0, "counter": 0}
    side_total = {"decisive": 0, "counter": 0}
    lineage: dict[Path, str] = {}
    for index, entry in enumerate(selected["entries"], start=1):
        item = by_identity[(entry["shard"], int(entry["row"]), entry["digest"])]
        row, side, targets = item["data"], item["side"], item["target_actions"]
        _verify_receipt(source_bank, source_files, item["source_key"])
        unpacked = reroll_t3._unpack_row(row)
        q = forward_q(
            final,
            {key: unpacked[key][None, ...] for key in ("local", "global", "scalars")},
            "rowwise",
        ).numpy()[0]
        raw = int(np.argmax(q))
        mask = np.asarray(unpacked["mask"], bool)
        advisory = int(np.where(mask, q, -np.inf).argmax())
        if side == "counter":
            raw_match, advisory_match = raw == targets[0], advisory == targets[0]
        else:
            raw_match, advisory_match = raw in targets, advisory in targets
        side_total[side] += 1
        raw_ok[side] += int(raw_match)
        advisory_ok[side] += int(advisory_match)
        world = reroll_t3._prepared_state(entry, unpacked, source_bank, lineage)
        world.hero.net = final
        outcomes = reroll_t3.bank_engine.rollout_all(
            world, targets, reroll_t3.DEPTH, int(entry["decision"])
        )
        kept = sum(bool(outcomes[action]["survives"]) for action in targets)
        target_total += len(targets)
        target_kept += kept
        details.append(
            {
                "source_key": entry["source_key"],
                "decision": entry["decision"],
                "side": side,
                "target_actions": targets,
                "target_actions_still_survive": [a for a in targets if outcomes[a]["survives"]],
                "final_raw_argmax": raw,
                "final_raw_argmax_matches_t4a_target": bool(raw_match),
                "final_advisory_argmax": advisory,
                "final_advisory_argmax_matches_t4a_target": bool(advisory_match),
            }
        )
        if index % 10 == 0 or index == len(selected["entries"]):
            print(f"reroll progress states={index}/{len(selected['entries'])}", flush=True)
    result = {
        "schema": "t4a-reroll-result/v1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "bank_manifest_sha256": selected["bank_manifest_sha256"],
        "source_manifest_sha256": selected["source_manifest_sha256"],
        "sample": str(sample),
        "sample_sha256": sha256(sample),
        "depth": reroll_t3.DEPTH,
        "target_definition": (
            "decisive: v2 prefer-nonboost set; counter: stored B8 action, "
            "classified by surviving_original[action]"
        ),
        "sample_states": len(details),
        "target_actions": target_total,
        "target_actions_still_survive": target_kept,
        "target_action_survival_fraction": target_kept / target_total,
        "raw_argmax_matches_t4a_target_by_side": {
            side: raw_ok[side] / side_total[side] if side_total[side] else None
            for side in side_total
        },
        "advisory_argmax_matches_t4a_target_by_side": {
            side: advisory_ok[side] / side_total[side] if side_total[side] else None
            for side in side_total
        },
        "states_by_side": side_total,
        "states": details,
    }
    reroll_t3._atomic_json(out, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
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
                    "target_action_survival_fraction",
                    "raw_argmax_matches_t4a_target_by_side",
                )
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
