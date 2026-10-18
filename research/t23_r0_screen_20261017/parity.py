"""T23's small, diagnostic Mac--Colab identity probe.

This is deliberately separate from the selection evaluator.  It replays four
*reserved* T23 worlds (two scripted, one frozen, one mixed) with the released
B8 hero, using its exact rowwise CPU evaluation path.  The resulting artifact
is portable: ``compare`` fails closed if a world, metric, frame digest, action
stream, or bound identity differs.  When the first action differs it includes
the recorded float32 top-two margin and ULP distance; a tie is evidence, never
an exemption.
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence
from unittest.mock import patch

import numpy as np

B8_SHA256 = "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"
SCHEMA = "t23-parity/v1"
HORIZON = 5000
MIX_COUNTS = {"scripted": 2, "frozen": 1, "mixed": 1}


def sha256_file(path: str | Path) -> str:
    """Return the streaming SHA-256 of ``path``."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _load(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        result = json.load(stream)
    if not isinstance(result, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return result


def _write(path: str | Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _first_difference(left: Sequence[Any], right: Sequence[Any]) -> int | None:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def _ulp_distance(a: float, b: float) -> int:
    """Unsigned float32 ULP distance, including values on opposite signs."""
    values = np.asarray([a, b], dtype=np.float32).view(np.uint32).astype(np.int64)
    # Flip the sign half onto a monotonically ordered unsigned interval.  This
    # intentionally maps -0 and +0 together.
    ordered = np.where(values & 0x80000000, 0x80000000 - (values & 0x7FFFFFFF), values + 0x80000000)
    return int(abs(int(ordered[0]) - int(ordered[1])))


def _decision_diagnostic(left: Any, right: Any) -> dict[str, Any]:
    """Describe an action mismatch without converting it into a pass condition."""
    out: dict[str, Any] = {"mac": left, "colab": right}
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        return out
    out.update({key: left.get(key) for key in ("frame", "slot", "policy", "action")})
    for side, packet in (("mac", left), ("colab", right)):
        q = packet.get("q") if isinstance(packet, Mapping) else None
        mask = packet.get("mask") if isinstance(packet, Mapping) else None
        if isinstance(q, list) and len(q) >= 2:
            eligible = (
                [i for i, enabled in enumerate(mask) if enabled]
                if isinstance(mask, list) and len(mask) == len(q)
                else list(range(len(q)))
            )
            ordered = sorted((float(q[index]), index) for index in eligible)
            if len(ordered) < 2:
                continue
            high, second = ordered[-1], ordered[-2]
            out[f"{side}_top_action"] = int(high[1])
            out[f"{side}_top2_gap_float32"] = float(np.float32(high[0]) - np.float32(second[0]))
            out[f"{side}_top2_ulp"] = _ulp_distance(high[0], second[0])
    return out


def compare_artifacts(mac: Mapping[str, Any], colab: Mapping[str, Any]) -> dict[str, Any]:
    """Return a fail-closed structural comparison of two T23 parity artifacts."""
    errors: list[str] = []
    diagnostics: list[dict[str, Any]] = []
    for name, value in (("mac", mac), ("colab", colab)):
        if value.get("schema") != SCHEMA:
            errors.append(f"{name}: wrong schema {value.get('schema')!r}")
        if value.get("horizon") != HORIZON:
            errors.append(f"{name}: horizon is not H5000")
        worlds = value.get("worlds")
        if not isinstance(worlds, list):
            errors.append(f"{name}: worlds is not a list")
    if errors:
        return {
            "schema": "t23-parity-comparison/v1",
            "pass": False,
            "errors": errors,
            "diagnostics": diagnostics,
        }

    for field in ("bank_sha256", "b8_sha256", "profile_digest", "source_sha256", "horizon"):
        if mac.get(field) != colab.get(field):
            errors.append(f"identity mismatch: {field}: {mac.get(field)!r} != {colab.get(field)!r}")
        if field != "horizon" and (not isinstance(mac.get(field), str) or not mac.get(field)):
            errors.append(f"mac: missing {field}")
        if field != "horizon" and (not isinstance(colab.get(field), str) or not colab.get(field)):
            errors.append(f"colab: missing {field}")
    if mac.get("b8_sha256") != B8_SHA256 or colab.get("b8_sha256") != B8_SHA256:
        errors.append("B8 SHA256 pin mismatch")

    def index(value: Mapping[str, Any], label: str) -> dict[str, Mapping[str, Any]]:
        out: dict[str, Mapping[str, Any]] = {}
        for row in value["worlds"]:
            if not isinstance(row, Mapping) or not isinstance(row.get("world_id"), str):
                errors.append(f"{label}: invalid world identity")
                continue
            if row["world_id"] in out:
                errors.append(f"{label}: duplicate world_id {row['world_id']}")
            out[row["world_id"]] = row
        return out

    left, right = index(mac, "mac"), index(colab, "colab")
    if set(left) != set(right):
        errors.append(f"world identities differ: mac={sorted(left)} colab={sorted(right)}")
    counts = {mix: sum(1 for row in left.values() if row.get("mix") == mix) for mix in MIX_COUNTS}
    if counts != MIX_COUNTS:
        errors.append(f"reserved mix counts wrong: {counts!r}")

    for world_id in sorted(set(left) & set(right)):
        a, b = left[world_id], right[world_id]
        for field in ("world_seed", "mix", "roster_digest", "world_digest", "record"):
            if a.get(field) != b.get(field):
                errors.append(f"{world_id}: {field} mismatch")
            if field in ("roster_digest", "world_digest") and not isinstance(a.get(field), str):
                errors.append(f"{world_id}: missing mac {field}")
            if field in ("roster_digest", "world_digest") and not isinstance(b.get(field), str):
                errors.append(f"{world_id}: missing colab {field}")
        actions_a, actions_b = a.get("actions", []), b.get("actions", [])
        if not isinstance(actions_a, list) or not isinstance(actions_b, list):
            errors.append(f"{world_id}: actions missing")
        else:
            if not actions_a or not actions_b:
                errors.append(f"{world_id}: empty action trace")
            comparable_a = [
                (
                    {key: event.get(key) for key in ("frame", "slot", "policy", "action")}
                    if isinstance(event, Mapping)
                    else event
                )
                for event in actions_a
            ]
            comparable_b = [
                (
                    {key: event.get(key) for key in ("frame", "slot", "policy", "action")}
                    if isinstance(event, Mapping)
                    else event
                )
                for event in actions_b
            ]
            difference = _first_difference(comparable_a, comparable_b)
            if difference is not None:
                errors.append(f"{world_id}: action mismatch at decision {difference}")
                diagnostics.append(
                    {
                        "world_id": world_id,
                        "first_action_difference": difference,
                        "tie_diagnostic": _decision_diagnostic(
                            actions_a[difference] if difference < len(actions_a) else None,
                            actions_b[difference] if difference < len(actions_b) else None,
                        ),
                    }
                )
        frames_a, frames_b = a.get("frame_digests", []), b.get("frame_digests", [])
        if not isinstance(frames_a, list) or not isinstance(frames_b, list):
            errors.append(f"{world_id}: frame digests missing")
            continue
        if len(frames_a) != HORIZON or len(frames_b) != HORIZON:
            errors.append(f"{world_id}: frame digest count must be H5000")
        difference = _first_difference(frames_a, frames_b)
        if difference is not None:
            errors.append(f"{world_id}: world digest mismatch at frame {difference + 1}")
            diagnostics.append({"world_id": world_id, "first_frame_difference": difference + 1})
    return {
        "schema": "t23-parity-comparison/v1",
        "pass": not errors,
        "errors": errors,
        "diagnostics": diagnostics,
        "mac_artifact_sha256": _digest(mac),
        "colab_artifact_sha256": _digest(colab),
    }


def markdown_report(result: Mapping[str, Any], mac_path: str, colab_path: str) -> str:
    """Render a short audit-oriented report; tie evidence never changes verdict."""
    lines = [
        "# T23 Mac--Colab parity",
        "",
        f"Verdict: **{'PASS' if result.get('pass') else 'FAIL'}**",
        "",
        f"Mac artifact: `{mac_path}`  ",
        f"Colab artifact: `{colab_path}`",
    ]
    if result.get("errors"):
        lines.extend(["", "## Differences", ""])
        lines.extend(f"- {item}" for item in result["errors"])
    if result.get("diagnostics"):
        lines.extend(
            [
                "",
                "## First-divergence diagnostics",
                "",
                "```json",
                json.dumps(result["diagnostics"], sort_keys=True, indent=2),
                "```",
            ]
        )
    return "\n".join(lines) + "\n"


@contextlib.contextmanager
def single_cpu_threads() -> Iterator[None]:
    """Bound the probe to one CPU execution lane and restore torch afterwards."""
    old_env = {
        name: os.environ.get(name)
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    }
    for name in old_env:
        os.environ[name] = "1"
    import torch

    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(old_threads)
        for name, value in old_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _action_packet(
    frame: int, slot: int, policy: str, action: int, q: np.ndarray, mask: np.ndarray
) -> dict[str, Any]:
    """Preserve float32 Q evidence so a later platform flip is explainable."""
    return {
        "frame": int(frame),
        "slot": int(slot),
        "policy": policy,
        "action": int(action),
        "q": np.asarray(q, dtype=np.float32).tolist(),
        "mask": np.asarray(mask, dtype=bool).tolist(),
    }


def _comparable_record(value: Mapping[str, Any]) -> dict[str, Any]:
    """Keep outcome facts, never machine-local checkpoint paths or provenance."""
    record = dict(value.get("record", value))
    keys = ("mass_integral", "survival_fraction", "deaths", "alive_frame_fraction", "no_death")
    return {key: record.get(key) for key in keys if key in record}


@contextlib.contextmanager
def _record_decisions(actions: list[dict[str, Any]], frame_digests: list[str]) -> Iterator[None]:
    """Read-only policy and GridBatchSim wrappers for one parity world.

    Original methods calculate and return the production action.  The duplicate
    Q calls below are diagnostic-only and never feed back into the simulator.
    """
    import src.simd_env.ego2s_policy as ep
    import src.simd_env.eval_engine as ee
    from research.redesign_scope_20261007.grid_h5000_identity import env_digest
    from src.simd_env.grid_sim import GridBatchSim

    original_step = GridBatchSim.step_with_policy
    original_ego = ep.Ego2sSimdPolicy.actions
    original_vector = ee.Vector61SimdPolicy.actions

    def stepped(sim: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_step(sim, *args, **kwargs)
        frame_digests.append(env_digest(sim, 0))
        return result

    def ego(self: Any, masks: np.ndarray, sim: Any, slots: np.ndarray) -> np.ndarray:
        # The original is called once; Q capture mirrors the same rowwise input.
        q = ep.student_q(self.net, sim, slots, self.forward).detach().cpu().numpy()
        rows = np.asarray(slots, dtype=np.int64).reshape(-1, 2)
        if self.mask_mode == "advisory":
            mask = sim.get_advisory_action_mask()[rows[:, 0], rows[:, 1]].astype(bool)
        else:
            mask = np.asarray(masks, dtype=bool).copy()
        mask[~mask.any(axis=1), :3] = True
        result = original_ego(self, masks, sim, slots)
        for row, value, eligible, selected in zip(rows, q, mask, result):
            actions.append(
                _action_packet(
                    sim.frame[int(row[0])], int(row[1]), "ego2s", int(selected), value, eligible
                )
            )
        return result

    def vector(self: Any, masks: np.ndarray, sim: Any, slots: np.ndarray) -> np.ndarray:
        states, exact = self.runtime.selection(np.asarray(slots).reshape(-1, 2))
        exact = exact.copy()
        exact[~exact.any(axis=1), :3] = True
        q = self.q_values(states).detach().cpu().numpy()
        result = original_vector(self, masks, sim, slots)
        for row, value, eligible, selected in zip(
            np.asarray(slots).reshape(-1, 2), q, exact, result
        ):
            actions.append(
                _action_packet(
                    sim.frame[int(row[0])], int(row[1]), "vector61", int(selected), value, eligible
                )
            )
        return result

    with (
        patch.object(GridBatchSim, "step_with_policy", stepped),
        patch.object(ep.Ego2sSimdPolicy, "actions", ego),
        patch.object(ee.Vector61SimdPolicy, "actions", vector),
    ):
        yield


def run_sample(bank: str | Path, b8: str | Path, out: str | Path) -> dict[str, Any]:
    """Run T23's reserved-world probe through the evaluator-owned adapter.

    ``evaluate.run_parity_sample`` owns the bank schema and exact evaluation
    semantics.  Keeping this bridge narrow prevents a diagnostic from becoming
    a second, subtly different evaluator.
    """
    bank_path, b8_path = Path(bank), Path(b8)
    if sha256_file(b8_path) != B8_SHA256:
        raise RuntimeError("B8 checkpoint bytes do not match the released SHA256 pin")
    from research.t23_r0_screen_20261017 import bank as bank_module
    from research.t23_r0_screen_20261017 import evaluate

    manifest = bank_module.load_world_bank(bank_path)
    worlds = list(manifest.get("parity_worlds", []))
    mix_counts = {mix: sum(1 for world in worlds if world.get("mix") == mix) for mix in MIX_COUNTS}
    if mix_counts != MIX_COUNTS or len(worlds) != 4:
        raise RuntimeError(f"T23 parity bank must reserve 2/1/1 worlds, got {mix_counts!r}")
    context = evaluate._context(1)
    profile = manifest.get("profile") or {}
    if (
        profile.get("base_profile_id") != "promotion-v2-watch-rect"
        or int(profile.get("horizon", 0)) != HORIZON
    ):
        raise RuntimeError("parity bank does not bind the H5000 promotion-v2-watch-rect profile")
    # B8 is the ego2s/raster hero and cannot be used as its own vector61
    # carrier.  The fixed released FRP3 vector policy supplies only evaluator
    # runtime plumbing; it never selects B8's actions.
    from research.redesign_scope_20261007 import grid_h5000_identity as gid

    carrier = tuple(context["lookup"][gid.FRP3_S12[1]])
    output_worlds: list[dict[str, Any]] = []
    with single_cpu_threads():
        for world in worlds:
            actions: list[dict[str, Any]] = []
            frame_digests: list[str] = []
            with _record_decisions(actions, frame_digests):
                result = evaluate.evaluate_world(
                    world, hero_checkpoint=b8_path, hero_carrier=carrier, context=context
                )
            record = result.get("record", result)
            output_worlds.append(
                {
                    "world_id": str(world.get("world_id", world["world_seed"])),
                    "world_seed": int(world["world_seed"]),
                    "mix": str(world["mix"]),
                    "roster_digest": str(world.get("roster_semantic_sha256", "")),
                    "world_digest": _digest(frame_digests),
                    "frame_digests": frame_digests,
                    "actions": actions,
                    "record": _comparable_record(record),
                }
            )
    artifact: dict[str, Any] = {
        "schema": SCHEMA,
        "horizon": HORIZON,
        "bank_sha256": sha256_file(bank_path),
        "b8_sha256": B8_SHA256,
        "profile_digest": str(
            manifest.get("profile_semantic_sha256", profile.get("base_profile_digest", ""))
        ),
        "source_sha256": _digest(
            {
                name: sha256_file(Path(__file__).parent / name)
                for name in ("parity.py", "evaluate.py", "bank.py")
            }
        ),
        "worlds": output_worlds,
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "cpu_threads": 1,
            "vector61_forward": "rowwise",
            "gpu_used": False,
        },
    }
    _write(out, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--bank", required=True)
    run.add_argument("--b8", required=True)
    run.add_argument("--out", required=True)
    compare = commands.add_parser("compare")
    compare.add_argument("--mac", required=True)
    compare.add_argument("--colab", required=True)
    compare.add_argument("--out-json", required=True)
    compare.add_argument("--out-md", required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        run_sample(args.bank, args.b8, args.out)
        return 0
    result = compare_artifacts(_load(args.mac), _load(args.colab))
    _write(args.out_json, result)
    Path(args.out_md).write_text(markdown_report(result, args.mac, args.colab), encoding="utf-8")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
