"""R32-1 synthetic exported-byte consumer rehearsal.

This module deliberately consumes the existing F4 residual interfaces without
creating a scientific candidate or a serving approval.  Its sole successful
output is a synthetic, immutable trace that binds the raw factory-shaped input,
adapter, fused checkpoint and a separately reloaded masked-action consumer.

The T34 strict/serving handoff remains blocked: the available strict validator
requires a live-engine request and compatible T34 evidence schema, neither of
which is supplied by an ego2s residual export.  We record that boundary rather
than replacing it with a test-only approval.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import io
import json
import os
import platform
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from research.bl_ppo.exports import (
    digest,
    export_bytes,
    export_json,
    export_torch,
    limit_exports,
    verify_export,
)
from research.bl_residual.data import tensor_cache, validate_rows
from research.bl_residual.fit import FitRecipe, _resolved_action_mask, fit_residual
from research.bl_residual.heldout import validate_candidate
from research.bl_residual.job import synthetic_table
from research.bl_residual.model import FrozenB8Residual, fuse_residual
from research.bl_residual.protocol import B8_SHA
from research.mac_lease import DEFAULT_LEASE_PATH, held
from src.model.ego2s_network import ARCH, load_ego2s_checkpoint, spec_for
from web.backend.ego2s_policy import Ego2sServingPolicy

SCHEMA = "t34-r32-consumer-rehearsal/v1"
NAMESPACE = "t34-r32-rehearsal-synthetic-v1"
ALLOCATION = "r32-fixture-allocation-0000"


def _sha(value: Any) -> str:
    """Return a canonical finite JSON digest for trace-only identities."""
    return hashlib.sha256(
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def _checkpoint_payload(
    net: Any, *, residual_kind: str, recipe: Mapping[str, Any]
) -> dict[str, Any]:
    """Use the ordinary B8-compatible checkpoint envelope consumed by production loaders."""
    return {
        "obs_spec": spec_for(net),
        "arch": ARCH,
        "config": net.config,
        "state_dict": net.state_dict(),
        "meta": {
            "schema": "bl-residual-fit/v1",
            "synthetic": True,
            "namespace": NAMESPACE,
            "allocation": ALLOCATION,
            "index": 0,
            "dose": 12,
            "residual_kind": residual_kind,
            "recipe": dict(recipe),
            "updates": int(recipe["updates"]),
        },
    }


def _action_rows(checkpoint: Path, table_path: Path) -> list[int]:
    """Reload one delivered checkpoint and apply its served mask/fallback selection."""
    verify_export(checkpoint)
    with torch.no_grad():
        # Reuse the live web serving policy's selector; this is deliberately
        # not a copied masked-argmax implementation.
        policy = Ego2sServingPolicy(str(checkpoint), device="cpu")
        table = torch.load(table_path, map_location="cpu", weights_only=False)
        local, glob, scalars = (table[name] for name in ("local", "global", "scalars"))
        q = policy.net(local, glob, scalars).detach().cpu().numpy()
        policy._q = q
        policy._id_to_row = {index: index for index in range(q.shape[0])}
        resolved = _resolved_action_mask(table["action_mask"])
        return [
            policy.select_action(None, snake_id=index, action_mask=mask)
            for index, mask in enumerate(resolved)
        ]


def _reload_subprocess(output: Path, checkpoint: Path, table_path: Path) -> list[int]:
    """Require a new interpreter for the delivered-byte action observation."""
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "research.r32_rehearsal.consumer",
            "--reload-actions",
            "--checkpoint",
            str(checkpoint),
            "--table",
            str(table_path),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=60,
    )
    value = json.loads(completed.stdout)
    if not isinstance(value, list) or not all(type(item) is int for item in value):
        raise RuntimeError("reloaded action consumer emitted malformed output")
    return value


def _production_refusal() -> str:
    """Exercise the real F4 held-out admission guard against a synthetic candidate."""
    try:
        validate_candidate(
            {
                "schema": "bl-residual-fit/v1",
                "synthetic": True,
                "dose": 12,
                "index": 0,
                "updates": 3,
                "namespace": NAMESPACE,
                "allocation": ALLOCATION,
                "recipe_sha256": "0" * 64,
            },
            {"synthetic": True, "doses": [12], "namespace": NAMESPACE},
            Path("not-opened-allocation.json"),
        )
    except ValueError as exc:
        return str(exc)
    raise RuntimeError("synthetic candidate unexpectedly passed production admission")


def _blocked_edges(*, mac_world_smoke: bool) -> list[dict[str, str]]:
    """State the missing real consumer interfaces without invoking gameplay or serving."""
    edges = [
        {
            "edge": "strict_request_validator",
            "status": "BLOCKED",
            "reason": (
                "src.evaluation.strict_promotion.freeze_strict_request requires a live-engine "
                "T34 request plus E0/pilot/calibration/serving evidence; no ego2s residual-to-T34 "
                "adapter or compatible schema is registered"
            ),
        },
        {
            "edge": "independent_audit_and_registry_admission",
            "status": "BLOCKED",
            "reason": (
                "the available serving registry pins released checkpoints and strict receipts; "
                "this synthetic export has no approved identity or audit envelope"
            ),
        },
        {
            "edge": "three_excluded_worlds_and_watch6_smoke",
            "status": (
                "DEFERRED_ADMISSION_REQUIRED" if mac_world_smoke else "NOT_RUN_PENDING_ADMISSION"
            ),
            "reason": (
                "the default path withholds Mac execution pending a gap; --mac-smoke runs "
                "only an isolated synthetic H3/Watch plumbing exercise"
            ),
        },
    ]
    return edges


def _mac_h3_and_watch_smoke(b8_path: Path, candidate_path: Path) -> dict[str, Any]:
    """Run three isolated H3 fixture worlds and a tiny explicit-checkpoint Watch session.

    This does not resolve a registry name, create a pin, or claim registry
    admission.  The explicit checkpoint keeps the smoke isolated from serving
    selection while exercising the ordinary live `GameSession` Watch loop.
    """
    from research.bl_residual.gameplay import _build_exact_roots, _run_exact_arm_from_root
    from research.redesign_scope_20261007 import grid_h5000_identity as identity
    from research.t3_rollout_labels_20261015.bank_engine import planned_rosters
    from web.backend.session import GameSession

    context = identity._context(1)
    mixes = ("scripted", "frozen", "mixed")
    plans = planned_rosters(context, (913401, 913402, 913403), mixes)
    worlds = []
    for plan in plans:
        root, templates = _build_exact_roots(
            plan, {"B8": b8_path, "synthetic_candidate": candidate_path}
        )
        worlds.append(
            {
                "mix": plan["mix"],
                "seed": plan["seed"],
                "b8": _run_exact_arm_from_root(root, templates["B8"].hero, horizon=3),
                "candidate": _run_exact_arm_from_root(
                    root, templates["synthetic_candidate"].hero, horizon=3
                ),
            }
        )
    session = GameSession(checkpoint=str(candidate_path))
    for _ in range(3):
        session.step()
    return {
        "status": "SYNTHETIC_SMOKE_ONLY",
        "h3_worlds": worlds,
        "watch": {
            "mode": session.mode,
            "frames": int(session.game.frame),
            "explicit_checkpoint": True,
            "registry_admission_exercised": False,
            "leaderboard_write": False,
        },
    }


def _run(output: Path, b8_path: Path, *, mac_world_smoke: bool = False) -> dict[str, Any]:
    """Write one fresh, synthetic exported-byte trace; never run gameplay or serving.

    ``output`` must not exist.  The caller supplies the immutable B8 path so the
    trace records the bytes actually used, rather than a caller-provided digest.
    """
    output, b8_path = Path(output), Path(b8_path)
    if output.exists():
        raise FileExistsError("R32 consumer rehearsal requires a fresh output directory")
    if not b8_path.is_file():
        raise FileNotFoundError(f"B8 checkpoint is absent: {b8_path}")
    if digest(b8_path) != B8_SHA:
        raise ValueError("R32 rehearsal requires the pinned B8 s3 checkpoint bytes")
    if mac_world_smoke and platform.system() != "Darwin":
        raise RuntimeError("the optional world/Watch smoke declaration is Mac-only")

    torch.set_num_threads(1)
    torch.manual_seed(3403)
    output.mkdir(parents=True)
    recipe = dataclasses.asdict(
        FitRecipe(updates=3, batch_size=4, doom_batch_size=2, ordinary_batch_size=2)
    )
    with limit_exports(output, total_bytes=96 * 1024**2, file_bytes=32 * 1024**2):
        base = load_ego2s_checkpoint(str(b8_path))
        raw = synthetic_table(base)
        # The raw factory table, including lineage/status/physical fields, is
        # the immutable adapter input.  Reload it before constructing the cache.
        raw["action_mask"][0] = False
        raw["bare"][0] = raw["q"][0, :3].argmax()
        validate_rows(raw)
        raw_buffer = io.BytesIO()
        np.savez(raw_buffer, **raw)
        raw_path = output / "factory-format-labels.npz"
        raw_export = export_bytes(raw_path, raw_buffer.getvalue())
        ledger_path = output / "label-verified-ledger.json"
        ledger = {
            "schema": "t34-r32-rehearsal-ledger/v1",
            "namespace": NAMESPACE,
            "allocation": ALLOCATION,
            "units": [
                {
                    "unit_id": f"{NAMESPACE}/{ALLOCATION}/0",
                    "name": raw_path.name,
                    "sha256": raw_export["sha256"],
                    "size": raw_export["size"],
                }
            ],
        }
        export_json(ledger_path, ledger)
        if (
            verify_export(raw_path) != ledger["units"][0]["sha256"]
            or raw_path.stat().st_size != ledger["units"][0]["size"]
        ):
            raise RuntimeError("verified ledger does not bind the raw factory payload")
        with np.load(raw_path, allow_pickle=False) as archive:
            raw = {name: archive[name] for name in archive.files}
        validate_rows(raw)
        table_path = output / "consumer-observations.pt"
        export_torch(
            table_path,
            {
                name: torch.as_tensor(raw[name].copy())
                for name in ("local", "global", "scalars", "action_mask")
            },
        )
        cache = tensor_cache(raw, base)
        cache_manifest = export_json(
            output / "adapter-pair-builder.json",
            {
                "schema": SCHEMA,
                "synthetic": True,
                "factory_table_sha256": digest(raw_path),
                "adapter": "research.bl_residual.data.validate_rows",
                "pair_builder": "research.bl_residual.data.tensor_cache",
                "cache_fields": sorted(cache),
                "empty_mask_row": 0,
                "q_tie_row": 0,
            },
        )

        zero = fuse_residual(FrozenB8Residual(base))
        zero_path = output / "zero-residual-fused.pth"
        export_torch(zero_path, _checkpoint_payload(zero, residual_kind="zero", recipe=recipe))

        fitted = FrozenB8Residual(base)
        state = fit_residual(fitted, cache, FitRecipe(**recipe))
        if not torch.count_nonzero(fitted.residual.weight).item():
            raise RuntimeError("tiny synthetic fit did not produce the required nonzero residual")
        fit_state_path = output / "tiny-fit-state.pth"
        export_torch(
            fit_state_path,
            {
                "schema": "t34-r32-tiny-fit-state/v1",
                "synthetic": True,
                "base_checkpoint_sha256": digest(b8_path),
                "recipe": recipe,
                "residual": fitted.residual.state_dict(),
                "optimizer": state.optimizer_state,
                "rng": state.rng_state,
                "updates_completed": state.updates_completed,
                "actor_state": "N/A supervised residual fit",
                "replay_state": "N/A supervised residual fit",
                "resume_claim": "not an exact training resume",
            },
        )
        candidate = fuse_residual(fitted)
        candidate_path = output / "nonzero-residual-fused.pth"
        export_torch(
            candidate_path,
            _checkpoint_payload(candidate, residual_kind="deterministic_nonzero", recipe=recipe),
        )
        tie = fuse_residual(FrozenB8Residual(base))
        with torch.no_grad():
            tie.value.weight.zero_()
            tie.value.bias.zero_()
            tie.adv.weight.zero_()
            tie.adv.bias.zero_()
        tie_path = output / "tie-head-fused.pth"
        export_torch(
            tie_path, _checkpoint_payload(tie, residual_kind="isolated_q_tie", recipe=recipe)
        )
        actions = {
            "zero": _reload_subprocess(output, zero_path, table_path),
            "deterministic_nonzero": _reload_subprocess(output, candidate_path, table_path),
            "isolated_q_tie": _reload_subprocess(output, tie_path, table_path),
        }
        parent_actions = {
            name: _action_rows(path, table_path)
            for name, path in (
                ("zero", zero_path),
                ("deterministic_nonzero", candidate_path),
                ("isolated_q_tie", tie_path),
            )
        }
        if parent_actions != actions:
            raise RuntimeError("fresh-process and parent serving action consumers disagree")
        production_refusal = _production_refusal()
        trace = {
            "schema": SCHEMA,
            "synthetic": True,
            "scientific_admission": False,
            "release_eligible": False,
            "namespace": NAMESPACE,
            "allocation": ALLOCATION,
            "b8_sha256": digest(b8_path),
            "factory_table_sha256": digest(raw_path),
            "label_ledger_sha256": digest(ledger_path),
            "adapter_manifest_sha256": cache_manifest["sha256"],
            "recipe_sha256": _sha(recipe),
            "checkpoints": {
                "zero": {"sha256": digest(zero_path), "actions": actions["zero"]},
                "deterministic_nonzero": {
                    "sha256": digest(candidate_path),
                    "actions": actions["deterministic_nonzero"],
                    "fit_updates": state.updates_completed,
                    "unfused_fused_action_comparison": (
                        "not_identical_claimed; fused bytes are authoritative"
                    ),
                },
                "isolated_q_tie": {
                    "sha256": digest(tie_path),
                    "actions": actions["isolated_q_tie"],
                },
            },
            "production_admission_refusal": {
                "function": "research.bl_residual.heldout.validate_candidate",
                "error": production_refusal,
            },
            "fit_state_sha256": digest(fit_state_path),
            "source_sha256": {
                str(path.relative_to(Path(__file__).resolve().parents[2])): digest(path)
                for path in (
                    Path(__file__),
                    Path(__file__).parents[1] / "bl_residual/data.py",
                    Path(__file__).parents[1] / "bl_residual/fit.py",
                    Path(__file__).parents[1] / "bl_residual/model.py",
                    Path(__file__).parents[2] / "src/model/ego2s_network.py",
                    Path(__file__).parents[2] / "web/backend/ego2s_policy.py",
                )
            },
            "blocked_edges": _blocked_edges(mac_world_smoke=mac_world_smoke),
        }
        if mac_world_smoke:
            trace["mac_h3_and_watch_smoke"] = _mac_h3_and_watch_smoke(b8_path, candidate_path)
        export_json(output / "trace.json", trace)
        return trace


def run(output: Path, b8_path: Path, *, mac_world_smoke: bool = False) -> dict[str, Any]:
    """Acquire the Mac lease before every tiny fit or optional live smoke."""
    current_nice = os.getpriority(os.PRIO_PROCESS, 0)
    if current_nice < 15:
        os.nice(15 - current_nice)

    def expired(_signum: int, _frame: Any) -> None:
        raise TimeoutError("R32-1 2700-second preparation ceiling reached")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, 2700)
    try:
        with held(
            "t34-r32-consumer",
            "R32-1 synthetic consumer rehearsal",
            seconds=2700,
            path=DEFAULT_LEASE_PATH,
        ):
            return _run(output, b8_path, mac_world_smoke=mac_world_smoke)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--b8", type=Path)
    parser.add_argument(
        "--mac-world-smoke", "--mac-smoke", dest="mac_world_smoke", action="store_true"
    )
    parser.add_argument("--reload-actions", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--table", type=Path)
    args = parser.parse_args()
    if args.reload_actions:
        if (
            args.checkpoint is None
            or args.table is None
            or args.output is not None
            or args.b8 is not None
        ):
            parser.error("--reload-actions requires only --checkpoint and --table")
        print(json.dumps(_action_rows(args.checkpoint, args.table)))
        return
    if args.output is None or args.b8 is None:
        parser.error("--output and --b8 are required")
    result = run(args.output, args.b8, mac_world_smoke=args.mac_world_smoke)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
