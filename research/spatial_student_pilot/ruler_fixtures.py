"""Pinned natural mature C/T fixtures, exclusively for ruler engineering probes."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch

from research.sandbox_a_resident.ct_snapshots import LosslessBoundedSnapshotStore
from research.sandbox_a_resident.training_rosters import training_rosters

BANK = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-ct-h5000-1029/revision-02/"
    "attempt-02/output-a100-fallback/exports/ct/stage-0001.pt"
)
BANK_SHA256 = "112a2aa4c369ecd0e0f735db410e9ac5a7d330aca09846a50d1a734e6b3175f0"
FIXTURE_NAME = "ruler-mature-fixtures.pt"
THREAD_ENV = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "OMP_THREAD_LIMIT",
    "NUMEXPR_NUM_THREADS",
    "MAX_JOBS",
    "TORCHINDUCTOR_COMPILE_THREADS",
)


def make_fixtures(bank: Path = BANK) -> dict[str, Any]:
    """Select fixed longest ordinary living parent per mix, without reading labels.

    This bank is development evidence already inspected before the pilot. No
    scientific producer accepts these roots, namespaces or original parent IDs.
    """
    with bank.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != BANK_SHA256:
        raise ValueError("historical fixture bank SHA mismatch")
    blob = torch.load(bank, map_location="cpu", weights_only=False)
    if blob.get("schema") != "ct-stage-0001/v1":
        raise ValueError("foreign historical fixture schema")
    store = LosslessBoundedSnapshotStore.from_state_dict(blob["dataset"]["store"])
    table = training_rosters(1024)
    roster_lookup = {r.identity: i for i, r in reversed(list(enumerate(table)))}
    pools = {}
    for mix in ("scripted", "frozen", "mixed"):
        eligible = sorted(
            (
                i
                for i, m in enumerate(store.metadata)
                if m.mix == mix
                and m.kind == "ordinary"
                and bool(store.snapshots[i].small_tensors["alive"][0])
                and int(store.snapshots[i].small_tensors["length"][0]) >= 600
            ),
            key=lambda i: (-int(store.snapshots[i].small_tensors["length"][0]), i),
        )
        parents = set()
        pools[mix] = []
        for i in eligible:
            m = store.metadata[i]
            if m.world_id in parents:
                continue
            identity = store.snapshots[i].opponent_state["roster_identity"]
            roster_index = roster_lookup.get(identity)
            if roster_index is None or table[roster_index].mix != mix:
                raise ValueError("historical fixture roster identity differs")
            parents.add(m.world_id)
            pools[mix].append(
                dict(
                    historical_root_index=i,
                    historical_parent=m.world_id,
                    mix=mix,
                    roster_index=roster_index,
                    snapshot=store.snapshots[i].to_state_dict(),
                )
            )
    rows = []
    for i in range(43):
        for pool in pools.values():
            if len(rows) == 128:
                break
            if i >= len(pool):
                raise ValueError("insufficient distinct mature ordinary parents per mix")
            rows.append(pool[i])
    return dict(
        schema="spatial-ruler-fixtures/v1", development_only=True, bank_sha256=actual, rows=rows
    )


def load_fixtures(path: Path, mode: str) -> list[dict[str, Any]]:
    """Decode only in the disjoint ruler, even when main uploads identical bytes."""
    if mode != "ruler":
        raise ValueError("historical fixtures are ruler-only")
    data = torch.load(path, map_location="cpu", weights_only=False)
    if (
        data.get("schema") != "spatial-ruler-fixtures/v1"
        or data.get("development_only") is not True
        or data.get("bank_sha256") != BANK_SHA256
        or len(data.get("rows", [])) != 128
    ):
        raise ValueError("foreign ruler fixture input")
    return data["rows"]


def thread_evidence(*, enforce: bool) -> dict[str, Any]:
    """Retain and check the process environment as well as effective Torch pools."""
    import os

    env = {key: os.environ.get(key) for key in THREAD_ENV}
    result = dict(
        environment=env,
        torch_threads=torch.get_num_threads(),
        torch_interop_threads=torch.get_num_interop_threads(),
    )
    if enforce and (
        any(v != "1" for v in env.values())
        or result["torch_threads"] != 1
        or result["torch_interop_threads"] != 1
    ):
        raise ValueError("one-thread environment/effective Torch pools differ")
    return result


def warm_native(progress: Any) -> dict[str, Any]:
    """Attribute lazy CUDA builds to startup; use the very same production loaders."""
    import time

    from research.gpu_sim_spike import event_cuda, flood_cuda, move_cuda, vector_geometry

    def events() -> Any:
        if event_cuda._EXTENSION is not None:
            return event_cuda._EXTENSION
        # append_fifo owns an inline lazy loader rather than a callable loader.
        # Read its production arguments from the source inventory, unchanged.
        from torch.utils.cpp_extension import load

        from research.gpu_sim_spike.native_compile import ROOT, inventory

        _, specs = inventory(ROOT)
        spec = next(s for s in specs if s["name"] == "snake_dqn_a0_event_v1")
        args = {k: v for k, v in spec.items() if k != "sources"}
        args["sources"] = [str(ROOT / p) for p in spec["sources"]]
        args["verbose"] = False
        event_cuda._EXTENSION = load(**args)
        return event_cuda._EXTENSION

    rows = []
    for name, loader in (
        ("flood", flood_cuda._load),
        ("geometry", vector_geometry._load_exact_extension),
        ("move", move_cuda._extension),
        ("events", events),
    ):
        progress(dict(stage="native_startup", extension=name))
        started = time.monotonic()
        binary = loader()
        rows.append(
            dict(
                extension=name,
                seconds=time.monotonic() - started,
                binary=Path(binary.__file__).name,
            )
        )
    return dict(loaders=rows, seconds=sum(r["seconds"] for r in rows))
