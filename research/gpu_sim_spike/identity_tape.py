"""Frozen world design, lossless state mapping and tape files for the GPU-vs-CPU
closed-loop identity falsifier (JUDGMENT_POST_I0S_DIRECTION_20261024 step 3).

One tape describes one development world played closed-loop for at most
``horizon`` decisions.  Seam ``t`` is the prepared selector seam before decision
``t`` (after frame advance, food maintenance and respawn), which is where both
the CPU authority (``trajectory.prepare_world``) and ``DeviceSim.prepare`` stop.
A tape holds ``T`` decisions and ``T + 1`` seam states: the last state follows
the final transition, so a hero death is compared as a state.

The mapping from simulator objects to tape arrays is the pre-registered,
lossless comparison mapping (PREREGISTRATION.json).  Integers are widened to
int64 before comparison only; no sorting, deduplication, rounding or sentinel
erasure happens.  Float fields keep their original dtype and bytes.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

SCHEMA = "gpu-identity-tape/v1"
UNIT_SCHEMA = "gpu-identity-unit/v1"
LEDGER_SCHEMA = "gpu-identity-ledger/v1"
UNIT_DIRECTORY = "identity-units"
DOMAIN = "t34-exit-b8-20261019/v1"
MIXES = ("scripted", "frozen", "mixed")
MODES = {
    # 4 natural development worlds per mix, H1000, batch width 12.
    "main": {
        "namespace": "t34-gpu-identity-20261024/main/v1",
        "worlds": 12,
        "mixes": MIXES,
        "horizon": 1000,
        "replicas": 12,
    },
    # Excluded transport/execution canary worlds: disjoint namespace, never scored.
    "canary": {
        "namespace": "t34-gpu-identity-20261024/canary/v1",
        "worlds": 2,
        "mixes": ("mixed", "frozen"),
        "horizon": 40,
        "replicas": 12,
    },
    # Local Mac proof of the regenerator/comparator; never dispatched.
    "local": {
        "namespace": "t34-gpu-identity-20261024/local-proof/v1",
        "worlds": 3,
        "mixes": MIXES,
        "horizon": 12,
        "replicas": 2,
    },
}
SNAKE_FIELDS = (
    "head_x",
    "head_y",
    "head_ptr",
    "seg_count",
    "length",
    "alive",
    "direction",
    "boost_frames",
    "frames_since_food",
    "respawn_timer",
    "_reward_prev_length",
    "_boosted_this_step",
    "_last_done",
    "_last_death_cause",
    "_last_kills",
    "_last_food_ate",
    "_last_ambient_food_ate",
    "_last_corpse_food_ate",
    "_last_transition_valid",
)
# Ring-position fields of dead snakes are inert (no mask, move or feature reads
# them); they are masked to -1 on both sides before comparison.
DEAD_MASKED_FIELDS = ("head_x", "head_y")
_ARRAY_FIELDS = SNAKE_FIELDS[2:]
_UNIT = re.compile(r"world-[0-9]{2}-(?:scripted|frozen|mixed)")


def world_rows(mode: str) -> list[dict[str, Any]]:
    """Development world rows from the ratified T34 generator, fresh namespace."""
    from research.t34_exit_b8.trajectory import make_world_rows

    spec = MODES[mode]
    rows = make_world_rows(
        {"namespace": spec["namespace"], "mixes": spec["mixes"]}, int(spec["worlds"])
    )
    return rows


def unit_id(index: int, row: Mapping[str, Any]) -> str:
    return f"world-{index:02d}-{row['mix']}"


def allocation(mode: str) -> tuple[str, ...]:
    return tuple(unit_id(index, row) for index, row in enumerate(world_rows(mode)))


def namespace(mode: str, commit: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("identity namespace requires a full lowercase commit")
    return f"gpu-identity-{commit}-{mode}"


def build_cpu_world(row: Mapping[str, Any], checkpoint: Path) -> Any:
    """The production T34 world constructor, stopped at the first selector seam."""
    from research.t34_exit_b8.trajectory import build_world, prepare_world

    world = build_world(dict(row), Path(checkpoint))
    prepare_world(world)
    return world


# --------------------------------------------------------------------------- mapping


def body_digest(cells: np.ndarray) -> np.ndarray:
    """SHA-256 of the head-first logical live body as little-endian int64 pairs."""
    data = np.ascontiguousarray(np.asarray(cells, dtype="<i8").reshape(-1, 2)).tobytes()
    return np.frombuffer(hashlib.sha256(data).digest(), dtype=np.uint8).copy()


def tensor_digest(value: np.ndarray) -> np.ndarray:
    """Bytes digest bound to dtype and shape (used for Tier-D diagnostics)."""
    array = np.ascontiguousarray(value)
    header = f"{array.dtype.str}|{array.shape}|".encode()
    return np.frombuffer(hashlib.sha256(header + array.tobytes()).digest(), dtype=np.uint8).copy()


def _logical_body(bodies: np.ndarray, head_ptr: int, count: int, cap: int) -> np.ndarray:
    rings = (int(head_ptr) - np.arange(int(count))) % int(cap)
    return np.asarray(bodies)[rings]


def cpu_seam_state(sim: Any) -> dict[str, np.ndarray]:
    """Map a one-environment GridBatchSim at its prepared seam to tape arrays."""
    if int(sim.E) != 1:
        raise ValueError("identity mapping requires a one-environment CPU world")
    slots, cap = int(sim.S), int(sim.cap)
    snake = np.zeros((slots, len(SNAKE_FIELDS)), dtype=np.int64)
    digests = np.zeros((slots, 32), dtype=np.uint8)
    for s in range(slots):
        alive = bool(sim.alive[0, s])
        head = np.asarray(sim.bodies[0, s, int(sim.head_ptr[0, s])], dtype=np.int64)
        values = [int(head[0]), int(head[1])] + [
            int(np.asarray(getattr(sim, name))[0, s]) for name in _ARRAY_FIELDS
        ]
        snake[s] = values
        if alive:
            digests[s] = body_digest(
                _logical_body(sim.bodies[0, s], sim.head_ptr[0, s], sim.seg_count[0, s], cap)
            )
    cells = list(sim.food_cells[0])
    corpse = sim.corpse_cells[0]
    food = np.asarray(
        [(int(c), int(r), int((c, r) in corpse)) for c, r in cells], dtype=np.int64
    ).reshape(-1, 3)
    return {
        "snake": _mask_dead(snake),
        "body": digests,
        "frame": np.asarray(int(sim.frame[0]), dtype=np.int64),
        "food": food,
        "reward": np.asarray(sim._last_reward[0]).copy(),
    }


def device_seam_state(sim: Any, env: int = 0) -> dict[str, np.ndarray]:
    """Map one DeviceSim environment (normally replica 0) to tape arrays."""
    import torch

    slots, cap = int(sim.S), int(sim.cap)
    with torch.inference_mode():
        head_ptr = sim.head_ptr[env].detach().cpu().numpy()
        seg = sim.seg_count[env].detach().cpu().numpy()
        alive = sim.alive[env].detach().cpu().numpy().astype(bool)
        longest = int(seg.max()) if seg.size else 0
        k = torch.arange(max(longest, 1), device=sim.device)
        ring = (sim.head_ptr[env][:, None] - k[None, :]) % cap
        gathered = (
            sim.bodies[env].gather(1, ring[..., None].expand(-1, -1, 2)).detach().cpu().numpy()
        )
        heads = sim.bodies[env, torch.arange(slots, device=sim.device), sim.head_ptr[env]]
        heads = heads.detach().cpu().numpy()
        arrays = {name: getattr(sim, name)[env].detach().cpu().numpy() for name in _ARRAY_FIELDS}
        count = int(sim.food_count[env].detach().cpu())
        cells = sim.food_cells[env, :count].detach().cpu().numpy().astype(np.int64)
        corpse = sim.corpse_food[env, :count].detach().cpu().numpy().astype(np.int64)
        reward = sim._last_reward[env].detach().cpu().numpy().copy()
        frame = int(sim.frame[env].detach().cpu())
    snake = np.zeros((slots, len(SNAKE_FIELDS)), dtype=np.int64)
    digests = np.zeros((slots, 32), dtype=np.uint8)
    for s in range(slots):
        snake[s] = [int(heads[s, 0]), int(heads[s, 1])] + [
            int(arrays[name][s]) for name in _ARRAY_FIELDS
        ]
        if alive[s]:
            digests[s] = body_digest(gathered[s, : int(seg[s])])
    del head_ptr
    food = np.concatenate((cells.reshape(-1, 2), corpse.reshape(-1, 1)), axis=1)
    return {
        "snake": _mask_dead(snake),
        "body": digests,
        "frame": np.asarray(frame, dtype=np.int64),
        "food": food.reshape(-1, 3),
        "reward": reward,
    }


def _mask_dead(snake: np.ndarray) -> np.ndarray:
    alive = snake[:, SNAKE_FIELDS.index("alive")] != 0
    for name in DEAD_MASKED_FIELDS:
        snake[~alive, SNAKE_FIELDS.index(name)] = -1
    return snake


# --------------------------------------------------------------------------- tapes


class TapeWriter:
    """Accumulate seam states and decisions; finalize into one npz payload."""

    def __init__(self, slots: int) -> None:
        self.slots = int(slots)
        self.states: list[dict[str, np.ndarray]] = []
        self.decisions: list[dict[str, np.ndarray]] = []
        self.replica_agree: list[bool] = []

    def state(self, value: Mapping[str, np.ndarray], replicas_agree: bool = True) -> None:
        if len(self.states) != len(self.decisions):
            raise ValueError("tape state/decision order violated")
        self.states.append(dict(value))
        self.replica_agree.append(bool(replicas_agree))

    def decision(self, value: Mapping[str, np.ndarray]) -> None:
        if len(self.states) != len(self.decisions) + 1:
            raise ValueError("tape decision without a preceding seam state")
        self.decisions.append(dict(value))

    def arrays(self) -> dict[str, np.ndarray]:
        if len(self.states) != len(self.decisions) + 1:
            raise ValueError("tape must end with a seam state")
        slots = self.slots
        steps = len(self.decisions)
        width = max((len(row["food"]) for row in self.states), default=0)
        food = np.full((len(self.states), max(width, 1), 3), -1, dtype=np.int64)
        for index, row in enumerate(self.states):
            food[index, : len(row["food"])] = row["food"]
        out = {
            "state_snake": np.stack([row["snake"] for row in self.states]),
            "state_body": np.stack([row["body"] for row in self.states]),
            "state_frame": np.stack([row["frame"] for row in self.states]),
            "state_food_count": np.asarray(
                [len(row["food"]) for row in self.states], dtype=np.int64
            ),
            "state_food": food,
            # Widened to a common dtype for storage (lossless); originals recorded.
            "state_reward": np.stack([row["reward"] for row in self.states]),
            "state_reward_dtype": np.asarray([str(row["reward"].dtype) for row in self.states]),
            "replica_state_agree": np.asarray(self.replica_agree, dtype=bool),
        }
        empty = {
            "hero_q": np.zeros((0, 6), dtype=np.float32),
            "hero_mask": np.zeros((0, 6), dtype=bool),
            "actions": np.zeros((0, slots), dtype=np.int64),
            "obs_digest": np.zeros((0, 3, 32), dtype=np.uint8),
            "opp_q": np.zeros((0, slots, 6), dtype=np.float32),
            "opp_mask": np.zeros((0, slots, 6), dtype=bool),
            "opp_state_digest": np.zeros((0, slots, 32), dtype=np.uint8),
            "opp_recorded": np.zeros((0, slots), dtype=bool),
            "replica_decision_agree": np.zeros((0,), dtype=bool),
        }
        for key, value in empty.items():
            out[key] = (
                np.stack([np.asarray(row[key], dtype=value.dtype) for row in self.decisions])
                if steps
                else value
            )
        return out


def empty_decision(slots: int) -> dict[str, np.ndarray]:
    return {
        "hero_q": np.zeros(6, dtype=np.float32),
        "hero_mask": np.zeros(6, dtype=bool),
        "actions": np.zeros(slots, dtype=np.int64),
        "obs_digest": np.zeros((3, 32), dtype=np.uint8),
        "opp_q": np.full((slots, 6), np.nan, dtype=np.float32),
        "opp_mask": np.zeros((slots, 6), dtype=bool),
        "opp_state_digest": np.zeros((slots, 32), dtype=np.uint8),
        "opp_recorded": np.zeros(slots, dtype=bool),
        "replica_decision_agree": np.asarray(True),
    }


def encode_tape(arrays: Mapping[str, np.ndarray], meta: Mapping[str, Any]) -> bytes:
    stream = io.BytesIO()
    payload = {key: np.asarray(value) for key, value in sorted(arrays.items())}
    payload["meta_json"] = np.frombuffer(
        json.dumps(meta, sort_keys=True, allow_nan=False).encode(), dtype=np.uint8
    )
    np.savez_compressed(stream, **payload)
    return stream.getvalue()


def decode_tape(data: bytes) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(io.BytesIO(data), allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    meta = json.loads(arrays.pop("meta_json").tobytes().decode())
    return arrays, meta


# --------------------------------------------------------------------------- publication


def _atomic_create(path: Path, payload: bytes) -> None:
    from research.gpu_sim_spike.a1_prof_units import _atomic_create as create

    create(path, payload)


def publish_bytes(path: Path, payload: bytes) -> str:
    """Create payload, ``.sha256`` and ``.complete`` (last), all create-only."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for candidate in (path, _sidecar(path, ".sha256"), _sidecar(path, ".complete")):
        if candidate.exists():
            raise FileExistsError(f"identity export already exists: {candidate}")
    digest = hashlib.sha256(payload).hexdigest()
    _atomic_create(path, payload)
    _atomic_create(_sidecar(path, ".sha256"), f"{digest}  {path.name}\n".encode())
    _atomic_create(_sidecar(path, ".complete"), b"complete\n")
    return digest


def publish_json(path: Path, value: Mapping[str, Any]) -> str:
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    return publish_bytes(path, payload)


def verify(path: Path) -> str:
    """Return the payload SHA-256 after checking both sidecars."""
    from research.gpu_sim_spike.a1_prof_units import _verify_sidecars

    return _verify_sidecars(Path(path))


def _sidecar(path: Path, suffix: str) -> Path:
    return path.with_name(path.name + suffix)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def unit_paths(root: Path, unit: str) -> tuple[Path, Path]:
    if not _UNIT.fullmatch(unit):
        raise ValueError(f"malformed identity unit id: {unit}")
    base = Path(root) / UNIT_DIRECTORY
    return base / f"{unit}.npz", base / f"{unit}.json"


def publish_unit(
    root: Path,
    unit_namespace: str,
    unit_allocation: tuple[str, ...],
    unit: str,
    tape: bytes,
    content: Mapping[str, Any],
) -> Path:
    """Publish the npz first, then the hash-binding unit record (published last)."""
    if unit not in unit_allocation:
        raise ValueError("unit is outside the frozen identity allocation")
    tape_path, record_path = unit_paths(root, unit)
    digest = publish_bytes(tape_path, tape)
    publish_json(
        record_path,
        {
            "schema": UNIT_SCHEMA,
            "namespace": unit_namespace,
            "allocation": list(unit_allocation),
            "unit_id": unit,
            "tape": {
                "path": f"{UNIT_DIRECTORY}/{tape_path.name}",
                "sha256": digest,
                "size": len(tape),
            },
            "content": dict(content),
        },
    )
    return record_path


def load_unit(
    root: Path, unit_namespace: str, unit_allocation: tuple[str, ...], unit: str
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    """Verify sidecars, namespace, allocation and tape hash; return record + tape."""
    tape_path, record_path = unit_paths(root, unit)
    verify(record_path)
    record = json.loads(record_path.read_text())
    if (
        record.get("schema") != UNIT_SCHEMA
        or record.get("namespace") != unit_namespace
        or record.get("allocation") != list(unit_allocation)
        or record.get("unit_id") != unit
    ):
        raise ValueError(f"foreign identity unit record: {record_path}")
    digest = verify(tape_path)
    tape = record.get("tape", {})
    if (
        tape.get("path") != f"{UNIT_DIRECTORY}/{tape_path.name}"
        or tape.get("sha256") != digest
        or tape.get("size") != tape_path.stat().st_size
    ):
        raise ValueError(f"identity tape binding differs: {tape_path}")
    arrays, meta = decode_tape(tape_path.read_bytes())
    if meta.get("unit_id") != unit or meta.get("namespace") != unit_namespace:
        raise ValueError(f"identity tape metadata differs: {tape_path}")
    return record, arrays, meta


def empty_ledger(unit_namespace: str, unit_allocation: tuple[str, ...]) -> dict[str, Any]:
    return {
        "schema": LEDGER_SCHEMA,
        "namespace": unit_namespace,
        "allocation": list(unit_allocation),
        "units": [],
        "published_last": "units",
    }


def load_ledger(
    path: Path,
    unit_namespace: str,
    unit_allocation: tuple[str, ...],
    exports: Path | None = None,
) -> set[str]:
    """Verified units to skip; paths/sizes/hashes only, never tape contents.

    The remote producer passes ``exports=None``: earlier attempts' payloads stay
    on the Mac, which re-verifies them (``exports`` set) before any reduction.
    """
    verify(path)
    ledger = json.loads(Path(path).read_text())
    if (
        ledger.get("schema") != LEDGER_SCHEMA
        or ledger.get("namespace") != unit_namespace
        or ledger.get("allocation") != list(unit_allocation)
    ):
        raise ValueError("foreign identity verified ledger")
    found: set[str] = set()
    for row in ledger.get("units", []):
        unit = row.get("unit_id")
        if unit in found or unit not in unit_allocation:
            raise ValueError("malformed or foreign identity ledger row")
        for kind in ("tape", "record"):
            entry = row.get(kind) or {}
            if (
                entry.get("path")
                != f"{UNIT_DIRECTORY}/{unit}.{'npz' if kind == 'tape' else 'json'}"
            ):
                raise ValueError(f"identity ledger path differs: {unit}")
            if exports is None:
                continue
            path_value = exports / entry["path"]
            if (
                not path_value.is_file()
                or verify(path_value) != entry.get("sha256")
                or path_value.stat().st_size != entry.get("size")
            ):
                raise ValueError(f"identity ledger payload differs: {unit}")
        found.add(unit)
    return found


def build_ledger(
    exports: Path, unit_namespace: str, unit_allocation: tuple[str, ...]
) -> dict[str, Any]:
    """Ledger of complete units from paths/sizes/hashes only (outcome-blind)."""
    ledger = empty_ledger(unit_namespace, unit_allocation)
    for unit in unit_allocation:
        tape_path, record_path = unit_paths(exports, unit)
        try:
            record_digest = verify(record_path)
            tape_digest = verify(tape_path)
        except (ValueError, OSError):
            continue
        ledger["units"].append(
            {
                "unit_id": unit,
                "tape": {
                    "path": f"{UNIT_DIRECTORY}/{tape_path.name}",
                    "sha256": tape_digest,
                    "size": tape_path.stat().st_size,
                },
                "record": {
                    "path": f"{UNIT_DIRECTORY}/{record_path.name}",
                    "sha256": record_digest,
                    "size": record_path.stat().st_size,
                },
            }
        )
    return ledger


def single_thread_environment() -> dict[str, str]:
    return {
        name: os.environ.get(name, "")
        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMBA_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        )
    }


def write_text_once(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        stream.write(text)
        temporary = Path(stream.name)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
