"""Production ``ReplaySource`` over hash-bound v2 teacher world exports.

Producer boundary (``research.t34_exit_b8_fit.interface``): every world is
regenerated with the ratified A2 regenerator ``replay.replay_world`` (all saved
state hashes must match before any observation leaves it), then turned into
fit rows.  Nothing here edits the fit loader, cache or frozen loss.

Payload (``payload.json`` in the export directory, schema ``t34-claim-a-payload/v1``)
lists each world export file with its SHA-256, size, split (``train`` or
``heldout``), ``round_age`` (0 = current round, >=1 = older DAgger rounds) and
``chunk_id``.  World files keep the ratified ``.sha256``/``.complete`` sidecars.

Row semantics:

* ``q_prev`` is the *previous frozen student* (the fit's initial model) on the
  regenerated observation, for every row of every round.  For rows whose
  export's rollout checkpoint is that same student, the recorded label Q must
  match it (``q_check``); a mismatch fails that world's proof, which the loader
  counts as a dropped world (>2% per chunk kills the fit).
* ``death_window_id`` for searched rows is the teacher's ``shadow_id``: one
  confirmed doom search per shadow fork, i.e. one death window.
* ``length`` is the hero length on the regenerated root.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch

from research.t34_exit_b8_fit.interface import ReplayWorld

PAYLOAD_SCHEMA = "t34-claim-a-payload/v1"
TEACHER_KERNEL = "shadow-lad-v2"
SEARCHED_KINDS = frozenset({"lad", "transition"})
_LABEL_FIELDS = (
    "decision",
    "label_kind",
    "t_star",
    "a_star",
    "F",
    "per_action",
    "student_action",
    "outcome_kind",
    "shadow_id",
)


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


# ---------------------------------------------------------------------------
# payload construction (Mac or runtime, pure file work)
# ---------------------------------------------------------------------------
class PreconditionFailed(ValueError):
    """The payload cannot support the registered fit (not a ratified tripwire)."""


def _split_token(world_id: str) -> int:
    token = hashlib.sha256(f"t34-claim-a-split/v1:{world_id}".encode()).digest()
    return int.from_bytes(token[:8], "big")


def _split_of(world_id: str, heldout_fraction: float, heldout_ids: set[str]) -> str:
    if heldout_ids:
        return "heldout" if world_id in heldout_ids else "train"
    return "heldout" if _split_token(world_id) / 2**64 < heldout_fraction else "train"


def _materialize(path: Path, out_dir: Path, chunk_id: str) -> Path:
    """Keep the payload tree self-contained: link (or copy) an outside export in."""
    import os
    import shutil

    source, root = path.resolve(), out_dir.resolve()
    if source.is_relative_to(root):
        return source
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in chunk_id) or "chunk"
    target_dir = root / "worlds" / safe
    target_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("", ".sha256", ".complete"):
        src = source.with_name(source.name + suffix)
        dst = target_dir / (source.name + suffix)
        if dst.exists():
            raise FileExistsError(f"payload world file collision: {dst}")
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    return target_dir / source.name


def build_payload(
    export_dirs: Sequence[tuple[Path, int, str]],
    out_dir: Path,
    *,
    heldout_fraction: float = 0.1,
    heldout_ids: Iterable[str] = (),
    heldout_decisive: int = 0,
    min_heldout_decisive_rows: int = 1,
) -> Path:
    """Write ``out_dir/payload.json`` indexing complete world exports.

    ``export_dirs`` holds ``(directory_or_world_file, round_age, chunk_id)``
    triples.  Exports outside ``out_dir`` are hard-linked (or copied) in with
    their sidecars, so the directory is one self-contained, shippable tree.

    Split precedence: explicit ``heldout_ids``; else ``heldout_decisive=K`` picks
    the K worlds (lowest split token) among those with >=1 decisive
    ``horizon_rescue`` label, so held-out decisive agreement is never vacuous
    (a qualification convenience; production splits are preregistered); else a
    deterministic sha256 world-id split at ``heldout_fraction``.

    A split whose held-out worlds carry fewer than ``min_heldout_decisive_rows``
    decisive labels raises :class:`PreconditionFailed` before any cache or fit
    work (``0`` disables the check).  Without it the ratified held-out scorer
    raises "stream is empty", which the ratified runner reports as a tripwire.
    """
    from research.t34_exit_b8.exports import load_world

    if not 0.0 <= heldout_fraction < 1.0:
        raise ValueError("heldout_fraction must lie in [0, 1)")
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "payload.json"
    if target.exists():
        raise FileExistsError(f"payload is create-only: {target}")
    heldout = set(heldout_ids)
    worlds, seen, decisive_ids = [], set(), set()
    carriers, rollouts = set(), set()
    for directory, round_age, chunk_id in export_dirs:
        if type(round_age) is not int or round_age < 0:
            raise ValueError("round_age must be a nonnegative integer")
        source = Path(directory)
        paths = [source] if source.is_file() else sorted(source.rglob("world-*.json"))
        if not paths:
            raise ValueError(f"no world exports under {directory}")
        for path in paths:
            exported = load_world(path)  # verifies the ratified sidecars
            world_id = str(exported["world_id"])
            if world_id in seen:
                raise ValueError(f"duplicate world id across exports: {world_id}")
            seen.add(world_id)
            if exported.get("teacher_v2", {}).get("kernel") != TEACHER_KERNEL:
                raise ValueError(f"{path}: Claim A fits require v2 shadow-LAD exports")
            decisive_rows = sum(bool(label.get("decisive")) for label in exported["labels"])
            if decisive_rows:
                decisive_ids.add(world_id)
            carriers.add(str(exported["checkpoint_sha256"]))
            rollouts.add(str(exported["rollout_policy_sha256"]))
            path = _materialize(Path(path), out_dir, str(chunk_id))
            worlds.append(
                {
                    "world_id": world_id,
                    "path": str(path.relative_to(out_dir.resolve())),
                    "sha256": sha256_file(Path(path)),
                    "size_bytes": Path(path).stat().st_size,
                    "split": _split_of(world_id, heldout_fraction, heldout),
                    "round_age": round_age,
                    "chunk_id": str(chunk_id),
                    "rollout_policy_sha256": str(exported["rollout_policy_sha256"]),
                    "decisive_rows": decisive_rows,
                }
            )
    if len(carriers) != 1:
        raise ValueError(f"all exports must share one carrier checkpoint, got {sorted(carriers)}")
    missing = heldout - seen
    if missing:
        raise ValueError(f"requested held-out worlds are absent: {sorted(missing)}")
    rule = "explicit heldout_ids" if heldout else f"sha256 world-id split, p={heldout_fraction}"
    if not heldout and heldout_decisive:
        if len(decisive_ids) <= heldout_decisive:
            raise ValueError(
                f"only {len(decisive_ids)} worlds carry decisive labels; cannot hold out "
                f"{heldout_decisive} and still train on one"
            )
        heldout = set(sorted(decisive_ids, key=_split_token)[:heldout_decisive])
        rule = f"{heldout_decisive} decisive-bearing worlds by sha256 split token"
        for world in worlds:
            world["split"] = "heldout" if world["world_id"] in heldout else "train"
    held_rows = sum(w["decisive_rows"] for w in worlds if w["split"] == "heldout")
    if held_rows < int(min_heldout_decisive_rows):
        raise PreconditionFailed(
            f"held-out split carries {held_rows} decisive rows; the registered minimum is "
            f"{min_heldout_decisive_rows} (split rule: {rule})"
        )
    payload = {
        "schema": PAYLOAD_SCHEMA,
        "carrier_checkpoint_sha256": carriers.pop(),
        "rollout_policy_sha256s": sorted(rollouts),
        "worlds": worlds,
        "split_rule": rule,
        "heldout_decisive_rows": held_rows,
        "min_heldout_decisive_rows": int(min_heldout_decisive_rows),
    }
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    with target.open("xb") as stream:
        stream.write(encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    (out_dir / "payload.json.sha256").write_text(f"{digest}  payload.json\n")
    (out_dir / "payload.json.complete").write_text("complete\n")
    return target


def load_payload(exports: Path) -> dict[str, Any]:
    """Load a payload only with its matching sha256 sidecar and completion marker."""
    payload = exports / "payload.json"
    sidecar = exports / "payload.json.sha256"
    if (
        not payload.is_file()
        or not sidecar.is_file()
        or not (exports / "payload.json.complete").is_file()
    ):
        raise ValueError("payload.json requires .sha256 and .complete sidecars")
    encoded = payload.read_bytes()
    if sidecar.read_text().split()[0] != hashlib.sha256(encoded).hexdigest():
        raise ValueError("payload.json hash mismatch")
    value = json.loads(encoded)
    if value.get("schema") != PAYLOAD_SCHEMA or not value.get("worlds"):
        raise ValueError("unknown or empty Claim A payload")
    return value


# ---------------------------------------------------------------------------
# replay -> rows
# ---------------------------------------------------------------------------
@dataclass
class SourceTimings:
    """Wall seconds spent inside replay, Q-prev forwards, and per-split world counts."""

    replay_seconds: float = 0.0
    q_prev_seconds: float = 0.0
    worlds: dict[str, int] = field(default_factory=dict)
    dropped: dict[str, int] = field(default_factory=dict)
    rows: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "replay_seconds": self.replay_seconds,
            "q_prev_seconds": self.q_prev_seconds,
            "worlds": dict(self.worlds),
            "dropped": dict(self.dropped),
            "rows": dict(self.rows),
        }


def _forward_q(
    model: torch.nn.Module, rows: Sequence[Mapping[str, Any]], device: torch.device, batch: int
) -> np.ndarray:
    from research.t34_claim_a.student import full_fp32
    from research.t34_exit_b8_fit.interface import forward_batch

    outputs = []
    keys = ("local", "global", "scalars") + (("full_sight",) if "full_sight" in rows[0] else ())
    model.eval()
    with torch.no_grad(), full_fp32():
        for start in range(0, len(rows), batch):
            part = rows[start : start + batch]
            tensors = {
                key: torch.from_numpy(np.stack([np.asarray(r[key]) for r in part])).to(device)
                for key in keys
            }
            outputs.append(forward_batch(model, tensors).float().cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float32)


class ExportReplaySource:
    """Regenerate hash-verified v2 world exports into fit rows, one world at a time."""

    is_fake = False

    def __init__(
        self,
        exports: Path,
        *,
        carrier_checkpoint: Path,
        previous_model: torch.nn.Module,
        previous_sha256: str | None,
        include_full_sight: bool = False,
        q_atol: float = 1e-3,
        q_rtol: float = 1e-4,
        q_batch: int = 1024,
        build_world: Any = None,
        replay_fn: Any = None,
    ) -> None:
        self.exports = Path(exports)
        self.payload = load_payload(self.exports)
        self.carrier_checkpoint = Path(carrier_checkpoint)
        carrier_sha = sha256_file(self.carrier_checkpoint)
        if carrier_sha != self.payload["carrier_checkpoint_sha256"]:
            raise ValueError("carrier checkpoint differs from the payload's world carrier")
        self.previous_model = previous_model
        self.previous_sha256 = previous_sha256
        self.include_full_sight = bool(include_full_sight)
        self.q_atol, self.q_rtol, self.q_batch = float(q_atol), float(q_rtol), int(q_batch)
        self.build_world = build_world
        self.replay_fn = replay_fn
        self.timings = SourceTimings()
        self._verified_counts: tuple[int, int] | None = None
        try:
            self.device = next(previous_model.parameters()).device
        except StopIteration:  # pragma: no cover - parameterless test doubles
            self.device = torch.device("cpu")
        self.worlds = sum(1 for w in self.payload["worlds"] if w["split"] == "train")
        if not self.worlds:
            raise ValueError("payload has no training worlds")

    # -- inventory ---------------------------------------------------------------
    def _entries(self, split: str) -> list[dict[str, Any]]:
        return [w for w in self.payload["worlds"] if w["split"] == split]

    def _path(self, entry: Mapping[str, Any]) -> Path:
        path = Path(entry["path"])
        return path if path.is_absolute() else self.exports / path

    def _load(self, entry: Mapping[str, Any]) -> dict[str, Any]:
        from research.t34_exit_b8.exports import load_world

        path = self._path(entry)
        if path.stat().st_size != entry["size_bytes"] or sha256_file(path) != entry["sha256"]:
            raise ValueError(f"world export bytes differ from payload: {entry['world_id']}")
        return load_world(path)

    def training_row_counts(self, epoch: int) -> tuple[int, int]:
        """Searched/unsearched counts of the verified retained training stream.

        Production (cached) path: the ratified ``build_cache`` asks only *after*
        its complete training replay, so these are exactly the verified worlds'
        counts and its drift check is satisfied by construction; the protection
        against silently dropped worlds there is the ratified >2%-per-chunk
        drop rule, not this inventory.  Uncached reference path: the loader asks
        first, so counts come from the exports and a later hash rejection raises
        the loader's count-drift error (the whole fit is invalid).
        """
        del epoch
        if self._verified_counts is not None:
            return self._verified_counts
        searched = unsearched = 0
        for entry in self._entries("train"):
            exported = self._load(entry)
            frames = len(exported["actions"])
            for label in exported["labels"]:
                if int(label["decision"]) >= frames:
                    continue
                if label.get("label_kind") in SEARCHED_KINDS:
                    searched += 1
                else:
                    unsearched += 1
        return searched, unsearched

    # -- regeneration ------------------------------------------------------------
    def _observe(self, world: Any, label: Mapping[str, Any]) -> dict[str, Any]:
        from research.t34_exit_b8.replay import _default_observation

        observation = _default_observation(world, label)
        observation["length"] = int(world.sim.length[0, 0])
        if self.include_full_sight:
            from research.t34_claim_a.student import full_sight_planes

            observation["full_sight"] = full_sight_planes(world.sim, np.array([[0, 0]]))[0].copy()
        return observation

    def _rows(
        self, entry: Mapping[str, Any], exported: Mapping[str, Any], observations: Sequence[dict]
    ) -> tuple[list[dict[str, Any]], str]:
        rows: list[dict[str, Any]] = []
        for observation in observations:
            label = observation["label"]
            kind = label.get("label_kind")
            if kind not in {"shadow_confirm", *SEARCHED_KINDS}:
                return [], f"unsupported label_kind {kind!r}"
            row = {key: label[key] for key in _LABEL_FIELDS if key in label}
            row.update(
                {
                    "local": observation["local"],
                    "global": observation["global"],
                    "scalars": observation["scalars"],
                    "round_age": int(entry["round_age"]),
                    "length": int(observation["length"]),
                    "world_id": str(exported["world_id"]),
                    "label_q": np.asarray(label["q"], dtype=np.float32),
                }
            )
            if "full_sight" in observation:
                row["full_sight"] = observation["full_sight"]
            if kind in SEARCHED_KINDS:
                row["death_window_id"] = int(label["shadow_id"])
            rows.append(row)
        if not rows:
            return [], "world has no labelled decisions"
        started = perf_counter()
        q_prev = _forward_q(self.previous_model, rows, self.device, self.q_batch)
        self.timings.q_prev_seconds += perf_counter() - started
        if not np.isfinite(q_prev).all():
            return [], "non-finite previous-student Q"
        same_policy = (
            self.previous_sha256 is not None
            and exported["rollout_policy_sha256"] == self.previous_sha256
        )
        if same_policy:
            recorded = np.stack([row["label_q"] for row in rows])
            error = np.abs(recorded - q_prev)
            bound = self.q_atol + self.q_rtol * np.abs(recorded)
            if not (error <= bound).all():
                return [], f"q_prev mismatch: max |dQ| {float(error.max()):.3g}"
        for row, q in zip(rows, q_prev):
            row["q_prev"] = q
            row.pop("label_q")
        return rows, ""

    def _replay(self, split: str) -> Iterator[ReplayWorld]:
        from research.t34_exit_b8.exports import state_hash
        from research.t34_exit_b8.replay import _default_builder, replay_world

        builder = self.build_world or _default_builder
        verified = {"searched": 0, "unsearched": 0}
        for entry in self._entries(split):
            started = perf_counter()
            self.timings.worlds[split] = self.timings.worlds.get(split, 0) + 1
            print(
                json.dumps(
                    {
                        "phase": "replay",
                        "split": split,
                        "world": self.timings.worlds[split],
                        "of": len(self._entries(split)),
                        "replay_seconds": round(self.timings.replay_seconds, 1),
                    }
                ),
                flush=True,
            )
            captured: list[Any] = []

            def build(row: Mapping[str, Any], checkpoint: Any) -> Any:
                world = builder(row, checkpoint)
                captured.append(world)
                return world

            world_id = str(entry["world_id"])
            try:
                exported = self._load(entry)
                if exported.get("teacher_v2", {}).get("kernel") != TEACHER_KERNEL:
                    raise ValueError("not a v2 shadow-LAD export")
                result = (self.replay_fn or replay_world)(
                    exported, self.carrier_checkpoint, build_world=build, observe=self._observe
                )
            except Exception as exc:  # malformed export: a dropped, reported world
                self.timings.replay_seconds += perf_counter() - started
                self._drop(split)
                yield ReplayWorld(
                    world_id, (), False, reason=f"malformed: {exc}", chunk_id=entry["chunk_id"]
                )
                continue
            if not result.accepted:
                self.timings.replay_seconds += perf_counter() - started
                self._drop(split)
                yield ReplayWorld(
                    world_id, (), False, reason=str(result.mismatch), chunk_id=entry["chunk_id"]
                )
                continue
            final = len(exported["actions"])
            expected_final = str(exported["hashes"][str(final)])
            actual_final = state_hash(captured[-1].sim)
            proof_expected = hashlib.sha256(
                f"{world_id}:{final}:{expected_final}".encode()
            ).hexdigest()
            proof_actual = hashlib.sha256(f"{world_id}:{final}:{actual_final}".encode()).hexdigest()
            self.timings.replay_seconds += perf_counter() - started
            rows, reason = self._rows(entry, exported, result.observations)
            del captured[:]
            if reason:
                self._drop(split)
                yield ReplayWorld(world_id, (), False, reason=reason, chunk_id=entry["chunk_id"])
                continue
            for row in rows:
                verified[
                    "searched" if row.get("label_kind") in SEARCHED_KINDS else "unsearched"
                ] += 1
            self.timings.rows[split] = self.timings.rows.get(split, 0) + len(rows)
            yield ReplayWorld(
                world_id=world_id,
                rows=rows,
                verified=True,
                hash_expected=proof_expected,
                hash_actual=proof_actual,
                chunk_id=str(entry["chunk_id"]),
            )
        if split == "train":
            self._verified_counts = (verified["searched"], verified["unsearched"])

    def _drop(self, split: str) -> None:
        self.timings.dropped[split] = self.timings.dropped.get(split, 0) + 1

    def replay_epoch(self, epoch: int) -> Iterable[ReplayWorld]:
        """Regenerate every training world (epoch-invariant, as the cache requires)."""
        del epoch
        return self._replay("train")

    def replay_heldout(self, epoch: int) -> Iterable[ReplayWorld]:
        del epoch
        if not self._entries("heldout"):
            raise ValueError("payload has no held-out worlds; decisive agreement is undefined")
        return self._replay("heldout")


# ---------------------------------------------------------------------------
# factory for the ratified fit runner (binding["source_factory"])
# ---------------------------------------------------------------------------
SOURCE_FACTORY = "research.t34_claim_a.replay_source:source_factory"
_LAST: list[ExportReplaySource] = []


def claim_a_settings(binding: Mapping[str, Any], exports: Path) -> dict[str, Any]:
    """Resolve the ``claim_a`` block of a binding (paths relative to ``exports``)."""
    settings = dict(binding.get("claim_a") or {})
    if "carrier_checkpoint" not in settings:
        raise ValueError("binding.claim_a.carrier_checkpoint is required to rebuild worlds")
    for key in ("carrier_checkpoint", "previous_checkpoint"):
        if settings.get(key):
            path = Path(str(settings[key]))
            settings[key] = path if path.is_absolute() else (exports / path).resolve()
    return settings


def source_factory(
    *, exports: Path, binding: Mapping[str, Any], previous_model: torch.nn.Module
) -> ExportReplaySource:
    """``module:function`` named by ``binding['source_factory']``."""
    settings = claim_a_settings(binding, Path(exports))
    previous_sha = settings.get("previous_sha256")
    if not isinstance(previous_sha, str) or len(previous_sha) != 64:
        # Without it the recorded-label Q identity check would be silently skipped.
        raise ValueError("binding.claim_a.previous_sha256 (64 hex) is required")
    source = ExportReplaySource(
        Path(exports),
        carrier_checkpoint=settings["carrier_checkpoint"],
        previous_model=previous_model,
        previous_sha256=previous_sha,
        include_full_sight=binding.get("arm") == "S1a",
        q_atol=float(settings.get("q_atol", 1e-3)),
        q_rtol=float(settings.get("q_rtol", 1e-4)),
    )
    _LAST.append(source)
    return source


def last_source() -> ExportReplaySource | None:
    return _LAST[-1] if _LAST else None
