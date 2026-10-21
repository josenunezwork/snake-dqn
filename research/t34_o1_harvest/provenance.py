"""Side-state food provenance for the T34 O1 harvest census.

The simulator deliberately exposes only ambient-versus-corpse telemetry.  O1
needs the two v2 corpse origins separately without changing simulator state or
the order in which it consumes RNG.  :class:`ProvenanceObserver` therefore
wraps *one simulator instance* and mirrors successful food mutations in Python
side state.  The wrapped simulator methods are always called exactly once.

It is intentionally a measurement aid, not a policy or a simulator subclass.
Use ``detached()`` while cloning a captured world, then attach a fresh observer
with ``clone_for``.  Copying an instance with closures bound to the source
simulator is unsafe.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from types import MethodType
from typing import Any, Dict, Iterator, Mapping, Sequence, Tuple

import numpy as np

from src.core.mechanics_constants import corpse_food_cap

AMBIENT = "ambient"
DEATH_DROP = "death_drop"
BOOST_PELLET = "boost_pellet"
SOURCES = (AMBIENT, DEATH_DROP, BOOST_PELLET)

CellKey = Tuple[int, Tuple[int, int]]


class ProvenanceObserver:
    """Observe hero food intake and boost costs without altering dynamics.

    Args:
        sim: A ``BatchSim``/``GridBatchSim`` instance.  Its methods are wrapped
            on this instance only, so sibling worlds remain uninstrumented.
        hero: Slot whose intake is reported.  O1 uses slot zero.

    The observer is reset at construction.  Call :meth:`begin` at a branch root
    to retain the current food-origin map while resetting branch measurements.
    """

    _METHODS = ("_add_food", "_drop_corpse_v2", "_drop_trail_pellets", "_consume_food", "_move_all")

    def __init__(self, sim: Any, hero: int = 0) -> None:
        if not 0 <= hero < int(sim.S):
            raise ValueError(f"hero slot {hero} outside simulator slots [0, {sim.S})")
        self.sim = sim
        self.hero = int(hero)
        self._original: Dict[str, Any] = {}
        self._original_instance: Dict[str, Any] = {}
        self._installed = False
        self._source_context: str | None = None
        self._sources: Dict[CellKey, str] = {}
        self._pending_boost_cost = np.zeros(int(sim.E), dtype=np.int64)
        self.begin()
        self._seed_sources_from_sim()
        self.install()

    # ------------------------------------------------------------------
    # Installation and clone lifecycle
    # ------------------------------------------------------------------
    def install(self) -> "ProvenanceObserver":
        """Install per-instance wrappers, refusing to stack observers."""
        if self._installed:
            return self
        owner = getattr(self.sim, "_t34_o1_provenance_observer", None)
        if owner is not None and owner is not self:
            raise RuntimeError("a ProvenanceObserver is already installed on this simulator")
        marker = object()
        self._original = {name: getattr(self.sim, name) for name in self._METHODS}
        self._original_instance = {name: vars(self.sim).get(name, marker) for name in self._METHODS}
        self._missing_instance_method = marker
        setattr(self.sim, "_t34_o1_provenance_observer", self)
        self.sim._add_food = MethodType(self._wrap_add_food, self.sim)
        self.sim._drop_corpse_v2 = MethodType(self._wrap_drop_corpse_v2, self.sim)
        self.sim._drop_trail_pellets = MethodType(self._wrap_drop_trail_pellets, self.sim)
        self.sim._consume_food = MethodType(self._wrap_consume_food, self.sim)
        self.sim._move_all = MethodType(self._wrap_move_all, self.sim)
        self._installed = True
        return self

    def uninstall(self) -> None:
        """Restore original bound methods before cloning or serialising ``sim``."""
        if not self._installed:
            return
        for name in self._original:
            original = self._original_instance[name]
            if original is self._missing_instance_method:
                # The common case: the original resolved through the class.
                # Deleting our override restores descriptor binding and avoids
                # serialising a bound method tied to this simulator instance.
                delattr(self.sim, name)
            else:
                setattr(self.sim, name, original)
        if getattr(self.sim, "_t34_o1_provenance_observer", None) is self:
            delattr(self.sim, "_t34_o1_provenance_observer")
        self._installed = False

    @contextmanager
    def detached(self) -> Iterator[Any]:
        """Temporarily remove wrappers so exact-world cloning is closure-free."""
        was_installed = self._installed
        self.uninstall()
        try:
            yield self.sim
        finally:
            if was_installed:
                self.install()

    def clone_for(self, sim: Any) -> "ProvenanceObserver":
        """Attach an independent observer to a clone with copied side state.

        The caller must create ``sim`` while this observer is detached.  This
        method checks the native food state before installing wrappers, so a
        clone that is not state-identical fails close to the mistake.
        """
        clone = self.__class__.__new__(self.__class__)
        clone.sim = sim
        clone.hero = self.hero
        clone._original = {}
        clone._original_instance = {}
        clone._missing_instance_method = object()
        clone._installed = False
        clone._source_context = None
        clone._sources = dict(self._sources)
        clone._pending_boost_cost = self._pending_boost_cost.copy()
        clone._initial_length = self._initial_length.copy()
        clone._intake = self._intake.copy()
        clone._boost_cost = self._boost_cost.copy()
        clone._weighted_intake = self._weighted_intake.copy()
        clone._weighted_boost_cost = self._weighted_boost_cost.copy()
        clone._events = copy.deepcopy(self._events)
        clone._reconcile()
        clone.assert_food_accounting()
        clone.install()
        return clone

    # ------------------------------------------------------------------
    # Branch measurements
    # ------------------------------------------------------------------
    def begin(self, initial_stored_length: Sequence[int] | None = None) -> None:
        """Start a branch ledger without discarding food-origin side state."""
        e = int(self.sim.E)
        if initial_stored_length is None:
            initial = np.asarray(self.sim.length[:, self.hero], dtype=np.int64).copy()
        else:
            initial = np.asarray(initial_stored_length, dtype=np.int64)
            if initial.shape != (e,):
                raise ValueError(
                    f"initial_stored_length must have shape {(e,)}, got {initial.shape}"
                )
        self._initial_length = initial
        self._intake = np.zeros((e, len(SOURCES)), dtype=np.int64)
        self._boost_cost = np.zeros(e, dtype=np.int64)
        self._weighted_intake = np.zeros((e, len(SOURCES)), dtype=np.int64)
        self._weighted_boost_cost = np.zeros(e, dtype=np.int64)
        self._events: list[dict[str, Any]] = []
        self._pending_boost_cost = np.zeros(e, dtype=np.int64)

    reset = begin

    def ledger(self) -> dict[str, Any]:
        """Return JSON-safe per-step and aggregate hero provenance facts."""
        per_env = []
        for env in range(int(self.sim.E)):
            sources = {source: int(self._intake[env, idx]) for idx, source in enumerate(SOURCES)}
            weighted = {
                source: int(self._weighted_intake[env, idx]) for idx, source in enumerate(SOURCES)
            }
            per_env.append(
                {
                    "env": env,
                    "initial_stored_length": int(self._initial_length[env]),
                    "sources": sources,
                    "cumulative_intake": sources.copy(),
                    "weighted_cumulative_intake": weighted,
                    "boost_cost": int(self._boost_cost[env]),
                    "weighted_cumulative_boost_cost": int(self._weighted_boost_cost[env]),
                    "final_stored_length": int(self.sim.length[env, self.hero]),
                }
            )
        aggregate_sources = {
            source: int(self._intake[:, idx].sum()) for idx, source in enumerate(SOURCES)
        }
        weighted_sources = {
            source: int(self._weighted_intake[:, idx].sum()) for idx, source in enumerate(SOURCES)
        }
        return {
            "hero": self.hero,
            "sources": aggregate_sources,
            "cumulative_intake": aggregate_sources.copy(),
            "weighted_cumulative_intake": weighted_sources,
            "boost_cost": int(self._boost_cost.sum()),
            "weighted_cumulative_boost_cost": int(self._weighted_boost_cost.sum()),
            "events": [dict(row) for row in self._events],
            "per_env": per_env,
        }

    def state_dict(self) -> dict[str, Any]:
        """Portable side-state snapshot for a branch checkpoint or diagnostic."""
        return {
            "hero": self.hero,
            "sources": [
                {"env": env, "cell": list(cell), "source": source}
                for (env, cell), source in sorted(self._sources.items())
            ],
            "ledger": self.ledger(),
        }

    def assert_food_accounting(self) -> None:
        """Assert side-state food coverage and valid native corpse classification."""
        native: set[CellKey] = set()
        for env, cells in enumerate(self.sim.food_cells):
            native.update((env, (int(cell[0]), int(cell[1]))) for cell in cells)
        if set(self._sources) != native:
            missing = native - set(self._sources)
            stale = set(self._sources) - native
            raise AssertionError(
                f"provenance food map diverged: missing={len(missing)} stale={len(stale)}"
            )
        for (env, cell), source in self._sources.items():
            if source not in SOURCES:
                raise AssertionError(f"unknown source {source!r} for env={env} cell={cell}")
            if bool(self.sim.v2):
                native_corpse = cell in self.sim.corpse_cells[env]
                observed_corpse = source in (DEATH_DROP, BOOST_PELLET)
                if native_corpse != observed_corpse:
                    raise AssertionError(
                        "provenance corpse tag diverged: "
                        f"env={env} cell={cell} source={source} native_corpse={native_corpse}"
                    )

    def assert_accounting(self) -> None:
        """Assert food-map consistency and per-env hero mass-balance identity."""
        self.assert_food_accounting()
        expected = self._initial_length + self._intake.sum(axis=1) - self._boost_cost
        actual = np.asarray(self.sim.length[:, self.hero], dtype=np.int64)
        if not np.array_equal(expected, actual):
            raise AssertionError(
                "hero mass accounting failed: "
                f"expected={expected.tolist()} actual={actual.tolist()} "
                "(a hero respawn requires a new branch ledger)"
            )

    # ------------------------------------------------------------------
    # O1 pairing utility
    # ------------------------------------------------------------------
    @staticmethod
    def death_attributed_delta(
        alternative: Mapping[str, Any], bare: Mapping[str, Any], horizon: int
    ) -> float:
        """Return O1's weighted death-drop delta for a paired H-frame branch."""
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        try:
            alt = int(alternative["weighted_cumulative_intake"][DEATH_DROP])
            ref = int(bare["weighted_cumulative_intake"][DEATH_DROP])
        except KeyError as error:
            raise ValueError("ledger lacks weighted death-drop intake") from error
        return float(alt - ref) / float(horizon)

    # ------------------------------------------------------------------
    # Wrapped native methods
    # ------------------------------------------------------------------
    def _wrap_add_food(self, _sim: Any, e: int, cell: Tuple[int, int], corpse: bool) -> bool:
        result = self._original["_add_food"](e, cell, corpse)
        # The original returns False on a deduplicated insert.  Preserve the
        # existing tag in that case; successful new inserts get the active kind.
        if result:
            self._sources[(int(e), (int(cell[0]), int(cell[1])))] = self._source_context or AMBIENT
        self._reconcile()
        return result

    def _wrap_drop_corpse_v2(self, _sim: Any, e: int, cells: Sequence[Tuple[int, int]]) -> None:
        before_cells = list(self.sim.food_cells[int(e)])
        before_sources = {
            cell: source for (env, cell), source in self._sources.items() if env == int(e)
        }
        with self._source(DEATH_DROP):
            self._original["_drop_corpse_v2"](e, cells)
        self._replay_bulk_corpse(int(e), before_cells, before_sources, cells)
        self._reconcile()

    def _wrap_drop_trail_pellets(self, _sim: Any, trail: np.ndarray) -> None:
        with self._source(BOOST_PELLET):
            self._original["_drop_trail_pellets"](trail)
        self._reconcile()

    def _wrap_move_all(self, _sim: Any, actions: np.ndarray) -> np.ndarray:
        trail = self._original["_move_all"](actions)
        # A valid trail is emitted only when the boost burn actually paid one
        # segment.  It is therefore the simulator's exact cost fact, not an
        # inference from requested boost actions.
        valid = np.asarray(trail)[:, self.hero, 2] == 1
        self._pending_boost_cost += valid.astype(np.int64)
        return trail

    def _wrap_consume_food(self, _sim: Any) -> np.ndarray:
        before = dict(self._sources)
        ate = self._original["_consume_food"]()
        gains = np.zeros((int(self.sim.E), len(SOURCES)), dtype=np.int64)
        for env in range(int(self.sim.E)):
            if not bool(ate[env, self.hero]):
                continue
            cell = self._hero_consumed_cell(env, before)
            if cell is None:
                raise AssertionError(
                    "native hero food telemetry had no pre-consumption provenance cell"
                )
            source = before[(env, cell)]
            gains[env, SOURCES.index(source)] += 1
        self._intake += gains
        self._boost_cost += self._pending_boost_cost
        self._weighted_intake += self._intake
        self._weighted_boost_cost += self._boost_cost
        for env in range(int(self.sim.E)):
            # One ledger row per completed simulator transition, including zero
            # intake/burn rows.  This lets the analyzer independently recompute
            # the H500 weighted cumulative-source integrals.
            self._events.append(
                {
                    "env": env,
                    "frame": int(self.sim.frame[env]),
                    "intake": {source: int(gains[env, idx]) for idx, source in enumerate(SOURCES)},
                    "cumulative_intake": {
                        source: int(self._intake[env, idx]) for idx, source in enumerate(SOURCES)
                    },
                    "weighted_cumulative_intake": {
                        source: int(self._weighted_intake[env, idx])
                        for idx, source in enumerate(SOURCES)
                    },
                    "boost_cost": int(self._pending_boost_cost[env]),
                    "cumulative_boost_cost": int(self._boost_cost[env]),
                    "weighted_cumulative_boost_cost": int(self._weighted_boost_cost[env]),
                }
            )
        self._pending_boost_cost.fill(0)
        self._reconcile()
        return ate

    def _hero_consumed_cell(
        self, env: int, before: Mapping[CellKey, str]
    ) -> Tuple[int, int] | None:
        for step in range(2):
            if not bool(self.sim._trav_valid[env, self.hero, step]):
                continue
            raw = self.sim._trav[env, self.hero, step]
            cell = (int(raw[0]), int(raw[1]))
            if (env, cell) in before:
                return cell
        return None

    @contextmanager
    def _source(self, source: str) -> Iterator[None]:
        old = self._source_context
        self._source_context = source
        try:
            yield
        finally:
            self._source_context = old

    def _seed_sources_from_sim(self) -> None:
        if any(self.sim.corpse_cells):
            raise ValueError(
                "cannot infer death-drop versus boost provenance for pre-existing corpse food; "
                "construct at a fresh world or use clone_for from its observer"
            )
        self._sources.clear()
        for env, cells in enumerate(self.sim.food_cells):
            for raw in cells:
                cell = (int(raw[0]), int(raw[1]))
                self._sources[(env, cell)] = AMBIENT

    def _reconcile(self, new_source: str | None = None, before: set[CellKey] | None = None) -> None:
        native: set[CellKey] = set()
        for env, cells in enumerate(self.sim.food_cells):
            native.update((env, (int(cell[0]), int(cell[1]))) for cell in cells)
        for key in set(self._sources) - native:
            del self._sources[key]
        for key in native - set(self._sources):
            # ``before`` is needed for GridBatchSim's bulk corpse fast path,
            # which performs insertions without calling the wrapped _add_food.
            self._sources[key] = new_source if before is not None and key not in before else AMBIENT

    def _replay_bulk_corpse(
        self,
        env: int,
        before_cells: Sequence[Tuple[int, int]],
        before_sources: Mapping[Tuple[int, int], str],
        incoming: Sequence[Tuple[int, int]],
    ) -> None:
        """Replay GridBatchSim's bulk v2 FIFO/dedup path in observer side state.

        A final-set difference is insufficient: a cap eviction can remove an
        old boost pellet and a later body cell can reinsert that same coordinate
        as a death drop in the *same* bulk call.  Mirror ordered insertion and
        oldest-corpse eviction so the final source tag follows native dynamics.
        """
        ordered = [(int(cell[0]), int(cell[1])) for cell in before_cells]
        tags = {cell: before_sources[cell] for cell in ordered}
        present = set(ordered)
        corpse = {cell for cell in ordered if cell in self.sim.corpse_cells[env]}
        # ``corpse_cells`` is post-call now; reconstruct pre-call membership
        # from side tags, whose v2 invariant was checked before every step.
        corpse = {cell for cell, tag in tags.items() if tag in (DEATH_DROP, BOOST_PELLET)}
        cap = corpse_food_cap(int(self.sim.cfg.max_food))
        for raw in incoming:
            cell = (int(raw[0]), int(raw[1]))
            if cell in present:
                continue
            ordered.append(cell)
            present.add(cell)
            corpse.add(cell)
            tags[cell] = DEATH_DROP
            while len(corpse) > cap:
                victim = next(item for item in ordered if item in corpse)
                ordered.remove(victim)
                present.remove(victim)
                corpse.remove(victim)
                tags.pop(victim)
        actual = [(int(cell[0]), int(cell[1])) for cell in self.sim.food_cells[env]]
        if ordered != actual:
            raise AssertionError("bulk corpse provenance replay diverged from native food order")
        for key in [key for key in self._sources if key[0] == env]:
            del self._sources[key]
        self._sources.update({(env, cell): tag for cell, tag in tags.items()})
