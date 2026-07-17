"""Shared game-mechanics constants and cell-canonicalization helpers.

Single source of truth for the mechanics-v2 magic numbers (blueprint §1) and for
the integer cell semantics used by collision and food-pickup checks. The future
vectorized simulator MUST import these from here too, so the two simulators can
never drift on load-bearing mechanics values.

Versioning: constants suffixed ``_V2`` take effect only when
``GameConfig.MECHANICS_VERSION == 2``; version 1 behavior stays unchanged.
The one exception is cell canonicalization (``cell_index``/``same_cell``),
which applies at ALL mechanics versions: for the shipped geometry (10px
segments moving in 10px cardinal steps) cell-occupancy equality is exactly
equivalent to the legacy ``distance < segment_size`` tests for any pair of
positions whose relative offset lies on the segment lattice.
"""

from typing import Tuple

# --- Head-on resolution (v2) -------------------------------------------------
# A snake whose logical length is >= HEADON_SIZE_RATIO x the other's survives a
# head-on collision and receives kill credit; near-equal stays mutual death.
HEADON_SIZE_RATIO: float = 1.15

# --- Corpse / trail food economics -------------------------------------------
# Fraction of a dead snake's segments converted to food. v1 keeps the legacy
# every-other-segment drop; v2 drops the full corpse (kills pay in mass).
CORPSE_DROP_FRACTION_V1: float = 0.5
CORPSE_DROP_FRACTION_V2: float = 1.0

# v2: corpse-class food (kill corpses + boost trail pellets) is exempt from the
# ambient maintain_count/max_food cap, so a kill's spoils no longer suppress
# baseline food spawns.
CORPSE_EXEMPT_FROM_CAP_V2: bool = True

# v2: corpse-class food is exempt from the AMBIENT cap but is NOT unbounded.
# Total corpse pellets on the board are capped at ``corpse_food_cap(max_food)``;
# when a new corpse add would exceed the cap, the OLDEST corpse pellet (first in
# board append order — "oldest rots first") is evicted. Unbounded corpse
# accumulation otherwise (a) floods the board with free food, removing any
# incentive to hunt (kills collapse to ~0), and (b) makes both the sim's food
# ops and the ego-raster featurizer scale with a monotonically growing pellet
# count F, collapsing training throughput as the board fills. The cap keeps F
# bounded (total food <= (1 + multiple) x max_food) so those costs stay flat.
# BOTH simulators MUST import this so the eviction target — and the resulting
# ordered food list — stay byte-identical (the parity gate depends on it).
#
# Sizing: featurizer/sim cost scales with the TOTAL live pellet count (ambient +
# corpse) summed across the batch, so the per-env cap must hold the corpse MEAN
# low, not just clip the max. At 0.5 the training board (max_food=300) settles at
# total food ~450 and a flat ~8.5k agent-steps/s, versus an unbounded slide from
# ~12k to <5k. Lower it for more speed / scarcer food (which also pressures the
# kills-0 pathology); raise it for a richer corpse economy.
CORPSE_FOOD_CAP_MULTIPLE_V2: float = 0.5
# Floor so low-``max_food`` configs still admit a normal-size corpse drop instead
# of instantly trimming it to near-nothing: sparse-food arenas and the small
# hand-checkable unit-test configs. A no-op for the training board (max_food=300
# -> 150) and for every parity config (all already >= this floor), so it changes
# only degenerate small-``max_food`` cases — never the shipped mechanics.
CORPSE_FOOD_CAP_FLOOR_V2: int = 10


def corpse_food_cap(max_food: int) -> int:
    """Max simultaneous corpse-class pellets under v2 (see the constants above)."""
    return max(int(max_food * CORPSE_FOOD_CAP_MULTIPLE_V2), CORPSE_FOOD_CAP_FLOOR_V2)


def evict_oldest_corpse(food, corpse_positions) -> object:
    """Evict the oldest corpse pellet from an ordered food list in place.

    Removes the first cell in ``food`` (append/board order) that is a member of
    ``corpse_positions`` from BOTH structures and returns it, or ``None`` if no
    corpse pellet is present. Shared by the live :class:`FoodManager` and the
    vectorized ``BatchSim`` so the eviction target is identical on both — a
    prerequisite for bit-exact food-list parity under the corpse cap. Callers
    holding extra membership indexes (e.g. the sim's ``food_set``) must discard
    the returned cell from those too.

    Args:
        food: Ordered list of ``(col, row)`` (or pixel) food cells; mutated.
        corpse_positions: Set of corpse-class cells; mutated.

    Returns:
        The evicted cell, or ``None`` when there is no corpse pellet to evict.
    """
    for cell in food:
        if cell in corpse_positions:
            food.remove(cell)
            corpse_positions.discard(cell)
            return cell
    return None


# v2: each boost segment burn drops a corpse-class pellet at the vacated tail
# cell (boost becomes a mass transfer instead of mass destruction).
BOOST_DROPS_TRAIL_V2: bool = True

# --- Training population floor (v2, train-mode only) --------------------------
# Training episodes end when fewer than this many snakes remain alive, fixing
# lone-survivor replay skew. Product/watch/play modes are unaffected.
POPULATION_FLOOR_V2: int = 3


def snap_to_cell(position: Tuple[int, int], cell_size: int) -> Tuple[int, int]:
    """Snap a pixel position to the nearest lower cell corner on the lattice.

    Spawn positions (food, snakes) are drawn from continuous ``random`` ranges,
    which puts entities on mismatched sub-cell offsets. On those offsets the
    canonical ``same_cell`` predicate is NOT equivalent to the legacy
    ``distance < cell_size`` radius test, so collisions/pickups diverge and the
    vectorized-sim parity premise fails. Snapping every spawn to ``(x // s) * s``
    places all entities on one shared lattice, where two distinct lattice points
    are at distance ``>= cell_size`` and coincide iff they share a cell — making
    ``same_cell`` exactly equivalent to the radius test.

    Args:
        position: (x, y) position in pixels.
        cell_size: Cell edge length in pixels (the segment size).

    Returns:
        (x, y) snapped to the lattice.
    """
    return (
        (position[0] // cell_size) * cell_size,
        (position[1] // cell_size) * cell_size,
    )


def cell_index(position: Tuple[int, int], cell_size: int) -> Tuple[int, int]:
    """Return the integer grid cell occupied by a position.

    Uses floor division, so negative (out-of-bounds) coordinates map to
    negative cells rather than aliasing onto cell 0.

    Args:
        position: (x, y) position in pixels.
        cell_size: Cell edge length in pixels (the segment size).

    Returns:
        (col, row) integer cell coordinates.
    """
    return (position[0] // cell_size, position[1] // cell_size)


def same_cell(
    position_a: Tuple[int, int],
    position_b: Tuple[int, int],
    cell_size: int,
) -> bool:
    """Return whether two positions occupy the same integer grid cell.

    This is the canonical entity-collision / food-pickup predicate (all
    mechanics versions). For positions whose relative offset lies on the
    segment lattice it is exactly equivalent to the legacy radius test
    ``distance(a, b) < cell_size``.

    Args:
        position_a: First (x, y) position in pixels.
        position_b: Second (x, y) position in pixels.
        cell_size: Cell edge length in pixels (the segment size).

    Returns:
        True if both positions fall in the same cell.
    """
    return cell_index(position_a, cell_size) == cell_index(position_b, cell_size)
