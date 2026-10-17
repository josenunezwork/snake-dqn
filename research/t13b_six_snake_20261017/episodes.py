"""T13b real Watch episodes at the six-snake default and research-only eight.

The immutable T13 source remains the behavioral implementation.  This module
temporarily supplies only the constructor count and a count-aware invariant;
it restores both symbols after every episode.
"""

from __future__ import annotations

import contextlib
from dataclasses import replace
from typing import Any, Iterator
from unittest.mock import patch

from research.t13_product_roster_20261017 import episodes as t13

ROSTERS = t13.ROSTERS
DEFAULT_SNAKES = 6
SUPPLEMENTAL_SNAKES = 8


def _new_session(checkpoint: str | None, seed: int, *, veto: bool | None, snakes: int) -> Any:
    """Use the real constructor; config override exists only for supplemental eight."""
    if snakes not in (DEFAULT_SNAKES, SUPPLEMENTAL_SNAKES):
        raise ValueError(f"T13b supports only {DEFAULT_SNAKES} or {SUPPLEMENTAL_SNAKES} snakes")
    from research.apex_veto_serving_20261001.serving_run import set_world_seed
    from web.backend.session import GameSession

    original_loader = __import__("web.backend.session", fromlist=["load_config"]).load_config

    def eight_snake_config(path: str) -> Any:
        config = original_loader(path)
        return replace(config, game=replace(config.game, num_snakes=snakes))

    set_world_seed(seed)
    with t13._selection(checkpoint, veto=veto), contextlib.ExitStack() as stack:
        stack.enter_context(patch("web.backend.session.served_resolution"))
        stack.enter_context(patch("web.backend.session.watch_tick"))
        if snakes == SUPPLEMENTAL_SNAKES:
            stack.enter_context(
                patch("web.backend.session.load_config", side_effect=eight_snake_config)
            )
        session = GameSession()
    session._release_monitoring = False
    return session


def _assert_roster(session: Any, roster: str, donor: Any | None, snakes: int) -> None:
    """T13's registry and wrapper checks, parameterized only by roster count."""
    expected = {
        "all_b8": "b8-s3",
        "all_frp3": "frp3-s12",
        "all_champion": "champion",
        "mixed": "b8-s3",
    }[roster]
    choice = session.served_checkpoint or {}
    if choice.get("name") != expected or choice.get("reason") or choice.get("pin_problems"):
        raise RuntimeError(f"T13b registry selection is not exact {expected}: {choice!r}")
    members = list(session.game.snakes)
    if len(members) != snakes or int(members[0].id) != int(session.hero_id):
        raise RuntimeError(f"T13b requires slot-zero hero and {snakes} snakes")
    if roster == "all_b8":
        if any(
            s.policy is not session.policy or getattr(s, "safety_veto", None) is not None
            for s in members
        ):
            raise RuntimeError("all_b8 roster is not entirely unwrapped served B8")
        return
    if roster in ("all_frp3", "all_champion"):
        wrapped = [int(s.id) for s in members if getattr(s, "safety_veto", None) is not None]
        veto = getattr(members[0], "safety_veto", None)
        if (
            any(s.policy is not session.policy for s in members)
            or wrapped != [int(session.hero_id)]
            or getattr(veto, "method", None) != "free-space-veto/v8-space-and-head(lambda=8.0)"
        ):
            raise RuntimeError("homogeneous v8 scope differs from serving semantics")
        return
    if donor is None or (donor.served_checkpoint or {}).get("name") != "frp3-s12":
        raise RuntimeError("mixed roster has no exact FRP3 donor")
    if (
        members[0].policy is not session.policy
        or getattr(members[0], "safety_veto", None) is not None
    ):
        raise RuntimeError("mixed roster B8 hero changed")
    for snake in members[1:]:
        veto = getattr(snake, "safety_veto", None)
        if (
            snake.policy is not donor.policy
            or getattr(veto, "method", None) != "free-space-veto/v8-space-and-head(lambda=8.0)"
            or getattr(veto, "head_avoidance", None) is not True
            or getattr(veto, "reference_lambda", None) is not None
        ):
            raise RuntimeError("mixed opponent is not exact FRP3 plus default strict v8")


@contextlib.contextmanager
def _t13_count_override(snakes: int) -> Iterator[None]:
    """Scope compatibility substitutions to one in-process episode invocation."""
    original_new, original_assert = t13._new_session, t13._assert_roster

    def constructed(checkpoint: str | None, seed: int, *, veto: bool | None) -> Any:
        return _new_session(checkpoint, seed, veto=veto, snakes=snakes)

    def asserted(session: Any, roster: str, donor: Any | None) -> None:
        _assert_roster(session, roster, donor, snakes)

    t13._new_session, t13._assert_roster = constructed, asserted
    try:
        yield
    finally:
        t13._new_session, t13._assert_roster = original_new, original_assert


def run_watch(
    roster: str, seed: int, horizon: int = 5000, *, snakes: int = DEFAULT_SNAKES
) -> dict[str, Any]:
    """Run one T13b episode and bind its requested/effective roster size in raw evidence."""
    if snakes == SUPPLEMENTAL_SNAKES and roster != "all_b8":
        raise ValueError("the supplemental eight-snake T13b row is all_b8 only")
    if snakes not in (DEFAULT_SNAKES, SUPPLEMENTAL_SNAKES):
        raise ValueError("T13b roster size must be 6 or 8")
    with _t13_count_override(snakes):
        record = t13.run_watch(roster, seed, horizon=horizon)
    record["schema_version"] = "t13b-six-snake-episode/v1"
    record["requested_snake_count"] = snakes
    if record["initial_world"]["effective_num_snakes"] != snakes:
        raise RuntimeError("T13b effective roster count differs from requested count")
    return record
