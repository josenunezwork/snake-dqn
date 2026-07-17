"""Shared command-line and determinism helpers for the eval harnesses.

Home of the plumbing that ``tournament_eval.py``, ``evaluate_checkpoints.py``
and ``ensemble_eval.py`` all need, so the three stay in lockstep:

* ``parse_seed_list`` — the ``--seeds`` argparse type, accepting BOTH the comma
  form (``0,1,2``) and the inclusive range form (``0-15``), plus any mix of the
  two (``0-3,7``).
* ``set_seed``       — seed every RNG before one deterministic rollout.

This lives beside ``eval_stats.py`` rather than inside it: ``eval_stats`` is the
home of the gate MATH and imports nothing but ``math``/``typing``, whereas
``set_seed`` needs torch and ``parse_seed_list`` needs argparse.
"""

from __future__ import annotations

import argparse
import random
import re
from typing import List

import numpy as np

# A seed token is either a bare non-negative int ("3") or an inclusive range
# ("0-15"). Negative seeds are rejected: numpy's global seeder only accepts
# [0, 2**32-1], and the "-" would be ambiguous with the range separator.
_SEED_TOKEN = re.compile(r"^(?P<start>\d+)(?:-(?P<end>\d+))?$")


def parse_seed_list(value: str) -> List[int]:
    """Parse a ``--seeds`` value into an explicit seed list.

    Accepts comma-separated seeds, inclusive ranges, or a mix of both. Order is
    preserved and duplicates are kept, so a caller can deliberately repeat a
    seed.

    Args:
        value: Seed spec, e.g. ``"0,1,2"``, ``"0-15"`` or ``"0-3,7"``.

    Returns:
        The expanded list of seeds.

    Raises:
        argparse.ArgumentTypeError: If a token is not a non-negative int or an
            inclusive range, if a range ends before it starts, or if no seed is
            given at all.
    """
    seeds: List[int] = []
    for raw_token in value.split(","):
        token = raw_token.strip()
        if not token:
            continue
        match = _SEED_TOKEN.match(token)
        if match is None:
            raise argparse.ArgumentTypeError(
                f"invalid seed {token!r}: expected a non-negative integer (e.g. '3') "
                "or an inclusive range (e.g. '0-15')"
            )
        start = int(match.group("start"))
        end_text = match.group("end")
        if end_text is None:
            seeds.append(start)
            continue
        end = int(end_text)
        if end < start:
            raise argparse.ArgumentTypeError(
                f"invalid seed range {token!r}: end {end} is before start {start}"
            )
        seeds.extend(range(start, end + 1))
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def set_seed(seed: int) -> None:
    """Seed all RNGs so paired rollouts share world construction.

    Args:
        seed: The seed applied to ``random``, ``numpy`` and ``torch``.
    """
    # Imported lazily so argparse-only consumers do not pay for torch.
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
