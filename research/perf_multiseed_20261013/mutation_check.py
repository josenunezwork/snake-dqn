#!/usr/bin/env python3
"""Mutation check for ``stacked_learner.py``: every deliberate break must FAIL its test(s).

Each mutant replaces one exact substring (it must occur once) of ``stacked_learner.py``, runs
the selected ``tests/test_perf_multiseed.py`` tests and expects an assertion failure; the
source is restored after every mutant (and on exit). Mutates the working tree: run it on a
clean checkout, not while editing the file.

    ./venv/bin/python research/perf_multiseed_20261013/mutation_check.py

Result 2026-10-13 (CPU, torch 2.9.1): 12/12 mutants caught, all by assertion failures.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "research/perf_multiseed_20261013/stacked_learner.py"
TESTS = "tests/test_perf_multiseed.py"
LN = 'x * p[name + ".weight"].unsqueeze(1) + p[name + ".bias"].unsqueeze(1)'

#: name, original substring, mutant, ``pytest -k`` selection that must fail
MUTANTS = [
    (
        "single_update_lam",
        "loss = td + lam * anc\n    finite = bool",
        "loss = td + lam * 1.01 * anc\n    finite = bool",
        "single_update_is_the_learner",
    ),
    ("nonfinite_steps", "ok = fin & ~self.halted", "ok = np.ones(self.K, dtype=bool)", "isolated"),
    (
        "per_seed_clip_noop",
        "    for g in grads:\n        g.mul_(",
        "    for g in grads[:0]:\n        g.mul_(",
        "clip_per_seed or tracks_k_single",
    ),
    ("argmax_fallback", "< 3\n    m = m |", "< 2\n    m = m |", "loss_pieces"),
    ("clip_eps", "(totals + 1e-6)", "(totals + 1e-5)", "clip_per_seed"),
    (
        "adam_shared_steps",
        "for _ in self.params] for _ in range(self.K)]",
        "for _ in self.params]] * self.K",
        "stacked_adam",
    ),
    (
        "grouped_flatten_layout",
        "x = x.reshape(n, K, -1).transpose(0, 1)",
        "x = x.reshape(K, n, -1)",
        "grouped_forward_is",
    ),
    (
        "grouped_ln_affine_seed0",
        LN,
        'x * p[name + ".weight"][:1] + p[name + ".bias"][:1]',
        "grouped_forward_is or tracks_k_single",
    ),
    (
        "grouped_dueling_axis",
        "a - a.mean(dim=2, keepdim=True)",
        "a - a.mean(dim=1, keepdim=True)",
        "grouped_forward_is",
    ),
    (
        "grouped_input_layout",
        "lo = local.transpose(0, 1).reshape(",
        "lo = local.reshape(",
        "grouped_forward_is",
    ),
    (
        "target_sync_seed0",
        "self.tparams[name][k].copy_(p[k])",
        "self.tparams[name][k].copy_(p[0])",
        "isolated or tracks_k_single",
    ),
    (
        "td_mean_axis",
        'reduction="none").mean(dim=1)',
        'reduction="none").mean(dim=1).mean().expand(K)',
        "tracks_k_single",
    ),
]


def main() -> int:
    orig = SRC.read_text()
    env = dict(os.environ, OMP_NUM_THREADS="2")
    missed = 0
    try:
        for name, old, new, sel in MUTANTS:
            if orig.count(old) != 1:
                print(f"{name}: pattern found {orig.count(old)}x (need 1): FIX THE MUTANT")
                missed += 1
                continue
            SRC.write_text(orig.replace(old, new))
            cmd = [sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider"]
            r = subprocess.run(
                cmd + [TESTS, "-k", sel], cwd=REPO, env=env, capture_output=True, text=True
            )
            asserted = "AssertionError" in r.stdout or "\nE       assert" in r.stdout
            ok = r.returncode != 0 and asserted
            missed += not ok
            verdict = "caught (assertion)" if ok else f"NOT CAUGHT (rc={r.returncode})"
            print(f"{name}: {verdict}", flush=True)
    finally:
        SRC.write_text(orig)
    print(f"{len(MUTANTS) - missed}/{len(MUTANTS)} mutants caught; source restored")
    return 1 if missed else 0


if __name__ == "__main__":
    raise SystemExit(main())
