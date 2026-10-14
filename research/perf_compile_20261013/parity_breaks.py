#!/usr/bin/env python3
"""Deliberate-break check for the perf-compile parity tests (tests/test_perf_compile.py).

Each break patches one source file IN THIS CHECKOUT, runs the tests that back the matching
claim, records which FAIL, then restores the file byte-for-byte (try/finally). A parity claim
is only backed if its test fails under the matching break. Run from a clean checkout (it
edits tracked files while it runs; do not run two copies at once)::

    ./venv/bin/python research/perf_compile_20261013/parity_breaks.py [B1 B4 ...]

Expected (2026-10-13, results/parity_breaks_20261013.json): every break fails at least one
of its tests; the PASSED entries are the ones the break cannot reach (B2: scope ``forward``
does not compile ``loss_terms``; B4: the 100-frame actor run never has fewer live heroes
than ``--worlds``; B6: the ``raise`` path never reaches the check; B10: the test-size
learners never hit the bf16 false positive the criterion test pins; B11: ``mismatch`` is
the fp32 check).
"""

import json
import re
import subprocess
import sys
from pathlib import Path

WT = Path(__file__).resolve().parents[2]
PY = sys.executable
LM = "research/redesign_m3c_20261008/learner_m3c.py"
CM = "research/redesign_m3c_20261008/compile_m3c.py"
NET = "src/model/ego2s_network.py"
T = "tests/test_perf_compile.py"

BREAKS = [
    (
        "B1 eager T1 loss changed (default path)",
        LM,
        "\n    td = torch.nn.functional.smooth_l1_loss(qa, y)\n",
        "\n    td = torch.nn.functional.smooth_l1_loss(qa, y, beta=0.999)\n",
        [T + "::test_compile_off_never_calls_torch_compile"],
    ),
    (
        "B2 compiled-only: target y x 1.001 inside torch.compile(loss_terms)",
        LM,
        "        y = R + disc * q_next\n    td = ",
        "        y = R + disc * q_next\n        if torch.compiler.is_compiling():\n"
        "            y = y * 1.001\n    td = ",
        [
            T + "::test_compiled_loss_terms_close_on_fixed_batch",
            T + "::test_compiled_updates_stay_close_to_eager",
            T + "::test_first_steps_verified_then_compiled",
        ],
    ),
    (
        "B3 compiled-only: Ego2sNet Q x 1.001 under compile",
        NET,
        "        return self.value(h) + a - a.mean(dim=1, keepdim=True)\n",
        "        q = self.value(h) + a - a.mean(dim=1, keepdim=True)\n"
        "        return q * 1.001 if torch.compiler.is_compiling() else q\n",
        [
            T + "::test_compiled_loss_terms_close_on_fixed_batch",
            T + "::test_compiled_updates_stay_close_to_eager",
            T + "::test_hero_forward_matches_eager_without_recompiles",
        ],
    ),
    (
        "B4 hero forward returns the wrong padded rows",
        CM,
        '            q = self.fn(dv["local"], dv["global"], dv["scalars"])[:n]\n',
        '            q = self.fn(dv["local"], dv["global"], dv["scalars"])[self.rows - n :]\n',
        [
            T + "::test_hero_forward_matches_eager_without_recompiles",
            T + "::test_actor_compiled_hero_forward_end_to_end",
        ],
    ),
    (
        "B5 hero forward compiles a frozen copy (weight updates not picked up)",
        CM,
        "        self.fn = compile_fn(torch, net, mode)\n",
        "        import copy\n\n        self.fn = compile_fn(torch, copy.deepcopy(net), mode)\n",
        [T + "::test_hero_forward_picks_up_in_place_weight_updates"],
    ),
    (
        "B6 verify step keeps the compiled result (no eager recompute)",
        LM,
        "        else:\n            self._verified[key] = self._verified.get(key, 0) + 1\n"
        "        return td, anc, qmax, finite\n",
        "        else:\n            self._verified[key] = self._verified.get(key, 0) + 1\n"
        "        return (*compiled_out, finite)\n",
        [T + "::test_learner_falls_back_to_eager_bitwise"],
    ),
    (
        "B7 every batch treated as canonical (stale-row batches compiled)",
        LM,
        "        return n == self.a_rows and m in (0, self.d_rows)\n",
        "        return True\n",
        [T + "::test_noncanonical_batches_run_eager"],
    ),
    (
        "B8 no recompilation-storm guard in the learner",
        LM,
        "        if self._gcount.storm:  # this step is fine; the following ones run eagerly\n",
        "        if False:\n",
        [T + "::test_recompile_storm_falls_back_to_eager"],
    ),
    (
        "B9 graph count = process-wide counter (not per learner)",
        CM,
        "        self.graphs += g - self._at[0]\n",
        "        self.graphs = g\n",
        [T + "::test_graph_count_is_per_learner"],
    ),
    (
        "B10 AMP check = fixed 5e-2 (no rounding floor): false positives",
        LM,
        "AMP_FLOOR_FACTOR = 4.0\n",
        "AMP_FLOOR_FACTOR = 0.0\n",
        [
            T + "::test_amp_compiled_steps_pass_the_rounding_floor_check",
            T + "::test_grad_check_criterion",
        ],
    ),
    (
        "B11 AMP check effectively off (floor factor 1e9)",
        LM,
        "AMP_FLOOR_FACTOR = 4.0\n",
        "AMP_FLOOR_FACTOR = 1e9\n",
        [T + "::test_amp_check_still_catches_wrong_steps", T + "::test_grad_check_criterion"],
    ),
]


def run(tests):
    r = subprocess.run(
        [PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rA", *tests],
        cwd=WT,
        capture_output=True,
        text=True,
    )
    res = {}
    for line in r.stdout.splitlines():
        m = re.match(r"^(PASSED|FAILED|ERROR) (\S+)", line)
        if m:
            res[m.group(2).split("::", 1)[1]] = m.group(1)
    return res, r.stdout[-4000:]


out = []
only = {a for a in sys.argv[1:] if a.startswith("B")}
for name, rel, old, new, tests in BREAKS:
    if only and name.split()[0] not in only:
        continue
    path = WT / rel
    orig = path.read_bytes()
    text = orig.decode()
    assert text.count(old) == 1, (name, text.count(old))
    try:
        path.write_text(text.replace(old, new))
        res, tail = run(tests)
    finally:
        path.write_bytes(orig)
    row = {"break": name, "file": rel, "results": res}
    row["all_failed"] = bool(res) and all(v != "PASSED" for v in res.values())
    print(json.dumps(row), flush=True)
    if not res:
        print(tail)
    out.append(row)
if "--out" in sys.argv:
    Path(sys.argv[sys.argv.index("--out") + 1]).write_text(json.dumps(out, indent=1) + "\n")
sys.exit(
    0
    if all(r["results"] and not all(v == "PASSED" for v in r["results"].values()) for r in out)
    else 1
)
