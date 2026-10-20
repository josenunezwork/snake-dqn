# T34 unpaced Colab arm — implementation verification, 2026-10-20

Implemented the [operator ruling](../../docs/research/specs/2026-10-20_t34_unpaced_arm_colab.md)
on `t34-unpaced-colab-20261020`. This is preparation and local verification,
not a study result, timing qualification, Colab execution, or ratification.
The registration remains `ratified: false`.

## Behavior and evidence boundary

- Mac registered execution opens only `bare` and `paced` at 24 fps. Timing is
  still first; study caps remain 40 elapsed hours / 60 CPU-hours. The reused
  T3 pre-check retains its separate eight-hour cap.
- The Colab entrypoint uses the identical registered 144 terminal worlds and
  32 product sessions, deterministic modulo shards, one native numerical
  thread per process, and `min(nproc - 1, 10)` process workers. It writes each
  completed world immediately as immutable JSON + SHA-256 + completion marker;
  the final shard index uses registered order. Hardware includes nproc and RAM.
- Public inputs are the six checkpoint identities from `s1-inputs-20261019`.
  Runtime path remapping uses their registered hashes, independent temporary
  lookup directories, and the existing registry/pins without changing their
  bytes. The production path is `output/unpaced`, beneath the output directory
  already created by the qualified supervisor. Verified Mac imports are under
  `exports/unpaced`.
- Production manifests require a ratified registration, passing source-bound
  trace evidence, matching pinned source bytes, the qualified transport,
  no-allocation supervisor validation, and bounded sparse-source preparation.
  The pinned launcher lives under the broker's recognized campaign prefix, so
  saved manifests receive the qualification recheck immediately before allocation.
  Runtime admission independently checks source and portable bank bindings;
  the supervisor's verified sparse fallback is supported without `.git`.
- The separate Colab cap is 42.533128 units, with 7.088855 units reserved:
  `362.498247 seconds * 176 / 3600 * 2.0 units/hour * 1.20`. Setup/export and
  twice-estimate timeout liability must fit the non-reserve allocation. All
  planned lanes plus existing account assignments must be at most three.
- Missing, failed, malformed, duplicate, or overflowing optional exports never
  alter the Mac verdict. Paced mass significance uses the frozen conservative
  two-hypothesis Holm threshold `p <= .05`; unpaced data cannot relax it.
  Descriptive summaries and per-world bare-prefix divergence are attached only
  after the Mac decision. Prefix hashing is disabled in timing probes and ends
  before the first action changed by assistance. Canonical hashes omit file
  locations and include physical state, RNG, carry, v8 state, and delivered action.

The original bank, seed audit, sample sizes, profile, released checkpoints,
serving pins, and qualified `research/colab_cli` executable bytes are preserved.
The preregistration and generated draft/OC are the explicitly requested
prospective updates. No live Colab, tmux, efficacy, training, strict-gate,
pre-check, or serving operation was performed.

## Final-source Mac entrypoint preflight

Artifact root:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-unpaced-colab-20261020`.

The six public checkpoint files were materialized using the existing public
input verifier with a local cache of the two release assets. Archive/member
sizes and SHA-256 values were checked; no released checkpoint was written.
The preflight input is an unratified copy explicitly labelled `preflight_only`,
with current source identity and two deterministic development seeds disjoint
from every registered terminal, product, and timing seed. It does not amend
the registered bank.

The final entrypoint exercise used source commit
`55b5bfea3d3f612ca4992aca4cc0e0f7843d6cc2`, whose executable/data closure has
206 files. Exact invocation (from the assigned worktree):

```bash
E=/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-unpaced-colab-20261020
env -u COLAB_CLI_ROOT \
  T34_UNPACED_PREFLIGHT=1 \
  T34_UNPACED_PYTHON=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python \
  T34_UNPACED_INPUT_ROOT="$E/public-inputs-01" \
  T34_UNPACED_OUTPUT_ROOT="$E/preflight-03" \
  T34_UNPACED_REGISTRATION="$E/preflight-registration-03.json" \
  T34_UNPACED_CATALOG=/Users/josenunez/Projects/ml/snake-dqn-unpaced-colab/research/t34_exit_b8/PUBLIC_INPUTS.json \
  T34_UNPACED_DEVELOPMENT_ROWS="$E/preflight-development-01.json" \
  T34_UNPACED_SOURCE_COMMIT=55b5bfea3d3f612ca4992aca4cc0e0f7843d6cc2 \
  T34_UNPACED_FRAMES=4 \
  /usr/bin/caffeinate -dims bash research/t34_exit_b8/unpaced_retention.sh
```

`preflight-03/preflight.json`: COMPLETE, two worlds, four frames each,
`metrics_persisted: false`, 7.181195 seconds total; terminal 2.505956 seconds,
Watch product 4.642571 seconds. Hardware: arm64 Mac, 18 available CPUs,
68,719,476,736 RAM bytes, `nproc: 1`, one process, one native pool thread,
at most two CPU threads including the assistance worker. These short fixture
costs are not H5000 forecasts or timing-gate measurements.

Earlier `preflight-01` and `preflight-02` completed in 8.376188 and 7.124153
seconds at sources `ef5fd56c` and `eb0ac880`. Both remain preserved; the final
repeat followed the shared-admission launcher fix. None stored gameplay metrics
or consumed registered seeds.

## Tests, source delivery, and registration refresh

All Python verification uses the primary project venv with CPU device and
`OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=NUMBA_NUM_THREADS=VECLIB_MAXIMUM_THREADS=NUMEXPR_NUM_THREADS=1`.
No xdist fan-out is used. The one spawn-pool regression runs two deterministic
single-thread fixture workers; it proves serial/pool world-export bytes and
merge order, not parallel gameplay or cross-platform action equality.

Focused completed selections: 87 runner/registration/analyzer/adapter tests;
then 57 analyzer/new-Colab tests after optional-input hardening and 27 tests
covering the final broker-admission fence, collector, and source binding. The final
combined selection and refreshed draft evidence are recorded in the closeout
below. Black (100 columns, one worker), isort (Black profile), flake8 (one job),
shell syntax, and `git diff --check` are the style checks.

Earlier development selections exposed stale three-arm test fixtures and a
manifest test using a now-refused lane count. Their fixtures were updated
without relaxing assertions or scientific bars; the subsequent selections
passed. No scientific row or Colab allocation was retried.

Independent read-only review through `55b5bfea` reported no remaining
actionable product finding. Its scope included the Mac decision boundary,
public/runtime mapping, source provenance, output-directory compatibility,
streaming/ordering, and optional-import error handling.

An independent verifier passed the no-allocation supervisor validator, including
the shared qualification admission, and the real sparse-source prepare/install
path at `55b5bfea` using a `DRAFT_TEST_ONLY_NO_ALLOCATION` fixture. All 206
registered source files were hash-bound; the full manifest inventory contained
212 files including the qualified transport. Isolated `python -I` imports of
both actual new modules resolved inside the installed fallback bundle.
No gameplay or allocation occurred. Receipt:
`verification-handoff/offline-source-wrapper-preflight.json`, SHA-256
`40befa5524b30b5bf6370894b9075797f1f0d353da53662065c596c38d2825da`.
The earlier source-delivery/import receipts at `eb0ac880` remain preserved.

The development-only OC refresh uses the same 20,000-replicate seed and unchanged
fixed-N design, into fresh paths:

```bash
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
E=/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-unpaced-colab-20261020
"$PY" -m research.t34_pathb.retention_oc_v2 \
  --out "$E/oc-02.json" --state "$E/oc-state-02.json" \
  --reps 20000 --seed 3420261020 --max-seconds 60
# Commit the completed OC before building the updated unauthorized draft.
"$PY" -m research.t34_pathb.registration_retention_v2 build \
  --out "$E/draft-02.json" \
  --checkpoints research/t34_pathb/RETENTION_CHECKPOINTS_V2.json
"$PY" -m research.t34_pathb.registration_retention_v2 validate
```

OC refresh: PASS, 20,000 completed, zero failed cells; maximum protected
regression pass rate .0262 against the fixed .06 limit. The refreshed OC was
committed before draft generation. Draft source commit:
`99b4246745b0e304e88c594e11e5a81380eba259`. Its 206 executable/data bindings
are identical to final preflight source `55b5bfea`; the later commits contain
the separately bound OC and upstream advice outside this closure. Draft validation returned
`{"valid":true,"authorized":false,"mode":"draft"}`.

| Artifact | SHA-256 |
|---|---|
| Updated draft / `draft-02.json` | `d6b317e5445e4c4faf8e1927d376f9cb5fb363f2223811678f8eeb79f717ed10` |
| Updated OC / `oc-02.json` | `4ad7065b45af11151f7288245e232e99cb6f3e4a3f310f938976adfdaa466717` |
| Preserved bank | `af54fd499e99575bcc960c78fc3c52d756f8e1c1a52ca65edb95bb5b013f6d95` |
| Preserved seed audit | `7d895340b742d2011fde9e2731c3e73c1b58484023b3e4afbf66e769704f3b51` |

Combined retention, adjacent v2 adapters/controllers/engines, and Colab CLI
tests use this exact command, with the single-thread environment stated above.
The two large unchanged-transport fault matrices are excluded; the remaining
specific transport fault/latency tests are retained:

```bash
"$PY" -m pytest -q -m 'not slow' \
  -k 'not test_acceptance_fault_classes_never_orphan_or_repeat_mutations and not test_seeded_random_fault_property_preserves_broker_safety' \
  tests/test_colab_cli_*.py tests/test_t34_pathb_*v2.py \
  tests/test_t34_retention_*.py tests/test_t34_unpaced_*.py
```

The Colab tests use offline fake-provider fixtures; their allocations/releases
are simulated. They perform no live Colab operations.
An earlier broader run (`tests-final-01.log`) was superseded before completion
to land the shared-admission launcher fix; it is not a completed verification gate.

The combined selection (`tests-final-02.log`) completed in 794.83 seconds:
414 passed, one failed, 56 deselected. All 227 selected offline CLI cases passed.
The sole failure was an older CLI test double that did not accept the new
`unpaced_shards` keyword. Only that test fixture was corrected; it now also
asserts the default empty descriptive-input list. The immediate affected-file
selection passed all 41 cases (`tests-final-03.log`, 2.17 seconds).
No executable or source-bound data changed for this correction.
The complete T34 selection was then repeated with the corrected fixture:

```bash
"$PY" -m pytest -q -m 'not slow' \
  tests/test_t34_pathb_*v2.py tests/test_t34_retention_*.py \
  tests/test_t34_unpaced_*.py
```

All 188 cases passed (`tests-final-04.log`, 20.30 seconds). Together with the
227 passing offline CLI cases, this covers all 415 selected regression cases.
Post-test validation confirmed the 206 current source bindings match the draft,
OC, and final preflight input; the bank, seed audit, profile, serving pins, and
entire qualified transport remain byte-identical to baseline `77175b81`.

Final style commands (all passed; 16 Python files unchanged):

```bash
verification_python_files=(research/t34_pathb/{adapter_v2,analyze_v2,registration_retention_v2,retention_v2,source_binding_v2,unpaced_colab,make_unpaced_manifest}.py
  tests/test_t34_pathb_{adapter_v2,analysis_v2,retention_v2}.py
  tests/test_t34_retention_{fixes2_runner,fixes_analyzer,fixes_runner}.py
  tests/test_t34_unpaced_{analysis,colab,registration}.py)
"$PY" -m black --workers 1 --check --line-length 100 "${verification_python_files[@]}"
"$PY" -m isort --profile black --check-only "${verification_python_files[@]}"
"$PY" -m flake8 --jobs 1 "${verification_python_files[@]}"
bash -n research/t34_pathb/unpaced_colab.sh
bash -n research/t34_exit_b8/unpaced_retention.sh
git diff --check
```

## Operator handoff

The exact preflight, manifest-build, validation, and dispatch commands are in
[README_V2.md](README_V2.md#descriptive-unpaced-colab-arm--operator-only).
Production dispatch remains refused until the operator supplies the actual
final-source trace, ratifies/publishes the registration, and reviews current
account assignments/rates. A generated draft or successful local preflight
does not authorize spending. No live-runtime performance, session release, or
cross-platform divergence claim is made by this implementation.
