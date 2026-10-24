# I2 FIX-FIRST round 3 — authoritative ancestry and committed bytes

This sibling handoff addresses the P1 in
`snake-dqn-artifacts/t34-sol/astra-review-i2-03/snake-dqn.last-message.md`,
reviewed at `844ba53b4894aac86401e658b1fc9916823723c3`. Work began in the clean
`snake-dqn-f2-i2` worktree, branch `sol-f2-i2-fix3-20261024`, at
`a2d03fd9594c6e2e8369f01d750d2220c73f47ed` (the reviewed main plus an unrelated
T4A commit). This is identity/admission implementation and synthetic verification;
no real F2 sealed run or R-A look was executed.

## Review-to-fix mapping

- **False ancestry from replacement refs or grafts:** `github_authority._contains`
  now always uses the read-only GitHub compare endpoint for the exact commit
  against the authoritative main tip and relevant push-event `before`/`after`
  SHAs. It checks the returned base, merge base and comparison status. Local
  ancestry never authorizes publication, even if all objects exist locally.
  The all-zero branch-creation `before` is absent by definition. A per-scan
  cache avoids duplicate immutable comparisons; no cache survives revalidation.
- **False ledger-to-commit binding from a replacement tree:** `_server_export`
  fetches the result ledger and BOTH sidecars through the read-only Contents API
  with `ref=<exact full commit SHA>`. It independently does the same for the
  input ledger and both sidecars at the result metadata's `input_published_ref`,
  checking that input commit's server ancestry against the result commit.
  Each response must be a correctly named base64 file with bounded exact size,
  valid framed Git blob SHA1, and SHA256 equality with the local file bytes.
  Missing files, malformed/truncated responses, API failures, unknown ancestry
  or any byte mismatch cannot establish publication evidence.
- `_identity` parses the server-verified result payload snapshot, derives its
  ledger SHA256 from that snapshot, and checks its receipt SHA256 against the
  supplied binding and input SHA256 against the server-verified input ledger.
  Neither capture nor revalidation deserializes the sealed statistic.
- **Local interpretation:** all identity Git calls flow through `f2._git`.
  It removes every inherited `GIT_*` variable, then sets
  `GIT_NO_REPLACE_OBJECTS=1`, `GIT_GRAFT_FILE=/dev/null` and
  `GIT_TERMINAL_PROMPT=0`; it also passes `--no-replace-objects` and
  `-c core.useReplaceRefs=false`. Git credential lookup uses the same sanitized
  environment. No repository configuration, replacement ref or graft is edited.
- **Unavailable means strikes:** `publication_evidence` continues to return
  `None` on unverifiable server ancestry or bytes. `ra_guard.receipt_binding`
  also returns `None` for unavailable local publication metadata, so raw Git's
  rejection of a forged ledger cannot bypass the strikes preparation route by
  failing before publication revalidation. A previously prepared packet whose
  binding/evidence disappears still refuses; this does not permit a second look.
  Trusted-time and once-only-claim rules remain in force for both branches.

## Frozen statistic and source identity

The **only** `f2.py` change is its local Git helper and environment helper.
Every byte from `def _published(` through EOF matches the reviewed `844ba53b`
source. In particular, `selected_survival` and `statistic` are byte-identical:

- Previous full `f2.py` SHA256:
  `11de4b76d8d06db435a5908ad09bb6fe8d0cf12a02916e98161e3f543be6890b`.
- New full `f2.py` SHA256:
  `d36eaa778dceca49e46a69f8103d6ee745087ee5d0ea9526e5cab06c0a4142f8`.
- Exact block from `def selected_survival(` up to `def _validate_plan(` SHA256,
  identical before/after:
  `23c86e2b3086d55a2fc26bc6a9bc5be4d0b46dced9cf11683ea12a2c466c39d1`.

Existing hash-bound input ledgers, receipts and registrations were not rewritten.
An old input ledger pinning the previous full F2 source does not authorize a new
sealed run with the changed helper; the existing source-closure check still
refuses it. Any future execution needs the operator's appropriate immutable
source-closure update under the existing governance. This fix does not run or
authorize that execution.

## Regressions and verification

The first review reproduction now includes both a real Git replacement and a
real `.git/info/grafts` attack in temporary synthetic repositories. An early
server push `B -> A` predates the cutoff by 60 seconds; result `C` is first
published by a force-push 60 seconds after cutoff. Ordinary local Git is first
shown to report the false ancestry with clean status. Raw identity Git rejects
it, and actual capture/revalidation selects the late server event. The pure
funding statistic with equal retention returns `STOP_FUNDING`.

The second reproduction gives genuine early commit `A` no result ledger, then
replaces it with a tree carrying the synthetic result ledger and sidecars.
Ordinary Git again presents the forged content and clean status. Raw identity
Git refuses it and the actual metadata-binding adapter returns unavailable.
An additional arm deliberately reinstates the old unsafe local Git helper:
server byte verification still rejects capture and the old forged evidence.
The pure funding statistic remains `STOP_FUNDING`.

Additional cases cover each of the six committed files independently, malformed
server blob/ancestry responses, API loss despite valid local objects, caller
`GIT_*` redirection/config/replacement/graft overrides, and actual preparation
of a strikes packet when local binding is unavailable. Existing timing, packet,
source, reserve, claim/replay and synthetic funding-consumer tests are retained.
The mock server uses independent raw Git reads; it does not delegate ancestry
or content responses to the client identity helpers.

All Python checks use `./venv/bin/python`, nice 15, CPU, serial pytest (no xdist),
all six thread limits set to 1 and no more than two test/probe processes:

```bash
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export SNAKE_DQN_DEVICE=cpu
nice -n 15 ./venv/bin/python -m pytest -q tests/test_bl_residual*.py \
  -k 'not sealed_runner and not success_writes and not no_payload_or_fallback and not unpublished_input_refused'
```

Final broad result: **416 passed, 7 deselected in 143.65 seconds**. The seven
tests calling the sealed F2 runner are deliberately excluded. Synthetic funding
integration uses temporary artifacts and stubbed held-out callbacks; there is no
real R-A claim or protected-data read. The focused correction run was
`pytest -q tests/test_bl_residual_f4_census_ra.py -k 'real_publication_receipt or unavailable_local_binding'`:
**4 passed, 32 deselected in 10.99 seconds**, with the same interpreter,
priority and thread settings. The initial independent authority/timing/F2
selection passed **68 tests, 7 deselected in 38.60 seconds**; the broad result
includes all final changes and 28 regressions added in this round.

The live read-only API qualification succeeded for exact commit
`844ba53b4894aac86401e658b1fc9916823723c3`: compare against itself plus the
input ledger and both sidecars at that SHA. The server/local input ledger SHA256
was `19a3d88a421493092b1bcdc12d8edfa2599816a20ac8703f1a2ab1eb02e94bfc`.
Each request passed the existing fresh HTTPS Date validation. No real F2
result was opened and no real publication-evidence receipt was captured.

Black `--check --line-length 100`, isort `--check-only --profile black`, flake8
and `git diff --check` pass on the final diff (three changed product Python
files and three changed test files). Independent read-only review found no
remaining actionable issue after the synthetic fixture corrections.

## Operator handoff and protected state

No dispatch command is needed for this fix. After publication, use a clean
checkout containing this fix for any independently authorized metadata capture
or packet preparation; keep the existing create-only evidence/packet paths and
once-only claim rules. Missing server proof means strikes, and a previously
bound packet that loses proof must be rejected.

Untouched: live D1024 factory, all tmux sessions, `exp/mac-lease.json*`,
`t34-round0-study-01/`, `research/bl_labels/*`, `research/colab_cli/*`, released
checkpoints, serving pins, existing hash-bound ledgers/registrations/receipts.
No Colab action, factory preemption, real F2 sealed run or R-A execution occurred.
