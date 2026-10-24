# I2 FIX-FIRST round 2 — publication and wait authority

This sibling handoff supersedes the publication-time and R-A preparation
instructions in `F2_I2_FIX_FIRST_20261024.md`. Existing ledgers, receipts,
packets, registration bytes and F2 source closure remain unchanged. This is
implementation and synthetic verification only; no F2 sealed run or real R-A
look was performed.

## Review-to-fix mapping

The remaining P1 at `b77f9408` conflated Git committer time with remote
publication time and used local wall time to cross the mandatory wait boundary.

- `github_authority.py` uses GET-only `gh api --hostname github.com`, fixed
  repository `josenunezwork/snake-dqn`, HTTPS GitHub API responses and bounded
  requests. With no `GH_TOKEN`/`GITHUB_TOKEN`, it privately reuses the existing
  HTTPS Git credential without writing auth configuration or exposing secrets.
- Publication is a GitHub repository-activity push transition: the exact
  result-ledger commit is absent from the ancestry of `before` and present in
  the ancestry of `after`. The current authoritative main ref must still
  contain it. Multi-commit pushes work. All available activity pages are
  scanned for the earliest qualifying event; malformed/foreign/cyclic or
  truncated pagination, absent transitions, unknown ancestry or API errors
  cannot authorize a non-strike. Git dates and file times are never consulted.
- `capture-publication` creates one immutable exported receipt next to the
  sealed F2 receipt, named `publication-evidence-<full-ledger-commit>.json`,
  with SHA sidecar and completion marker last. It binds the result-ledger
  SHA256, F2 receipt SHA256, exact ledger commit, repository/ref, push event ID,
  before/after SHAs and server timestamp, plus the capture-time observation.
  Rebases require a different exact commit and new matching publication proof;
  preserved author/committer dates confer no eligibility.
- Packet v3 and the fixed once-only claim carry the revalidated publication
  identity and evidence-file SHA256. Preparation and funding re-fetch GitHub
  activity. Funding revalidates again immediately before claiming; changed
  evidence invalidates the packet before held-out callbacks. Missing or
  unverifiable publication proof selects strikes. A previously verified packet
  whose proof becomes unavailable is stale and refuses; prepare a fresh unused
  strikes packet if appropriate, preserving the existing once-only claim rule.
- The metadata-only binding adapter verifies existing published ledgers and
  receipt hashes before any semantic F2 result read. Missing authority never
  opens the sealed result. `f2.py` and the pinned F2 statistic remain unchanged.
- The packet binds the time-authority policy and a preparation observation.
  Immediately before the atomic claim, funding obtains a fresh HTTPS GitHub
  `Date` observation and uses its epoch as the cutoff input and claimed time.
  The claim/output retain that observation. Missing/malformed Date, non-200
  response, API failure, stale Age, request longer than 30 seconds, a wall-clock
  jump during the request or skew greater than 120 seconds against either local
  request boundary fails closed before a claim or held-out look. Fresh server
  time is required for both funding branches; local time never substitutes.

The original cutoff remains **2026-10-25 18:00 America/Los_Angeles**, strictly
preceded by F2 publication and by the R-A claim for F2 eligibility. Equality is
late. The existing D4096 start authority, reserve identity, preflight, source
pin and once-only claim defenses are retained. Activity history outside
GitHub's retained window cannot prove an earlier publication; an absent
transition fails closed, and a later visible reintroduction cannot backdate a
result. No claim or existing receipt may be deleted to recover admission.

## Read-only repository qualification

On 2026-10-24, the authenticated GET to
`repos/josenunezwork/snake-dqn/activity?ref=refs/heads/main&activity_type=push&per_page=3`
returned HTTP 200, HTTPS Date, pagination and server push fields. A subsequent
probe through the actual new scanner found commit
`b77f94088520b15d7b746a2fae3c27ce3edd7688` first reachable in event
`47653604007`, timestamp `2026-10-24T19:20:18Z`, before
`56c007c473e34a2d950455a6fde786dd59bfd561`, after that exact `b77f9408` commit.
The actual `check-time` CLI also succeeded. These were implementation probes,
not publication evidence for an F2 result. No GitHub API write was made.

## Operator commands — metadata capture, prepare and print only

Use a clean checkout of the published round-2 fix or later main containing it.
Do not update a checkout that owns live jobs. Use unused packet/output paths;
old packet schemas are refused. The source pin must equal the executing HEAD.

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-f2-i2
git fetch origin
git merge --ff-only origin/main
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export SNAKE_DQN_DEVICE=cpu

# Read-only API/time availability check, independent of F2 results.
nice -n 15 ./venv/bin/python -m research.bl_residual.github_authority check-time

# ONLY after the canonical F2 result ledger and its sidecars are published.
# This reads publication metadata/hashes; it does not run F2 or parse its result.
# Capture is create-only. An existing complete receipt is revalidated by prepare;
# never remove or overwrite one to recapture it.
nice -n 15 ./venv/bin/python -m research.bl_residual.github_authority capture-publication

nice -n 15 ./venv/bin/python -m research.bl_residual.f4_ra_prepare \
  --d256-pack /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/f4/d256-metadata-pack-02 \
  --d1024-pack /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/f4/d1024-pack-01 \
  --census-dir /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/f4/r-a-archive-census-01 \
  --funding-out /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/f4/r-a-i2-fix2-01/funding-look.json \
  --packet-out /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/f4/r-a-i2-fix2-01/command-packet.json \
  --f2-receipt /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/f2-i2-20261024/sealed-receipt.json \
  --pin "$(git rev-parse HEAD)"

nice -n 15 ./venv/bin/python -m research.bl_residual.f4_ra_prepare \
  --print-command /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/f4/r-a-i2-fix2-01/command-packet.json \
  --source /Users/josenunez/Projects/ml/snake-dqn-f2-i2 \
  --pin "$(git rev-parse HEAD)"
```

For no published F2, omit the capture command and `--f2-receipt` from preparation.
Missing publication evidence also prepares the strikes branch even if a valid
result ledger exists. All branches still require trusted time. An early D4096
start still forces waiting until timely verified F2 publication or the cutoff.
To bind newly available proof before any claim, use another unused packet/output
pair; the authoritative series/reserve claim remains the same.

The printed dispatch is operator-owned and requires a fresh registered Mac
accounting job ID and the existing factory-preemption procedure. These commands
do not authorize a second R-A look. This implementation did not run the printed
dispatch or `f2 run-sealed`.

## Verification

All tests use synthetic repositories/metadata and stubbed held-out callbacks.
All Python checks use `./venv/bin/python`, nice 15, all six thread settings above
set to 1, CPU, no xdist workers and at most two concurrent test/probe processes.
The seven tests invoking the sealed F2 runner are explicitly excluded.

Final broad regression command:

```bash
nice -n 15 ./venv/bin/python -m pytest -q tests/test_bl_residual*.py \
  -k 'not sealed_runner and not success_writes and not no_payload_or_fallback and not unpublished_input_refused'
# 354 passed, 7 deselected in 113.44 s
```

The focused authority/timing/packet run passed 79 tests in 61.25 s; the final
broad run includes two additional end-to-end cases using the actual synthetic
Git publication receipt through preparation and claim. These prove both exact
evidence-hash binding and refusal before semantic/result callbacks if the
receipt is altered. Missing-proof integration proves no sealed-result parse;
the server-time integration proves the cutoff input is independent of local
time and records the fresh observation. API failure and both clock-skew
directions refuse before a claim. Tests cover delayed publication, independent
author/committer dates, an actual `--committer-date-is-author-date` rebase,
multi-commit pushes, earliest activity across pages, malformed/foreign/cyclic
evidence, and evidence replacement/replay.

Black `--check --line-length 100`, isort `--check-only --profile black` and
flake8 all pass on the four changed product Python files and three changed
test files. `github_authority --help`, `f4_ra_prepare --help`, `funding --help`
and a fresh live `github_authority check-time` all exit 0. `git diff --check`
passes. `f2.py` remains SHA256
`11de4b76d8d06db435a5908ad09bb6fe8d0cf12a02916e98161e3f543be6890b`,
exactly the frozen input-ledger pin; `job.py` has no diff. Independent read-only
review found no remaining actionable product-code finding. No real F2 result
publication-evidence receipt was captured, since this task does not run or open
the real F2 result; the actual server probe above was metadata-only.

The live D1024 factory, tmux, `exp/mac-lease.json*`, `t34-round0-study-01/`,
`research/bl_labels/*`, released checkpoints, serving pins, existing ledgers,
registrations and receipts are untouched.
