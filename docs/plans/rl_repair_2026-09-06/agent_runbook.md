# RL repair agent runbook

## Purpose and boundary

This runbook turns the planned RL contract repairs into executable, reviewable
agent waves. Every package is **PLANNED** until the root dispatcher explicitly
starts it. This turn creates plans only; it does not start implementation packages,
training, corpus generation, evaluation or publishing. Prior authorization and
preferences persist when execution resumes; do not ask for repeated approval.

Read this with the package cards and DAG in this directory. `dispatch.json` is
the sole ownership and dependency authority. A clean merge or a clean
`git cherry-pick` does not waive a dependency: the upstream package's required
evidence must be present and accepted before a dependent package starts.

## Immutable base and worktree preparation

The current `main` base is `950d583` and is clean. When implementation is
authorized, the root creates one integration branch named
`codex/rl-contract-repair` from the commit that contains this plan and descends
from `6986fd3`; do not silently substitute a newer checkout.

Before any writer starts, the root must:

1. Record integration `HEAD`, `origin/main`, `git status --short`, the resolved
   package card, and `dispatch.json` SHA-256 in a wave receipt.
2. Create one isolated worktree per write package from that immutable branch.
   A package never writes in another package's worktree or in the integration
   worktree.
3. Read `/Users/josenunez/Projects/ml/snake-dqn/AGENTS.md` and provide its
   operative instructions to every agent. `AGENTS.md` is present on main but
   absent from this ignored coverage worktree. The audit overrides stale claims
   in older material about a 58-D-only product, PyQt UI, or certified complete
   parity; current source and the audit evidence control.
4. Freeze each package's source/config/checkpoint/protocol inputs by SHA-256.
   A package that changes one of these inputs must declare a new protocol
   revision before running a dependent command.
5. Resolve `EVIDENCE_ROOT` from the manifest template to an absolute directory
   outside all disposable worktrees. Allocate its `packages/<id>/` directory to
   that package alone. Record the root and allocation in every launch prompt;
   an unresolved placeholder is not a valid artifact destination.

The root owns branch creation, worktree cleanup, dispatch, integration,
heavy commands, and all cross-package interface changes.

## Reusable builder prompt

Copy this prompt and fill the bracketed fields from the package card.

```text
You own only package [ID]: [TITLE] in worktree [PATH]. You are not alone in
the repository. Read the supplied main AGENTS.md instructions, this package
card, and dispatch.json. Base SHA is [SHA]; do not rebase, merge, or edit files
outside [OWNED FILES].

Dependencies [LIST] have accepted evidence: [RECEIPTS]. Do not begin if any is
missing, even if Git reports no conflict. Implement only the stated contract.
Do not silently change defaults, reward semantics, promotion thresholds,
checkpoint interpretation, evaluation authority, or public behavior beyond the
card. Put new tests only in [OWNED TEST FILES].

Request a named test slot from the root before running [NARROW COMMANDS] with
the shared project venv. Wait until that slot is allocated; do not infer a slot
from being dispatched. Release it with the command/exit-code receipt as soon as
the command ends. Preserve logs, failing inputs, source/config/checkpoint/protocol
hashes, and incident artifacts in [ABSOLUTE DURABLE PACKAGE ARTIFACT PATH].
Commit only your owned files with a focused message.

Return exactly: outcome; evidence with file/symbol and artifact paths; commands
and results; changed files; commit SHA; remaining risks/unknowns; and whether
the dependency interface changed. Do not dispatch subagents or run training.
```

## Dispatch and ownership rules

- Start only ready nodes whose `depends_on` entries in `dispatch.json` have
  accepted receipts. Follow the card's `owned_files` exactly: one writer per
  file at a time.
- A package may propose an interface change, but cannot edit another package,
  reinterpret an upstream output, or widen its scope. It hands a concrete
  request to the root, which sequences a new card or amendment.
- Use at most six simultaneous write packages, and only when their owned files,
  tests, artifacts, and hardware lanes are disjoint. Four to six read-only
  reviewers may run in parallel; the root still runs all heavy commands.
- No nested fan-out beyond two levels. Agents do not create tasks or agents
  merely to increase concurrency.
- Every writer commits a verified package. The root serially cherry-picks those
  commits into `codex/rl-contract-repair`, resolves any conflict itself, reruns
  the package test, then records the integrated SHA in the wave receipt.

## Wave gates

Each wave has three gates. They are sequential, not optional.

1. **Entry:** immutable-base receipt; dependencies accepted; ownership and
   resource lanes reserved; source/config/checkpoint/protocol hashes frozen.
2. **Package exit:** focused test proves the stated invariant, `git diff --check`
   passes, failure artifacts are retained, and the writer's return is complete.
3. **Wave exit:** root integrates commits serially, reruns each material narrow
   test on the integration branch, runs the affected test union and changed-file
   lint, and records result hashes. A red gate halts dependent waves and
   preserves artifacts for diagnosis; do not retry by changing defaults or
   discarding evidence.

Use meaningful narrow tests named by the card, such as the owning module's
pytest file and one regression for the witnessed semantic mismatch. Do not add
tests that merely duplicate an implementation line. There is no mypy in this
project. `make test-fast` uses `-n auto` and is prohibited for this plan.

Use the shared interpreter explicitly:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q tests/[owned_test].py
make PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python lint
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q -n 2 -m "not slow"
```

The full suite runs alone at `-n 2` maximum at G0 and final integration, or when
a broad shared change or new failure justifies it. It is not repeated at every
wave exit. Run the required slow parity cases separately. Do not use
`-n auto`, add a type-checker command, or sum overlapping focused shard counts
as a suite result.

## Resource lanes and fail-closed behavior

| Lane | Limit | Rule |
| --- | --- | --- |
| Small CPU checks | Two commands at once | Each command sets one native numerical thread; no other CPU-heavy task shares the lane |
| Full suite | One command | Run alone with `-n 2` maximum |
| MPS learner | One process | Two native CPU threads; no tests, builds, corpus jobs, or evaluation while it runs |
| Evaluation | One serial job | Start only after the learner exits and records its terminal artifact |
| Read-only review | Four to six agents | No heavy commands; root owns any benchmark or corpus command |

The root maintains a queue with named `cpu-1`, `cpu-2`, or one exclusive token
for full-suite, corpus, evaluation or MPS work. An agent requests a slot by
message, stating command, worktree and expected resource use, and waits for a
root allocation before execution. A builder assignment is not a compute token.
Release the token with the exit code and absolute log path. The root verifies
that the process exited before reissuing the token; failures cannot leave a
background worker running. At most two small CPU commands run together, and
an exclusive token requires both small slots and all other heavy jobs to be
idle. Agents can edit or review while compute is reserved elsewhere.

For any authorized learner, start with a 4-GiB RSS cap, 8-GiB MPS-driver cap,
and 6-GiB system-memory reserve. Breach of any cap fails closed: stop the wave,
retain logs/checkpoint/telemetry, and rebaseline before a continuation. RSS and
MPS driver allocation are unified-memory counters; never add them as separate
physical memory totals. This turn launches no runs. On later execution, honor
existing authorization and the selected package/protocol without repeat check-ins.

## Evidence and handoff format

Each package retains, under its card-defined absolute durable artifact directory:

- immutable input and output SHA-256 values for source, configuration,
  checkpoint bytes, protocol, dispatch card, and relevant command output;
- exact command lines, exit codes, test logs, and a short interpretation;
- witnessed failure input and output, or an explicit statement that no failure
  was reproduced; and
- a machine-readable receipt that names base SHA, package commit, integrated
  SHA when available, dependency receipts, and remaining risk.

Return absolute artifact paths, not paths relative to a package worktree. The
root verifies the durable files and their hashes, freezes an accepted receipt
and adds it to the campaign manifest before accepting a dependency. Tracked
summaries may live in the card's owned repository directory; they link to the
durable raw evidence and its manifest. If any tool wrote an artifact inside a
worktree, copy it into the allocated durable directory and verify byte hashes
before cleaning that worktree. A Git commit alone does not preserve untracked
logs, checkpoints or failure reproductions. The root owns cleanup and cannot
remove a worktree until this evidence check passes.

Checkpoint and protocol changes require forward and legacy-read tests. Any
repair that changes target semantics, reward, observation/action meaning,
evaluation metric, or world rules requires an explicit version and migration or
rejection behavior. Only REL may produce a real qualification/promotion receipt,
through E2's strict authority and the selected frozen protocol. Other packages
may produce labeled test fixtures or readiness receipts. No package silently
changes a production default or copies/promotes a champion.

At the end of each wave, the root publishes one concise handoff: completed and
blocked package IDs, integrated commits, evidence paths, exact gates run,
resource observations, unchanged defaults, and the next ready DAG nodes.
