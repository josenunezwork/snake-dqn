# T11 LOCAL30 screen result — 2026-10-17

**Decision: KILL, before gameplay evaluation.** LOCAL30_s13 exited 1 and halted at
300,160 transitions: greedy boost for lengths below 700 was **0.010729699488491048
(1.072970%)**, above the registered **0.01 (1%)** limit. Its actor exits were
[0, 0], but the trainer halt/nonzero exit independently fixes KILL. There is no
seed-13 final checkpoint. The retained `aborted.pth` is failure evidence and was
not evaluated or substituted. No retry, best-checkpoint selection, extension,
retrain, or promotion was performed.

Authority: [probe spec](specs/t11_probe_spec_2026-10-17.md) and
[binding preregistration](specs/t11_prereg_2026-10-17.md). “Finished” in the probe
handoff includes one failed run; it does not establish five completed 8M runs.
The four completed finals predate the seed-13 failure in their summary timestamps.

## Registered gameplay result

Primary paired scripted episode-survival delta versus CTRL: **null**.
Paired training-seed SE: **null**. Evaluated worlds/pairs: **0**.
The required five-seed final roster does not exist, so neither a four-seed screen
nor cached results from another round are substituted.

| Mix | LOCAL30 mass | Mass delta vs CTRL | Mass delta vs incumbent | Episode survival delta vs CTRL | Alive-frame survival delta |
|---|---:|---:|---:|---:|---:|
| Scripted | null | null | null | null | null |
| Frozen | null | null | null | null | null |
| Mixed | null | null | null | null | null |

Round 74's registered allocation remains 96/32/32 H5000 worlds per checkpoint;
round 75 was not started because KILL does not authorize EXTEND.

## Training and bank-agreement readouts

Every available 100k window is retained in
[analysis.json](../../research/t11_screen_20261017/results/analysis.json): 80 each
for seeds 10/11/12/14 and 3 for seed 13, **323 total**. `all_windows` and
`per_seed.*.per_100k_windows` preserve individual values; `per_100k` gives the
available-seed mean and sample count. Strict five-seed family means are null
once seed 13 has no window. Final five-seed 8M means are **null**.

| Seed | Transitions | Exit | Final 8M decisive agreement | Final 8M counter agreement | Last-window greedy boost (<700) |
|---|---:|---:|---:|---:|---:|
| 10 | 8,000,000 | 0 | 1.0 | 1.0 | 0.003004808 |
| 11 | 8,000,128 | 0 | 1.0 | 1.0 | 0.001718350 |
| 12 | 8,000,000 | 0 | 0.9998397435897436 | 1.0 | 0.002213542 |
| 13 | 300,160 | 1 | null | null | 0.010729699 |
| 14 | 8,000,000 | 0 | 1.0 | 1.0 | 0.002153446 |

Four completed finals only (descriptive n=4): mean decisive agreement
**0.9999599358974359**, counter **1.0**. Seed 13's last, nonfinal window had
**0.9874920076726342** decisive and **1.0** counter agreement. Both families
exceed 0.9 in these available terminal windows, supporting the preregistered
learnability interpretation; this diagnostic does not override the boost halt.

These are the trainer's separate family metrics: raw six-action argmax,
preferred-surviving-set membership for decisive, and stored B8-action equality
for counter. Family draw denominators were not recorded in the source logs and
remain unavailable/null; no aggregate `bank_agree` is substituted for them.

The [boost curve](../../research/t11_screen_20261017/results/boost_curves.svg)
shows available-seed means, with n=5 through 300k and n=4 thereafter. Individual
seed curves and length-bucket decision denominators are retained in JSON.
A mean can hide an individual tripwire; seed 13's three <700 rates were
0.000588681354, 0.008233173077, and **0.010729699488**.

TD loss, anchor loss, bank-margin loss, Q maximum, tightness, death rates and
mean death lengths are retained for every available window. Null tightness
values remain null. Training death lengths are descriptive and are not
substituted for fresh-evaluation death lengths.

## Other preregistered readouts

- **Outside-bank CTRL-owned states:** raw flips, consequential flips (CTRL Q gap
  >0.3), cross-boost share and standardized support are **null**, with zero
  captured states. No fresh trajectories were produced after KILL. Difference
  from the historical **0.99%** drift reference is **null percentage points**.
  No alternative bank weights/fingerprints, extrapolation, or reused trajectory
  estimates are substituted; recomputed same-world B8/CTRL drift is also null.
- **Shared survivors:** mean length delta versus CTRL and pair counts at frames
  1000, 2000, 3000, 4000 and 5000 are **null / 0**, on every mix. The implementation
  records post-frame masses and conditions on both paired heroes alive.
- **Deaths within 30 decisions after decisive-like states: not measurable.**
  Candidate and CTRL coverage is zero; death counts/rates are null. Validated
  branch-replay trajectories under the original 300-decision continuation
  protocol were not generated. No Q proxy or overlapping-window death count is used.

## Freshness audit and execution

The refreshed [round-74 audit](../../research/t11_screen_20261017/results/seed_bank_audit_round74.json)
**PASSed** before any gameplay. Both reserved rounds have 96 unique seeds with
zero registered, observed, running-T7 or cross-round collisions. The receipt
publishes both seed lists/digests, complete source inventory, historical bank
namespaces and the complete T7 namespace. The immutable preregistration audit,
bank and source files were not edited. No worlds were consumed by this task.

The CPU-only report ran successfully (exit 0) in this task's newly created
`tmux t11-probe`; the session exited with the report. No existing sessions or
unowned processes were inspected or controlled. No gameplay processes started.
The reusable watcher caps workers at three single-thread CPU children plus one
watcher (four processes total). All task numerical work used CPU/thread limits.

## Checkpoint and evidence identities

The full source summaries, terminal receipts and SHA-256 map are preserved in
[training_receipts.json](../../research/t11_screen_20261017/results/training_receipts.json).
Each final checkpoint below is fixed at the completed run's final; seed 13 is
explicitly missing.

| Seed | Final checkpoint SHA-256 |
|---|---|
| 10 | `64d5355b9ac103fea2859d610e8c1c00919191d60a29ea192dc3259d4abbd651` |
| 11 | `e993e4d510af3ce4401ae475362d724d0f869b1b38b1a2a926c047b19a2d5cc8` |
| 12 | `79f317f3e68f934665d69fad79cf3dcc68f474ce0793ee27fcbd6cf575eb8b22` |
| 13 | `null (halted)` |
| 14 | `209414a7243f1ee8b782424c80cb88a7d2e58c3a602b1f5d20765e7e6383fe6c` |

Seed 13 aborted checkpoint (not evaluated): `f2bf8ace4ec8bd64757592b88d35ed516a2a0f6f4ca4e9413b4a62945d20404f`.

Validation: focused T11 screen tests, flake8, Black and independent read-only
review; exact final test counts and implementation identity are recorded in
[completion.json](../../research/t11_screen_20261017/results/completion.json).
The Git workflow pushes every task commit to `main` and `t4p-20261016` after
fetching and merging `origin/main`. Pre-existing untracked T4e files are preserved.
