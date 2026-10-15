# T3 launch review (advice, 2026-10-15)

Advisory note from the cloud Claude session, written at the request of the local snake-dqn
session. Not gate evidence, not a decision record. Reviewed at origin/main commit `46b13d5`:
the panel summary, T1 commitment study, T2/T2b screens, the dense ceiling test, the T3 specs,
`research/t3_rollout_labels_20261015/` and the `--label-bank*` path in `train_m3c.py` /
`learner_m3c.py`. I could not open anything under the Mac artifact root, so bank statistics,
`margin.json`, the M6 CTRL `settings.json` and the audit receipt are unverified here.

## Verdict

Launch is justified once the bank audit passes, with four changes before the first run:

1. Resolve `{TD_FLAGS}` to **no Huber change** (`--td-huber-delta 1.0`, anchor floor 0.1). T2 and
   T2b adopted nothing; every delta > 1 bought survival with a mass tax that fails the -30 rule.
   T3 must measure labels alone.
2. Pre-register the decision rule (section 4) before any run starts. The specs define the bank
   and the arms but no verdict rule for the training result.
3. Add the attribution arm in section 3 if the Mac budget allows, because the registered LABELS arm
   changes three things at once versus CTRL (section 2, risk 1).
4. Reuse the five M6 CTRL finals instead of retraining CTRL, if and only if the template's `common`
   flags match `redesign-m6-mac-20261014/CTRL_s10/settings.json` except for perf flags proven
   bitwise. `--batch-scale` changes dynamics and must match. T2 already made this reuse.

## 1. Ranked risks

**Risk 1, confounded arm (highest).** LABELS versus CTRL differs in three ways: the on-policy v8
anchors are removed (`--label-frac 0`, no `--onpolicy-labels`), bank anchors are added, and the
anchor weight is 1.0 for the first 2M transitions instead of 0.1 throughout. The panel measured
the 0.1 anchor at about 0.5% of the loss, so a 10x anchor for 2M is the strongest anchor regime
tried on this recipe. A win or a loss cannot be attributed to the labels. Mitigation: the
attribution arm LABELS_KEEPV8 in section 3 (bank 0.5 + on-policy v8 0.5, same schedule). If only
two arms fit, keep the registered LABELS arm and state the confound in the result doc.

**Risk 2, mass tax.** Every survival gain since M4 (HUBER, HUBER10, HUBER30) came with mass loss on
at least one mix. A bank that teaches "escape, boost if needed" at L >= 700 can trade mass the same
way, and 17 of 46 T1 escapes were boost-only. Mitigation: the mass clause in the decision rule
(>= -30 versus CTRL on every mix) and the greedy boost-rate telemetry. CTRL runs sit at about
0.001; ANCHOR0 runs that lost mass sat at 0.05 to 0.08. Record it per 100k, do not make it a
tripwire.

**Risk 3, stale surviving sets.** Each surviving set was computed under B8's own greedy
continuation for 300 transitions. As the student moves away from B8, an action labelled
surviving may no longer survive under the student's continuation. Mitigation: the 2M hold at
weight 1.0 happens while the student is closest to B8, which is the right order. Add a post-hoc
check: re-roll 200 bank states under each final with the T3 engine and report how many labelled
survivors still survive. Cheap, and it tells you whether a 20M retrain can keep using this bank.

**Risk 4, bank size and anchor exposure.** About 15k rows (10k decisive + 5k counter) feed every
anchor batch of an 8M run, so the bank is seen hundreds of times. The hinge saturates once a row is
satisfied, so memorisation is bounded, but with bank-only anchors there is no on-policy trust
region for 2M transitions. Mitigation: watch `bank_agree`. Expect it to climb from B8's starting
value (the decisive rows are by construction states where B8's argmax is not in the surviving
set, so the starting agree is roughly the counter share, about 0.33) to above 0.9 by 2M. Flat
`bank_agree` means the labels are not being learned; an immediate jump to 1.0 with a mass drop
means the anchor dominates. Both are observations for the result doc, not halts.

**Risk 5, bank distribution.** Decisive rows come only from the 300-decision death window of dying
worlds, pooled over all five B8 finals, with no Q filter; counter rows come from out-of-window
states that pass the fixed `-max Q` prefilter. The bank is therefore all "dangerous late game".
That is intended, but the student of seed k also trains on states from the other four B8s. State
this in the disclosure. Do not reweight rows post hoc; the loader reads `weight` and all rows are
1.0 (`bank_engine.py:569`).

**Risk 6, template gate versus actual manifest.** The template refuses LABELS unless the manifest
has `schema t3-label-bank/v1`, `provenance.collection_mode == "death-window"`,
`stats.decisive_rows >= 10000`, `counterweight_rows == decisive_rows // 2` exactly, a
`dense_go_receipt_sha256` equal to the receipt at `dense/ceiling-go-for-bank.json`, and a
hash-bound `files` map. Any drift between the final shard builder and these keys blocks the
launch at the last minute. Mitigation: run the template's validation block alone against the
audited bank before the launch window (section 5).

**Risk 7, anchor-weight cliff.** `--anchor-hold 2000000 --anchor-floor-at 2000001` drops the
weight from 1.0 to 0.1 in one step. Expect a transient in `bank_agree` and the TD loss just after
2M. Keep the declared schedule, but do not read telemetry between 2.0M and 2.3M as a signal.

**Risk 8, evaluation power and bank hygiene.** T2b resolved +0.07 at a seed SE of about 0.02 with
5 seeds x 96 scripted worlds, so the +0.06 threshold is detectable. Use a fresh round index that
the seed audit shows unused (T2 used 60, its extension 61), 96 scripted + 32 frozen + 32 mixed
worlds per checkpoint, the T2 probe path (rowwise ego2s forward, advisory mask, numba vector61,
fast anchors, no veto), and probe the incumbent once on the same worlds.

**Risk 9, wall clock and CPU contention.** Two arms x 5 seeds x 8M sequential, two actors plus one
learner each. Do not start while the 12-worker collector or the audit still runs: nothing changes
numerically, but the projection from T2's per-run time will be wrong. Project the 8M time from the
smoke run in section 5 and record it.

**Risk 10, disclosure.** The owner ruling allows training-time rollout labels as "no veto" only
with disclosure. The result doc must say the labels came from simulator rollouts of B8 on the
`t3-rollout-labels-dense/v1` namespace, that the served artifact stays a bare network, and that
the bank is development data. Keep `--tight-telemetry` on in every arm so CTRL and LABELS log the
same fields.

## 2. Trainer contract notes (verified in code, no change needed)

- Bank-only anchors are legal: `label_frac 0 + label_bank_frac 1 == 1.0` passes the bv8 check
  (`train_m3c.py:1664`), the bank is validated before shared memory or actors exist
  (`train_m3c.py:2481`), and on bv8 there are no demonstration rows, so every anchor row is a bank
  row from the first update. CTRL instead runs TD-only until 10,000 on-policy labels exist
  (`LABEL_ANCHOR_MIN_ROWS`). That is a small early-phase difference; note it, do not fix it.
- `surviving_margin` takes both maxima over all six actions without the observation mask, so an
  illegal action can be pushed below the surviving set. Harmless and arguably wanted.
- The margin constant is the run's `margin.json` value times `--label-bank-margin-scale` (1.0).
  T1 found the fatal action ahead by a median 0.45 Q units; any margin above about 2 Q units
  covers the p90 gap of 2.55, so there is no reason to scale it.
- Bank rows are sampled with replacement by the learner's own RNG, so the LABELS RNG stream
  diverges from CTRL's. Expected, not a bug.
- RAM: 15k unpacked rows are about 185 MB. Fine on the Mac.

## 3. Launch flags per arm

Replace `{TD_FLAGS}` in `t3_lanes.sh.template` with `--td-huber-delta 1.0`. Writing it
explicitly makes every `settings.json` self-describing. Everything below assumes the template's
`common` array unchanged (8M transitions, 2 actors, 128 worlds, mass reward, gamma 0.998 fixed,
`--eps-end-at 1`, `--mixed-frac 0.25`, bv8, numba vector61, `--learner-fast --batch-scale 2
--obs-pack`, checkpoints every 2M, `--tight-telemetry`), seeds 10-14 from B8_s0-4.

CTRL (reuse the M6 finals if the recipe matches; otherwise run this):

```text
--td-huber-delta 1.0
--anchor-mode v8margin --anchor-hold 0 --anchor-floor-at 1 --anchor-floor 0.1
--onpolicy-labels --label-frac 1
```

LABELS (the registered arm, as the spec prescribes):

```text
--td-huber-delta 1.0
--anchor-mode v8margin --anchor-hold 2000000 --anchor-floor-at 2000001 --anchor-floor 0.1
--label-frac 0 --label-bank "$bank" --label-bank-frac 1 --label-bank-margin-scale 1.0
```

LABELS_KEEPV8 (attribution arm, add if the budget allows; third priority after LABELS and CTRL):

```text
--td-huber-delta 1.0
--anchor-mode v8margin --anchor-hold 2000000 --anchor-floor-at 2000001 --anchor-floor 0.1
--onpolicy-labels --label-frac 0.5 --label-bank "$bank" --label-bank-frac 0.5
--label-bank-margin-scale 1.0
```

The trainer accepts this combination (fractions sum to 1.0 with `--onpolicy-labels`). Its anchor
batch is half bank rows and half on-policy v8 rows once 10,000 labels exist, and bank-only before
that. If LABELS and LABELS_KEEPV8 agree, the labels carry the effect. If only LABELS moves, the
anchor regime does.

Do not run LABELS_RETURN. The template refuses it and no return target is implemented.

## 4. Decision rule to pre-register (mirrors T2, declared before data)

Primary: paired scripted episode-survival delta, LABELS minus CTRL, 5 seeds, fresh round.

- GO: primary >= +0.06 AND mass delta versus CTRL >= -30 on every mix AND no frozen or mixed
  episode-survival drop beyond 0.05. GO means a 20M retrain with the bank and a fresh-bank Phase R
  under the amended guard clause. It does not promote anything.
- KILL: primary <= 0, or a halted run (exit != 0) in the arm.
- Otherwise EXTEND once: 96 more scripted worlds at round + 1 for LABELS and CTRL, re-decide at
  192, stop after that.
- Secondary, reported, not deciding: alive-frame survival, mass versus the incumbent on all three
  mixes, death lengths, final greedy boost rate, `bank_agree` at 2M and 8M, and the post-hoc
  survivor re-roll from risk 3.

## 5. Smoke test before the launch window

Run once with the real audited bank, one seed, a throwaway `--out` under a scratch directory:

```text
LABELS flags above + --transitions 200000 --checkpoint-every 100000
```

Confirm: the template's manifest and hash block passes; the bank loads with `--obs-pack`;
`bank_rows_mean` equals the anchor row count; `bank_margin_loss` and `bank_agree` appear in the
100k log row; no tripwire; wall time per 100k, used to project the 8M run. Then delete the smoke
output. The template refuses an existing `--out`, so a leftover smoke directory with a real arm
name would block the launch.

Also run the audit (`audit_bank.py`) and bind its receipt in the manifest before the smoke test,
not after. The template validates the manifest, not the audit, so the audit PASS is the owner's
evidence that the manifest describes the bank.
