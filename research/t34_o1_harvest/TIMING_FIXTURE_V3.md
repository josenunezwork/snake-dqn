# O1 excluded timing fixture V3 — before any study world

Authority: owner's explicit 2026-10-21 recovery instruction. This sibling
supersedes only the excluded timing roots and Mac fixture settings in
`PREREGISTRATION_O1.md` and the V2 timing correction. Original preregistration,
bank, audits, registrations V1/V2 and failed attempt remain byte-for-byte intact.
The replacement source registration is `REGISTRATION_O1_TIMING_V3.json`.
No dispatch is authorized by this correction.

## Diagnosis from retained failed evidence

`colab-census-01/exports/o1.tar.gz`, SHA-256
`846a286531dc61d5c98a7395b72acba56ad489c8db5052f84fd1ad63ee130351`,
contains start, failure and exactly two excluded timing world payloads; no study
payload. The outer receipt is INCOMPLETE, `stop_verified: true`, no cleanup error.
This is retained operational evidence, not a scientific KILL or GO.

- Frozen world seed 943080529: terminal hero died after 4479 transitions;
  prepared roots 1000 and 3000 were collected.
- Mixed world seed 2671445542: terminal hero died after 2216 transitions;
  only prepared root 1000 existed. Root 3000 was unreachable.
- Both had `frames=5000`, `horizon=500`, no nearby-death events, and
  `event_limit=0`. Event roots were disabled and cannot supply missing roots.
- The earlier Mac preflight used the separate `excluded_preflight_rows`
  (seeds 1441017192 and 1940838160), frames 40, roots 10/20, H8. It never exercised
  the L4 timing worlds, long trajectories or the later-root assertion.

Evidence root: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-o1-20261021/`;
full log `colab-census-01.log`; failed runner elapsed 248.680837405 seconds.

## Fixed mechanical correction, frozen before replacement fixture

Use the SAME two `BANK_O1.json.excluded_timing_rows`, in their original order,
same seeds, rosters, mechanics, terminal hero semantics, H5000 base trajectory
and H500 branch horizon. Select fixed prepared decisions **10 and 20**, with
`event_limit=0`. These early fixed decisions measure branch work without
conditioning on surviving to a late decision. No fallback, replacement, reroll,
event substitution or outcome-based ranking is introduced. If either root is
absent, fail closed before any parity or study world. Early timing roots do not
establish late-state cost equivalence; the retained worst measured component
rates and 2x caution are extrapolations, and live L4 admission remains mandatory.

Mac preflight now uses that EXACT specification and those EXACT two L4 rows,
one CPU worker, nice10, one numerical thread, fresh `preflight/run-03/` outputs.
It retains the existing 180-second local cap. The original Mac-only rows remain
in the immutable bank and audit but are unused by V3. H5000 remains a terminal
hero maximum; do not force a dead hero to complete 5000 transitions.

`timing.py` is the shared spec/validator; both runner paths call the same
collector. The analyzer validates exact timing world/root identities, and the
manifest builder verifies the preflight spec and both referenced payload hashes
and contents. This prevents substituting the old short Mac fixture under V3.

Unchanged: all 48 study worlds, their four time roots 1000/2000/3000/4000 plus up
to four registered event roots, reservoir, ranking, attribution, one-look
KILL/GO rule, parity bank, caps, workers, forecast arithmetic, qualified
transport, lease, immutable exports, Mac strict/serving requirements and owner
release authority. No study world is played during this recovery.

Sequence: publish code/doc; freeze and publish V3 source registration; execute
excluded Mac preflight; prepare and locally validate a new manifest targeting
fresh `colab-census-02`; publish verification. Preserve previous manifests.
No Colab calls. Exports retain `.complete` and `.sha256` sidecars.
