# Guard clause: A, B or C (advice, 2026-10-15)

Advisory note from the cloud session, third request from the local session. Not a decision record.
Sources: panel memo 5.1, Mac answers 4-5, M4 Phase R result, clause c6 in `example_rules.py`,
T1/T3. SE figures below are derived from the recorded paired SE (0.067 at 32 worlds, world
correlation -0.03, so pairing adds nothing): per-world SD of an alive-frame difference about 0.38,
of an episode-level difference about 0.69.

## Recommendation

**B, as the panel wrote it, with the numbers below.** It is the only option whose comparator is
the agent being replaced, and its episode-level floor already carries the product protection.
Pick C over B only if the owner states that the Watch product goal is "the hero rarely dies",
and accepts that the campaign target then becomes the full lookahead ceiling with mass secondary.
Do not pick A under any priority.

## 1. A versus C

**What they measure.** A is time alive out of 5000 frames against a comparator that dies small
and early (median death L 520, frame 2870). A death at frame 4500 costs 0.1, at frame 1100 it
costs 0.78. B8's deaths are late and large (median L 974, frames 3344-4554), so A under-weights
exactly the deaths a viewer notices most. C counts deaths; it is the product-faithful metric for
Watch mode, where the visible event is the big snake dying.

**Power at feasible counts (5 banks x 96 scripted worlds, pooled, 480 worlds).**

| Option | SE | False fail at true equality | B8 as trained |
|---|---|---|---|
| A, alive-frame, -0.05 | 0.017 | 0.2% | census gap -0.039: fails about 25% of the time |
| C, episode, -0.05 | 0.032 | 6% | census gap -0.157: fails essentially always |

A has excellent power but B8 sits at its margin, so it keeps producing PARTIALs that carry no
information. C fails B8 for a real reason and passes a candidate only at pooled scripted episode
survival of about 0.66 or better (champion 0.69, threshold -0.05, half an SE of margin).

**What each demands of the next experiments.** A needs roughly +0.03 alive-frame over B8: a
quarter of late deaths removed, or the same deaths delayed. A candidate can pass A by dying later
rather than less. C needs B8's scripted episode survival to rise from 0.53 to about 0.66, which
is two thirds of the dense lookahead ceiling (0.84 on the s0/s3 banks, less pooled) distilled into
the network. That is the T4 expert-iteration line run to completion, likely two or more cycles.

**Between the two: C.** A combines the wrong metric with the wrong comparator and a threshold the
current policy straddles. C is honest about the product goal and sets a bar that the T3/T4 line
can actually be measured against.

## 2. Is B still preferable

Yes under the goal in STATUS.md. The champion never lives at L >= 800, so a champion guard mostly
asks whether the candidate stayed small, and B8's mass advantage comes from not staying small. B
keeps a survival guard, against the agent viewers see today, plus an absolute episode floor that
no comparator can game. Choose C over B only if deaths per Watch session are the product
complaint; then say so in the pre-registration, as the panel asked, and accept the longer campaign.

## 3. Clause text for B

- **c2 scripted survival.** Alive-frame surv5, hero (no veto) minus frp3-s12 + v8, pooled over
  the five seeds' banks, 96 scripted worlds each, >= -0.03. SE 0.017; false fail at equality 4%.
  B8 recorded -0.003.
- **c2 frozen and mixed survival.** Same metric and threshold, 64 worlds per bank (320 pooled),
  SE 0.021; the current 32 give SE 0.030 and a 16% false-fail at equality.
- **Per-seed floor.** On each seed's own bank, alive-frame surv5 hero minus served agent
  >= -0.15 at 96 worlds. SE 0.039 per seed; false fail at equality under 0.1% across five
  seeds; a seed truly at -0.10 fails 10%. B8's worst seeds recorded -0.110 and -0.109.
- **Absolute floor.** Hero episode survival >= 0.50 on every mix, pooled. SE 0.023 at 480
  worlds; B8 at 0.531 fails about 9% by noise, so B8 as trained is at this floor, not above it.
- **Champion diagnostic.** champion_a5 + v8 measured on all five banks, both metrics, reported
  in the result doc, never deciding.
- **Secondary.** Length-matched hazard and deaths per 1,000 decisions at L >= 800.

If the owner chooses C instead: pooled scripted episode survival, hero minus champion_a5 + v8,
480 worlds, >= -0.05 (SE 0.032, false fail at equality 6%; use -0.07 for 2%), with B's per-seed
floor and absolute floor kept.

## 4. Amendment discipline

Amend once, prospectively, before any new gate data. Fresh banks under a new round index with the
seed audit; never re-score look 2 or the census bank as gate evidence. Publish the sensitivity
table of B8 under the old and new clause on both metrics (seed 0 look 2: -0.156 alive-frame,
-0.188 episode; census: -0.039 and -0.157; pooled vs served agent: -0.003 and -0.050). Disclose
the PARTIAL and its role in motivating the change. No second amendment after a second failure.
