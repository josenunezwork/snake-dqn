# A0 repair and A1 ruler — 2026-10-22

Authority: owner's A0 work order, `big-lever/PLAN_A100.md` §3 (SHA-256
`1c87898be554d8e66788da6f0a09ee7b842a514e530c1d0e1c624bcf58c3962c`),
and `big-lever/round-20/JUDGMENT.md`, read in full before implementation.
This is the reopened E1b A100 development lane. Claude dispatches and takes the
ruler look. No GPU result is strict, product, serving, or release evidence.

## Repair and independent evidence

`device_sim_frozen.py` is the verbatim pre-A0 simulator at
`01334e1ed04b49ebf61ab16a6df0cf6c7e4c2884`; its SHA-256 is
`7fbabf67b439c6e8ebf2cbb252abed5dadcc1ce08194a8d6177605ab4bbe8d88`.
The candidate rejects empty death events before corpse assembly and compacts
masked segments in stable death/body order. CUDA insertion/deduplication/FIFO
runs in one native launch; its serial loop is over actual candidates. CPU uses
the same bounded tensor algorithm for tests; meta checks retain static shapes.
Capacity stays 10,400. Food maintenance selects needy worlds, reuses occupancy,
and advances every counter by the original fixed `max_food` draw count.

Codex-V's `fixtures/a0_event_oracle/CODEX_V_REVIEW.json` records 11 CPU tests,
including all five fixture classes at capacity 10,400, 1,000 maintenance ticks,
and full/refill/no-space/exhausted rows. It does not claim native qualification.
The runtime **must pass** `event_cuda --selfcheck` against the frozen oracle
before the ruler; that compares full food/corpse buffers on unequal actual
2,304/100-candidate streams padded to 62,400. Native compilation and launch
counts require the same-pin A100 canary. No CUDA hardware was used by Sol.

## Frozen ruler and decisions

The main job measures all nine E=6/256/1,024 × scripted/frozen/mixed cells, six
800/900/1,000/1,100/1,300/1,500 bodies, B8 s3 and the pinned real mix opponents.
Each cell has three warmed repeats × eight complete ticks. Full timing includes
resident reset, food/respawn, policy/v8/vector61/observation, movement/death and
collection writes. TF32 is disabled; fp32 remains the execution regime.
A separate instrumented pass records exclusive stage seconds and actual CUDA
kernel counts. One frozen unrepaired E=6 tick is the same-allocation control.
A tiny H1 recorded CPU action-tape replay is classified separately; it cannot
qualify the engine. Unexplained mismatches are reported as bugs, never RNG claims
without evidence. Setup and local output serialization are reported separately;
final transport/export seconds come from the supervisor terminal evidence.

- Unexplained oracle mismatch: **INVALID**.
- E6 full-tick speedup **<5×**: event-path hypothesis rejected; further kernel
  work requires a trace naming a stage **≥50%**.
- Median complete steps/s at E1024 **<2,000 on any mix**: **STOP_ENGINE**;
  WP1b/WP1c/WP2 unfunded, WP3/WP4 may use CPU-labelled banks only.
- **2,000–20,000**: fund WP1b and repeat the same ruler; stop if still <20,000.
- **≥20,000 on every mix**: fund WP1c and WP2 tranche 1.
- Missing coverage, timeout, transport stall, or missing export: **INCOMPLETE**.
  One outcome-blind mechanical retry only after verified stop and within the
  original series envelope; a second stall goes to the judge. Never retry a
  scientific failure or promote partial evidence.
- Two consecutive mechanics/parity INVALID packages pause the engine lane for
  a judge. Sealed/gate-data access stops the lane. Mac lease overlap invalidates
  the attempt; no Mac timing workload is part of this package.

## Dispatch envelopes and preparation

A1 forecast: one allocated A100-40 hour, roughly 5.4 units at the plan's historical
rate; live admission must use current account evidence. Main base/effective
caps are 1h/10u → **2h/20u**, reserve 2u → **4u**, rate ceiling **10u/h**. Canary
base/effective envelope is 9min/1.5u → **18min/3u**, reserve .3u → **.6u**.
The 18-minute priced cap is inside the plan's ≤30min/≤3u ceiling. First cap wins,
including setup/export. Rates and N are not doubled. No unpriced substitution.

The existing qualified transport tree is
`553af3f692515b2bf66173356fe4d099acc96942`; it already accepts A100. No transport
code changed and no source requalification is required. The A1-specific,
plan-SHA-bound ops exemption permits 120-minute acceptance only for this exact
ruler, leaving generic acceptance at 60 minutes. It is covered by the owner's
explicit work order selecting PLAN_A100's exact caps.

From this worktree, after publishing the reviewed source commit:

```sh
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
ART=/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-a1
"$PY" -m research.gpu_sim_spike.make_a1_pack --commit FULL_PUBLISHED_SHA \
  --artifact-dir "$ART" --validate
```

This creates the canary, main template, empty ledger and `DISPATCH.txt` containing
absolute commands. It verifies the published source and qualified transport pin.
Claude first populates the reviewed endpoint→positive-rate inventory (≤3 total
assignments), then dispatches the canary exclusively through `tools/ops/dispatch.sh`.
After PASS, verified native/ruler exports and confirmed backend release, Claude
binds that immutable receipt using `--canary-receipt` and validates the resulting
main manifest. The main template is intentionally not dispatchable before this
proof. Keep the source pin and input inventory identical across both jobs.

Prepared only. No allocation, dispatch, tmux mutation, checkpoint change, serving
pin change, or sealed-data access was performed.
