# Revised P missing-measurement probes — 2026-10-29

**Prepared only. Successor envelopes remain UNQUALIFIED; DRAFT unchanged.**
This supersedes ONLY the prepared follow-up section of `FOLLOWUP_20261029.md`.
Its hash audit, measured tables, host comparison and deadline analysis stand.
The first unexecuted `packet-v3` is rejected/superseded and must not be dispatched.

## Independent review response and reduced scope

Argon reviewed source `14c1d04098b22e0510ebae4bb1a71aa71ffca8db`, read-only in its
own detached worktree, and rejected the combined engineering packet. Its sizing
concern is accepted: combined baseline work was already ~885–1145 s inside a
600/1200-s guard, with little long-body headroom. The work is now three independent
native L4 probes with distinct manifests, output/state directories and receipts:

- **contention:** original four excluded mature engineering roots; only the
  missing homogeneous w8 and w11 cells, 11 one-tick P0 tasks per cell. Check opaque
  identity across these two cells. Completed serial/w4 mixed sweeps are not rerun.
- **bare:** the first canonical v2 engineering seed, one complete 6,000-tick
  ordinary bare session, with construction/chunk/snapshot timing and no windows.
- **tail:** exactly the FIRST canonical v2 engineering seed, fixed prospectively,
  never substituted if unsupported. A separate 5,000-tick bare prefix (not a complete session), with
  constructor/chunk/snapshot timing. At 250-tick samples select first-largest alive
  body (peak), and latest alive L600 root (late). Export their engineering
  frame/length metadata; no scientific outcome payload is opened. On each root,
  run the unchanged exact P0=8/P48=64/bare=64 windows once, plus one repeated bare
  window for identity. Four workers are nominal; only three distinct arms and
  one small duplicate are queued, so actual overlap must govern interpretation.

The exhaustive all-legal continuation stress is **removed**, as are three of
the four full bare traces and repeated P0/P48 windows. The third probe separates whole-bare timing from exact tail windows so that each
whole-session baseline estimate fits the native 15/30-minute envelope. Full searched sessions,
late alternative-branch tails, cross-worker tail parity and sustained large-worker
tail throughput remain explicit missing evidence. This is an incremental bounded
measurement package; it cannot alone fill the main registration. Engineering
scope reduction does not change the proposed scientific N=32 or any scientific
controller, threshold or endpoint. All engineering v2 seeds remain excluded.

## Prospective envelope arithmetic

Use observed L4 mean per-frame rates for the baseline calculation, without
claiming any unobserved long-body uplift. Contention forecast sensitivity:
~116 s root preparation + ~80 s for the two missing sweeps including warmup,
~196 s child work, or <240 s at a 20% engineering allowance. Bare sensitivity: 6,000 × serial mean bare 0.058705 = 352.23 s, plus one pool barrier.
Tail sensitivity: 5,000 × 0.058705 = 293.525 s; two mixed short-window groups at
roughly their ~41-s P48 critical path + repeated bare scheduling, about 90 s;
three pool barriers about 10 s. Total ~394 s, or **~473 s at 20% child-work allowance**.
These are finite engineering projections from early roots, not measured tail
bounds or successor forecasts. Unsupported long roots and excess time remain
possible and must be retained as missing evidence without replacement/retry.

For EACH probe: base/effective child work **600/1200 s**, base/effective whole
allocation **15/30 min**, L4 ceiling **1.54 u/h**, caps **0.39/0.78 u**, reserves
**0.078/0.156 u**, whole-timeout exposure **0.77 u**. Three original probes expose
**2.31 u** across **1.5 runtime-hours** at their whole timeouts (three cap guards total
2.34 u; reserves total 0.468 u). These are runaway guards, not credit rationing.
Observed prior non-child elapsed was about 440–484 s, including bootstrap,
transport, polling/export/stop. Baseline whole-session sensitivities are about
636–680 s for contention, 795–839 s for bare and **834–878 s for tail**. Round
these up to the 900-s whole base and apply exactly 2× for the 1800-s whole guard.
The 20% allowance is an additional child-work sensitivity, not an extra multiplier
silently applied to the whole forecast. Unknown long-body/window uplift is not
called measured; first-cap-wins and immutable partial timing exports still apply.
No automatic retry or assumed free account assignment. Recheck all rates/slots
at actual dispatch. Keep the unchanged upload route and native transfer/staging
caps. Short-canary resume exemption applies only to these engineering probes.

## Wiring, publication and tests

- Existing `tools/ops/dispatch.py::supervisor_launcher` ALREADY recognizes
  `p_profile_source_delivery` (lines 517–521 at the starting source), validates
  `host.require_profile`, and selects `research/watch_lead_p_successor/launch.sh`,
  which executes `python -m research.watch_lead_p_successor.host`. Argon's first
  routing blocker arose from omitted selector source, not a missing route. A new
  offline selector test verifies it. Native `research/colab_cli` stays unchanged.
- Worker watchdogs now use the active pool's remaining absolute coordinator
  deadline + 5 s; task pulses no longer install a stale 900-s timer. Owned-worker
  cleanup still runs at the coordinator deadline. No worker replacement/retry.
- Unsupported peak/late roots ALWAYS publish explicit supported=false empty
  summary exports, so required-export validation and negative support agree.
- The restore timing wrapper uses `staticmethod` to preserve both class and
  instance access. The real `LiveWorld.restore` is a classmethod; `clone` uses
  deepcopy, not instance restore. Extra tests nevertheless exercise instance
  restore under the actual scheduler/search seam, and count initial/all/final
  digest timing. An actual local mature-root P0/P48 wrapper check is also required.
- The scan observer is removed in `finally`, never serialized into a root. Root
  size is bounded by P's 16-MiB rule. Preflight now refuses orphan components.
- The host adapter admits only the three frozen phase strings; the phase is bound
  in manifest env and passed explicitly to the common entrypoint. Each phase
  has its own real local smoke and source/manifest pin. No prepared main shard,
  scientific data, registration ratification, release or serving change exists.
