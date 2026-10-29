# P successor packet-v4 assessment — 2026-10-29

**INSUFFICIENT FOR SUCCESSOR ENVELOPES. DRAFT remains NOT RATIFIED and is not
READY-FOR-RATIFICATION. No scientific shard packet or dispatch is authorized or
prepared.** All three bounded engineering probes succeeded; successful engineering
execution does not supply the measurements explicitly absent from their design.
This sibling assessment preserves the original `DRAFT_REGISTRATION.md` bytes.
Astra is the designer. No 09e outcome payload, successor scientific world, effect
estimate, variance, bootstrap, reducer or partial-data analysis was opened/run.

## Verified custody and execution

Artifact root: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol/`.
New evidence: `astra-p-successor5/analysis-verified.json`, produced by the committed
`assess_followup.py`. It combines only the five named engineering outputs. Prior
A100 and L4 outputs retain their INCOMPLETE status; their completed measurements
are usable only within their disclosed engineering scope.

| Output | Verified JSON payloads | Receipt status / return code | Whole elapsed s | Transfer s / 150-s margin |
| --- | ---: | --- | ---: | ---: |
| packet-v4/contention | 30 | PASS / 0 | 613.942670 | 136.146006 / 13.853994 |
| packet-v4/bare | 4 | PASS / 0 | 643.540861 | 141.505044 / 8.494956 |
| packet-v4/tail | 14 | PASS / 0 | 610.206484 | 94.563966 / 55.436034 |
| Earlier A100 profile | 93 | INCOMPLETE | 1383.862451 | 98.699687 / 51.300313 |
| Earlier L4 profile | 83 | INCOMPLETE | 1340.049930 | 68.341661 / 81.658339 |

The 48 new payloads have 96 matching SHA/name sidecars and marker-last completion
files: 144 verified components. Earlier profiles add 176 triples, giving 224
verified engineering payloads / 672 components overall. Exact receipt inventory
equality rejects missing, duplicate and unlisted exports; no orphan components.
Individual task exports agree with their sweep summaries and overlap is recomputed.
All three new packet seals, dispatched-manifest hashes, uploaded bank/checkpoint
identities and source archives verify. Each archive contains 320 files / 6,739,829
unpacked bytes, matching published source pin
`d7e5a1a51ea99ad3b8bffc3f10b9805dcf8bf7de` and current source bytes for those paths.
The compressed archive identities differ across jobs; equality is established per
file against Git, rather than by assuming equal tarball hashes.

New receipts record `stop_verified=true`, `cleanup_error=null`, and succeeded
terminal children with returncode 0. Retained final account session inventories
omit each allocated endpoint. Ops logs are under the artifact root's `ops/logs/`,
not this worktree; all three end `EXIT=0`. These are retained execution/release
receipts, not a live session check. Each calibrated allocation counted three
account assignments, including two other A100 jobs; none was stopped or altered.
New L4 rate is 1.54 u/h, same as the prior L4; prior A100 rate is 5.30 u/h.
The three new receipt elapsed times imply 0.798956 modelled own units, not an
isolated provider debit. Original probes were each priced for the entire 30-min
timeout (0.77 u; cap 0.78 u), including setup/export/idle/stop.

## Measurements that are now available

All new hosts expose Xeon CPU @ 2.20 GHz, 12 logical / 6 physical cores, affinity
0–11 and no visible cgroup CPU quota. Workers assert Torch intra/inter-op=1.
This remains CPU work; no L4 GPU throughput is credited. The prior comparison
still favors L4: A100 offered no material measured CPU wall-time advantage at
3.44 times the hourly rate.

| L4 homogeneous 11-task P0 set | w1, earlier | w4, earlier | w8, new | w11, new |
| --- | ---: | ---: | ---: | ---: |
| Sweep wall seconds | 180.701 | 50.724 | 35.692336 | 29.935069 |
| Sampled active CPU cores | see prior report | see prior report | 6.549618 | 10.202477 |
| Mean per-task overlap | see prior report | see prior report | 7.112443 | 10.684476 |

These one-tick tasks execute one H300 search each; ~30 seconds of homogeneous
search pressure is useful finite scaling evidence, not sustained full-session
tail throughput. New w8/w11 seed/arm/frame/initial/final/event-trace identities
agree exactly. Earlier serial/w4 and new w8/w11 event-trace hashes agree for all
11 tasks, **but all 11 initial and final full-state hashes differ across the old
and new allocations**. Thus the table compares timing sensitivities on the same
seed/task definitions; it does not prove one byte-identical full-state grid.
Absolute uploaded checkpoint paths are retained in GameSession/policy state and
included by the digest (`live.py::digest`, `served_tick_probe.py::_canonical`,
`session.py`, `ego2s_policy.py`). This is a plausible explanation, not a proven
diagnosis. Do not normalize existing hashes or waive the discrepancy. A future
same-pin engineering check must identify the differing fields without opening
scientific outcomes, or rerun a prospective grid with explicitly comparable
checkpoint/input provenance.

The fixed first v2 engineering seed's complete bare session takes **196.491119 s**
for 6,000 ticks, 0.032748520 s/tick, including constructor 0.154380 s and snapshot
observer overhead 0.883445 s. Per-1,000-tick chunks rise from 19.901 to 38.326 s.
This single trace measures bare work only. Cold worker startup takes 23.267049 s,
including initialization 21.557831 s and first primer step 21.090904 s; those
components are nested, not additive. Later warmed pools take about 2.6 s.

The separate tail scan stops at 5,000 ticks (166.834567 s). Its frozen peak and
late selection both choose the **same** frame-5000 length-942 root. The required
peak length >=1,000 test is false, so exported long-body support is false. The
bare probe reaches length 1,022 at frame 6,000, but no P0/P48 window was run on
that root; it cannot be substituted retrospectively into the tail result.

| Same root, two separately timed labels | P0: 8 ticks | P48: 64 ticks | bare: 64 ticks |
| --- | ---: | ---: | ---: |
| Inclusive seconds, peak label | 30.988952 | 36.288415 | 3.829202 |
| Inclusive seconds, late label | 31.296388 | 36.418739 | 3.760243 |
| Seconds/tick, peak / late | 3.873619 / 3.912048 | 0.567006 / 0.569043 | 0.059831 / 0.058754 |

Each searched window executes two H300 searches. P0 total digest time is
8.348–8.438 s and restore time 0.0473–0.0476 s; P48 digest time is 7.666–7.690 s
and restore 0.0466–0.0484 s. These costs are already inside inclusive wall time.
They are not a full-session additive constructor/finalizer estimate. Both labels
and duplicate bare windows agree in opaque trajectory/state identity within this
allocation. They are repeats on one root, not two independent tail regimes.
P0 task overlap is only 2.242–2.246, P48 2.064–2.067 despite nominal w4; aggregate
sampled busy CPU is ~2.06 cores. No w8/w11 tail pressure was measured.

## Precisely what still blocks registration and a first shard

1. **Full searched-session costs and tail uplift L:** zero complete natural-start
   6,000-tick P0 or P48 engineering sessions. Need prospectively fixed excluded
   fixtures at the real endpoint, inclusive construction/finalization and observed
   expensive search tails. The length-942 windows and bare length-1,022 observation
   do not measure searched long-body or costly late H300 alternatives. Freeze any
   new engineering fixtures before execution; no replacement of the unsupported
   first root after seeing support. No unobserved uplift is set to 1.
2. **Residual contention and scheduling:** short w8/w11 bursts do not price sustained
   tail work. Select concurrency from measured full-session work and exact
   session-grouped scheduling, and resolve cross-allocation state-hash identity.
   Without these, q_residual and a whole-session shard makespan remain unknown.
3. **Transport headroom:** existing transfer succeeds but bare is only 8.495 s
   under the cap; upload/setup elapsed is 146.929 s. A future packet needs exact
   checkpoint inputs staged and hashed before sealing, plus a supported measured
   delivery route with margin inside 150 s. Local copies alone do not demonstrate
   remote transfer speed. Preserve exact B8/FRP3 identities; do not remove weights,
   relabel hashes, enlarge the cap or assume this probe adapter admits long jobs.
   `host.py::require_profile` intentionally permits only 30-minute native probes.
4. **Executable successor resume/union proof:** the new namespace/allocation,
   immutable whole three-arm sessions, empty first ledger, metadata-only exact
   skip, same-host pairing, corruption/foreign/duplicate refusal, missing-N look
   gate and at most two MANUAL outcome-blind completions per shard are still a
   specification. Existing predecessor runtime constants/ledger must not be used
   as a successor implementation. Need offline proof and a current-pin target
   canary for the actual successor package, not these profiling phases.
5. **Frozen complete operational envelope and independent authority:** after 1–4,
   freeze fresh-bank collision audit, disjoint session shard indices, L4 host and
   software identity, workers, measured estimates and source/manifest/input hashes.
   Compute base timeouts from whole shard costs and effective timeouts = 2× base;
   price the ENTIRE effective timeout, double caps/reserves exactly once, include
   all originals and two completion allowances per shard. Every shard must remain
   **strictly below 24 h including overhead**, and account-wide assignments must
   remain <=3 including other work. No free slots are presumed. The official
   [Colab FAQ](https://research.google.com/colaboratory/faq.html), checked 2026-10-29,
   describes Pro+ execution up to 24 h with sufficient units and variable resource
   limits; it does not guarantee uptime. Ratification is a later independent act.

Consequently **L, q_residual, full-session c and measured scheduler/transport
envelopes remain UNFILLED**. There is no justified first scientific shard timeout,
whole-timeout unit price or <24-hour completion margin yet. The requested
conditional READY-FOR-RATIFICATION label and first-shard dry run are not issued.
This is an evidence insufficiency, not a credit/budget refusal. No new probe,
allocation, main manifest, seed allocation, dispatch or automatic retry was made.

## Independent reviewer checklist before any future READY label

- Reproduce export custody, packet/source/input identities and terminal stop/
  retained disappearance evidence; distinguish completed engineering evidence from
  missing qualification and a live account check.
- Check the frozen support rule against length 942, identical peak/late roots and
  absence of full P0/P48 sessions; reject extrapolating these into measured tails.
- Explain old/new state-hash differences; event-trace equality alone does not
  establish full state/RNG/provenance identity across allocations.
- Recompute inclusive arm costs, actual contention, exact shard scheduling and
  setup/staging/bootstrap/control/export/stop allowances without double counting.
- Inspect staged exact checkpoint bytes, archive/input sizes and measured route
  margin below 150 s; verify the actual long-job route and source pin.
- Verify new bank/allocation audit excludes 09e and every burned engineering seed;
  preserve N=32, three arms, H6000, W1/K1/H300 and P48-only primary inference.
- Audit whole-session ledger/publication/retry and complete-N reducer tests,
  same-host pairing, refusal paths and current-pin target qualification.
- Recompute 2× timeouts/caps/reserves, whole-timeout prices, full attempt-series
  exposure, <24 h per shard and <=3 account assignments; units are runaway guards.
- Check bootstrap definitions, zero-death handling, thresholds and decision
  precedence against the DRAFT; confirm no closed-bank look or seed substitution.
- Require the other implementer (Argon) to cross-review Astra's executable package;
  independent campaign judge ratifies prospectively. READY is not RATIFIED, and
  neither grants serving/release authority. Only then prepare/validate/dry-run a
  first scientific shard; dispatch remains an explicit operator action.

## Reproduction

Use `/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python`, CPU and single-thread
OMP/OPENBLAS/MKL/NUMBA/VECLIB/OMP_THREAD_LIMIT/TORCHINDUCTOR caps. The offline command
is `python -m research.watch_lead_p_successor.assess_followup --artifacts
/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol --output FRESH_ABSOLUTE_JSON`.
It creates its report exclusively and performs no gameplay or provider calls.
Focused tests cover corrupt/missing/symlinked payloads, invalid sidecars, orphan
markers and receipt duplicates/missing entries. Verification and review evidence
are retained beside `astra-p-successor5/SUMMARY.md`.
