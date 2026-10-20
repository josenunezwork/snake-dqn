# T34 Path B v2 implementation protocol — Amendment 2

Written before any v2 efficacy number. Implementation and smoke authority only;
the operator sizes, registers and runs the development pre-check and retention
study. This file is not that study's full pre-registration or release authority.

Binding, in order: `SUCCESS_CONTRACT.md`, `DESIGN_V2_SHADOW.md` Amendment 2,
`P7_STATISTICS.md`, and the unsuperseded final plan. Historical M2 and M3
NO-GO results stand. Neither v1 depth-40 search nor serial LAD search is used.

## Delivered action rule

Bare B8 s3 is the comparator and policy initializer. The exact greedy shadow
looks D=500 transitions ahead. Initial/refork catch-up runs on the one background
worker; thereafter one newly consumed real transition admits one shadow advance.
The shadow stops at hero death. Confirmations bind ordered bodies and food, all
simulator RNG streams, opponent carry including v8 state, frame, hero life,
session and checkpoint identity. Invalidation discards every associated result.
The engine uses private per-world RNG state; speculative work never swaps the
process-global Python or NumPy RNG state.

On doom detection, search branch points earliest-first starting at the first
detection decision. Evaluate alternatives in descending B8 Q, ties lowest action
index. Reuse the prepared source; evaluate alternatives together with independent
mutable opponent/v8 carry and persistent bounded scratch. Native kernels release
the GIL. PyTorch CPU rowwise inference is the default. A batched simulation does
not authorize changing rowwise Q numerics. ONNX requires an explicit flag and a
separate gameplay action-identity receipt before use.

Continuation horizon is 500 transitions, censored by the terminal-world horizon
where applicable. Rank full per-action outcomes by the T3 rule: retain bare if
it survives; otherwise prefer horizon survivors; if all die prefer longest alive;
ties use B8 Q, then bare, then action index. Only a fully evaluated improvement
can be published. Horizon rescues and all-die delays are logged separately;
delay choices are not decisive teacher examples. A result schedules "at frame
t*, take a*" only if t* is still future when published. At execution it must still
belong to the confirmed shadow and match session/life/checkpoint and legal action.
Late, stale, invalid or unavailable results use bare. Interventions invalidate
the old shadow and cause a background refork. There are no hindsight rewinds.

One extra CPU thread, at most one outstanding request, bounded snapshots,
digests, scratch and logs. Polls do not wait for work. No unbounded queue and no
blocking tick. Record proposed versus executed action, interventions, misses,
fallback reasons, worker CPU time and first-result latency. First-result p95
<=2 seconds is an engineering target, not a scientific gate.

## Verification and execution sequence

Before trusting a timing number compare optimized traces against the reference
continuation on recorded cases, including Q ties, deaths, v8 carry and stale
results. Smoke tests may use tiny worlds and synthetic states; they do not
qualify gameplay efficacy or quiet-host serving latency.

1. Development pre-check: the reused T3 dense 64-world bank, bare versus online
   v2 at real 24 fps. Print survivors/64; only >=50/64 admits the fresh-bank run.
   This is reused scripted development evidence and decides no Claim B result.
2. Operator commits the full retention pre-registration, source SHA, checkpoint
   hashes, fresh disjoint collision-audited seed namespaces, banks and load
   profile before running. Fixed N=32 paired terminal H5000 worlds per mix and
   16 fresh continuous-respawn mixed-Watch6 paired sessions. Arms: bare, paced
   v2 at 24 fps, same frozen v2 unpaced (descriptive). One look, Holm over the
   two active arms, P7 paired-world/session inference and equal mix weighting.
   First cap wins: 32 h elapsed / 48 CPU-h / $0. Run after Phase 0 frees the Mac.
3. GO/KILL/INCONCLUSIVE exactly per `m2m3_fable_on_astra.md` section 4. Missing
   rows, identity/parity failure or cap expiry is INCONCLUSIVE, never efficacy
   success. Unconditional faults or timing overruns kill a complete arm. GO
   requires every contract skill and product bar, counting all fallbacks:
   mass >=+40 and LB90>0, scripted no-death gain >=.03, survival guards A/B,
   boost share [.005,.10], paired session death reduction with 95% interval
   excluding zero, mass relative-difference lower 95% bound >=-.05, zero paced
   overruns and update p95<=11.688 ms. Near-boundary uncertainty follows the
   cited advice; no extension or optional second look is automatic.
4. Timing ON/OFF: T15 real Watch6 update and full-tick p95/p99, delivered cadence,
   24/12 fps overruns, shadow share of a core and doom first-result latency.
   Report long-body and contention tails. Quiet-host final timing is deferred
   while Phase 0 runs. Client latency and wider-roster serving qualification
   remain separate release checks.

## Operational boundaries

Mac CPU only; numerical library pools (ORT/BLAS/Numba/Torch) capped at one.
Implementation verification uses at most two CPU threads while Phase 0 runs.
No live services or tmux operations. No default serving, release monitor,
leaderboard, released checkpoint, pin or hash-bound historical record changes.
Assistance remains opt-in and visibly labelled assisted hero. Rollback is
assistance off plus `SNAKE_SERVE_CHECKPOINT=b8-s3`. Operator release is separate.
Teacher label rounds R1+ remain unfunded pending the registered decision or
Round 0; this implementation launches neither labels nor efficacy banks.
