# Prospective amendment 05 — outcome-blind DENSITY-2 completion

Authority: ruling (1) of `docs/research/advice/2026-10-21_big_lever/JUDGMENT_RESUME_POLICY.md`
(the identical artifact ruling at `big-lever/JUDGMENT_RESUME_POLICY.md` was read in full),
plus the owner's 2026-10-22 instruction doubling all unit caps, reserves and timeouts.
This sibling supersedes amendment 04's no-completion clause only as ruled. Amendments
01–04, prior manifests, exports, receipts and all scientific thresholds stay immutable.
No outcome field of any production episode has been opened in this preparation.

The decision to complete is unconditional and irrevocable, before any outcome read.
Any premature outcome read voids completion and yields INCOMPLETE_PEEKED. Pilot data
are never pooled. The only scoring look is the Mac `analysis.py merge` command after
the authorized attempts. Remote attempt summaries are sealed diagnostics, never read.

## Allocation and science identity

Science baseline: `f1904777aded79366703c94afbf5dc13155bb0ee`. Namespace remains
`bl-density-round14b-20261022`, allocation remains 492, workers 10, horizon 5000,
stride 200, H300 continuations and B8 checkpoint bytes unchanged. Operational code
runs from a published main commit carrying the new runner and requalified transport;
the manifest records both that pin and the original science baseline. Frozen science
helpers and decision functions are checked against the original source in targeted tests.

The ledger comes only from verified local exports of
`big-lever/exp/bl-density/density2-20261022-gatefix-f1904777/production-output-02`.
Its builder reads paths, sizes, SHA-256/complete sidecars and terminal export inventory;
it never parses episode JSON. It verifies local bytes against that inventory. Entries
are `{episode_id, sha256, size}` in a namespace-bound envelope. The runtime refuses
foreign namespaces, duplicate/unknown identities, malformed hashes and invalid sizes.
It constructs the full original roster plans before subtracting the ledger, preserving
index-based seeds and opponent rotations.

Arithmetic clarification of the ruling: 408 verified episodes leave **84 episodes**
(72 C-P1 plus twelve B70p/mixed episodes). The **85 missing files** include summary.json.
The actual verified inventory identifies B70p/mixed indices **22 and 25–35** as
missing; index 24 is already verified. The ruling's prose range 24–35 is a second
metadata discrepancy. Its mandatory allocation-minus-ledger rule determines the
workload: execute every missing ID, skip every verified ID, with no substitutions.
No 85th episode is added. Twelve fixed A replay episodes (indices 0–3 in each mix)
bring the completion workload to 96, not 97. Replay exports go exclusively under
`episodes-replay/`, never `episodes/`, and never enter the scoring N.

## Preconditions, attempts and caps

The telemetry correction must merge to main, its exact executable tree must have
fresh final-only qualification, and a real supervisor skip-ledger canary must pass
before completion dispatch. Canary uses the same entrypoint/source/input inventory,
a separate excluded 12-episode tiny namespace and a synthetic ledger skipping three
A units. PASS requires no skipped export/recomputation, all nine remaining exports
recovered with valid sidecars, matching family/package proofs, and verified release.
The new canary is 1 unit with 0.2 reserve; timeout 30 minutes (2x the existing
15-minute forecast and within the supervisor's hard 30-minute family gate and ruling's
2x 20-minute ceiling). Its producer work limit is 20 minutes.

At most two completion attempts, completion-01 and completion-02; each estimate
35 minutes, timeout **180 minutes**, unit cap **5.0**, reserve **1.0**, L4 only at
<=1.54 units/hour, first wall/unit cap wins. Prior series charge is conservatively
4.50 units, never subtracted again from an independently capped completion job.
Series worst case for the two completion jobs is 4.50 + 2*5.0 = 14.50 units;
each excluded new canary adds <=1.0 unit (at most two, **16.50 including both**). Reserve is included
in each cap, not added. Account assignment rates must be reviewed immediately
before dispatch. Preparation performs no Colab calls and is not live qualification.

Completion-02 is permitted only after verified mechanical failure, stop_verified true,
missing registered units, and no outcome read; its ledger grows mechanically from
verified files. Each attempt has a fresh manifest/output/receipt. It gets a fresh
same-input skip-ledger canary if changed ledger bytes invalidate the family proof.
After attempt two, the study is terminal: merge whatever is verified, report cell
attempted/complete counts (attempted counts executions, including partial work
and repeated attempts; complete counts unique registered episodes), apply the registered rule (possibly INSUFFICIENT), and
permit no third attempt, re-sizing, pilot pooling or outcome-dependent retry.

## Mac merge and CPU session confounding

`analysis.py merge` verifies sidecars, receipt/inventory bindings, namespace, frozen
allocation/index/seed identity, scientific binding and B8 identity before aggregation.
It rejects duplicate primary episode IDs across trees. The twelve replay identities
must match their originals exactly under `verify.semantic` (timing-free canonical
form): 12/12 or IDENTITY_FAIL, completion void. Original INCOMPLETE receipt remains
immutable; the new create-only completion summary cites every attempt receipt.

CPU calibration r = sum(replay cpu_seconds)/sum(original cpu_seconds), over those
fixed twelve A episodes. The original parent's overhead is unavailable because its
summary was lost. The completion's measured non-episode overhead per attempted episode
(including replays) is charged uniformly to all 492 registered episode slots. With two
attempts, measured overhead and attempted counts are summed before dividing. This
uniform per-episode charge cannot favor an arm. Partial work is exported separately
under episodes-partial/ or episodes-replay-partial/, never admitted to the resumable
ledger or scoring N, and its measured CPU remains charged to its own arm/mix.
The producer exports outcome-free
attempt metadata separately so the Mac never reads the sealed remote summary.
If a killed attempt loses its terminal accounting export, merge writes a terminal
INCOMPLETE artifact with verified counts and the missing-accounting reason and
issues no statistical admission. Missing replay exports produce IDENTITY_FAIL.
Neither condition authorizes a third attempt.

Evaluate the unchanged per-X-by-mix rule on raw CPU. Also report the evaluation with
completion-session episode CPU rescaled by 1/r, together with r, in every case.
If two completion sessions contribute records, each session requires its own fixed
twelve-replay identity/calibration check and its own r; rescale each record by its
source session's r and report each session. A missing/failed replay set voids that
completion. Never substitute the second session's ratio for the first. If
r is in [0.80,1.25], raw alone decides. Otherwise any cell whose evaluations disagree
becomes INSUFFICIENT_SESSION and can never be ADMITTED; a raw REJECTED may remain
REJECTED. Gameplay, realism, thresholds and all non-CPU metrics are unchanged.

The Mac CLI checks the clean published amendment against origin/main before opening
inputs. No strict gate, serving change, release, P1 or standing-policy implementation
is authorized by this package. Owner approval remains required for release.
