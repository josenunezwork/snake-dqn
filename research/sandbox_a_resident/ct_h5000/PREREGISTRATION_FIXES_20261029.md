# C/T H5000 prospective operational amendment — packet revision 02

Written before production data or dispatch. Owner instruction, 2026-10-29:
fix the independent Argon stand-in ACCEPT-WITH-FIXES review, reseal, test and
publish the branch; do not dispatch or merge. This sibling supersedes only the
operational clauses named below in PREREGISTRATION.md. Its frozen bytes and
packet revision 01 remain unchanged. No new science, N, quotas, support bars,
seed allocation, sampling or collection policy is introduced.

1. Collection base **6,600 seconds (110 minutes)**, effective **13,200 seconds
   (220 minutes)**. The earlier 98–108-minute natural-collection forecast excluded
   uncertain extra root admissions, growing ring/body work and assembly. 110 minutes
   is the base planning forecast, not measured H5000 timing. The effective 2x guard
   supplies margin. Whole-session base **180 minutes**, effective **360 minutes**,
   including setup, collection, assembly, exports and cleanup. Neither measured
   rates nor N are doubled. H100 timing remains unmeasured; no speedup is assumed.
   Expected end-to-end wall remains an uncertain **roughly 2–3 hours**; this is not
   a guarantee. Informational timeout-priced fields: H100 72 units, A100 48 units;
   the standing no-unit-budget policy applies.
2. Clock starts before collector initialization. No further natural tick or
   candidate capture is admitted after collection expiry. If all H5000 ticks have
   completed, deterministic bank assembly and support audit may finish under the
   outer session guard; a fully collected bank is not discarded because assembly
   crosses 13,200 seconds. Export natural-collection and assembly durations and
   the cap-crossing flag. An unfinished bank remains INCOMPLETE_COLLECTION with
   support not evaluated. No extra worlds/ticks or outcome-driven extension.
3. Logging-only immutable gzip JSONL exports record every observed hero death:
   world, partition, mix, life, pre-step length, tick, below-length rejection,
   missing same-life lookbacks, candidate-pool rejection and candidate admission.
   Every tick records living-length histograms separately per mix with the
   original six length bins. Preserve these plus replay diagnostics on terminal
   cap, support, signal and ordinary fault exits. Journals have a combined 96 MiB
   compressed bound and SHA/completion-last publication. An abrupt hard kill can
   still prevent sealing; the existing watchdog evidence remains separate.
   Logs do not select data or authorize an interim scientific look.
4. The lookback ring now retains the actual batched float32 Q alongside its
   recorded action. A target batch-1/batch-1024 action mismatch is accepted only
   if both actions are legal greedy choices in their respective Q vectors and
   their pairwise gaps are <=4 float32 ULP in **both** vectors. Export both Q
   vectors, legal mask, actions, gaps, tolerances, world/life/tick and qualification.
   A non-finite, illegal or unexplained mismatch still refuses. The returned root
   carries the original batched Q and action; replay dynamics retain recorded
   actions. No epsilon, altered collection decision, tie-break policy or discarded
   root is introduced. Startup mechanics/topology checks remain mandatory; this
   handles explained ties rather than claiming batch-size equality on CUDA.
5. Stage cap **896 MiB**, recursive export cap **1 GiB**, supervisor transfer bound
   **1,536 MiB** (within the existing 2 GiB transport maximum). For 2,560 mature
   roots, conservatively project `98.5e6 + 2560*(28*1024 + 6*L*4)` bytes: retain the
   entire predecessor stage, add 28 KiB per root, then all six raw int16 x/y bodies
   without compression credit. Evaluate L=600,1700,10400 in export-projection.json.
   This intentionally double-counts prior root/body bytes as headroom. Mature
   topology, anchor compression and serializer overhead remain unmeasured; the
   projection is not a worst-case memory guarantee. Existing 512 MiB root-store
   and 1.5 GiB in-memory anchor caps remain; all overflows refuse without truncation.
   The recursive bound leaves 128 MiB above the stage cap for <=96 MiB journals,
   summary, progress and watchdog seals. No success is preclaimed from sizing.

The unchanged namespace is still unused by a production run. The revision-02
seal binds this amendment, runtime closure, control, ledger, audit and exact
H100/A100 variants. Any mechanical completion remains separately prepared and
manual; scientific shortfall or collection cap is not a retry trigger.
