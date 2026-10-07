# Sequential Phase R (`sequential-phase-r-obf-hk-v1`), opt-in for future Tier-1 screens

* **Design:** [`docs/research/sequential_phase_r_design_2026-10-07.md`](../../docs/research/sequential_phase_r_design_2026-10-07.md).
* **Governance draft:** [`docs/research/governance_amendment_sequential_phase_r_2026-10-07.md`](../../docs/research/governance_amendment_sequential_phase_r_2026-10-07.md).

**Status: code, tests and simulation only.** Nothing here has played a game. No existing study
(FRP-v2 ... v5-S2) imports or is changed by it.

## Files

| File | Role |
|---|---|
| `src/evaluation/sequential_phase_r.py` | Pure plan, statistics and decision. Builds the plan and its boundaries (`make_plan`), validates the declarative rule spec (`validate_rule`), and decides one look with replay (`sequential_phase_r_decision`). |
| `example_rules.py` | FRP-v4-like and FRP-v5-H/S2-like rule specs, with the optional NO_GO. |
| `hooks.py` | Look schedule (`look_ranges`, `look_units`), records to look deltas (`look_deltas`), and `analyse_look`, which writes the receipt. |
| `receipts.py` | Create-only `plan.json` and `looks/look-<k>.json`, plus `look_gate` (the shard-start binding). |
| `audit.py` | Independent audit. Standard library only, run with `python -I`. |
| `simulate.py`, `oc_table.py` | Vectorised operating-characteristics simulation and its Markdown tables. A test replays replicates through the pure decision function. |
| `variance_pool.py`, `variance_pool_20261007.json` | Resampled per-world paired-delta spread from FRP-v2 Phase 2, FRP-v3 Phase R and FRP-v4 Phase R. |
| `oc_results_20261007.json` | The OC run reported in the design doc, section 9. |

## How a future study opts in

1. **Pre-register the rule spec.** Put the rule spec JSON in the pre-registration, and pin its
   sha256 and `make_plan(N, RULE).sha256()` by test. If the study wants NO_GO, say so and give
   G. Mark the protective clauses.
2. **Freeze the plan before any Phase R shard.** Call
   `receipts.write_plan(seq_root, plan.as_dict(), banks, {...commit, ratification, prereg sha...})`.
3. **Run each look k:**
   1. Call `binding = receipts.look_gate(seq_root, k)` and write `binding` into each look-k
      shard's start marker.
   2. Run `hooks.look_units(plan, k, banks, heroes, controls)`, which covers only the world
      indexes `[n_{k-1}, n_k)`. The prefix controls run in look 0.
   3. Merge as the study already does.
   4. Call `hooks.analyse_look(seq_root, plan, entries, k, {"prefix_controls": <bool>})`.
   5. Stop unless the receipt says `CONTINUE`.
4. **Audit.** Run `python -I research/sequential_phase_r/audit.py --root <seq_root>
   --record-dirs <every shard dir>`. It must report `PASS`. `UNCLOSED` or `FAIL` makes the study
   INVALID / INCOMPLETE.
5. **Report.** The study reports the last receipt's status. After an early stop:
   * the point estimates are biased upward and are descriptive only;
   * the final-only labels are not resolved after a GO, KILL or NO_GO stop.

## Pod runner compatibility (`research/pod_phaser`, branch `frp-v5h-phaser-rp`; not modified here)

* **Blocks per look.** A sequential study's stage list becomes `phaser-L<k>-<lo>-<hi>` blocks
  per look, built from that look's units only. Shards are world groups inside one look
  (`group = L<k>|s<seed>|<mix>|<batch start>`), so the per-world platform rule holds: every hero
  of a world is on one pod.
* **Launch gate.** `podrun` must refuse a look-k block until `receipts.look_gate(seq_root, k)`
  succeeds on the Mac. That means receipt k-1 exists and says CONTINUE. `podrun` then passes the
  binding (plan sha256, previous receipt sha256, index range) into the task. The task agent
  writes it into each shard's start marker, next to the commit and ratification shas it already
  binds.
* **Look k+1 starts only after:**
  * look k's shards are pulled (`podrun fetch`);
  * the records are merged on the Mac;
  * `analyse_look` has run.

  A look's wall time therefore includes one pull and merge round trip.
* **Sizing.** A block's runtime (the identity verdict's slowest unit, the 4x rule) is sized per
  look. Each block is about a third of the fixed design's work.
* **Recovery.** The rule is unchanged: rerun the whole shard of the same look, at the same
  commit. A recovery attempt must carry the same binding. A look whose shards are incomplete is
  never analysed; the study is then INCOMPLETE.
* **Records left on the volume.** Never pull records from beyond a stopping look, for example
  from a block launched by mistake. The audit FAILs if a `--record-dirs` directory holds one.

## Running the simulation (one thread, numpy)

```
OMP_NUM_THREADS=1 ./venv/bin/python -m research.sequential_phase_r.simulate --reps 10000 \
    --out research/sequential_phase_r/oc_results_<date>.json
./venv/bin/python -m research.sequential_phase_r.oc_table research/sequential_phase_r/oc_results_<date>.json
```
