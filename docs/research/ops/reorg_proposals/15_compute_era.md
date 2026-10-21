# Reorg proposals — scope 15 (compute-era packages)

No files moved; proposals only (research/** is frozen for ratified-study hashing).

1. **S1 verdict pointer (top priority).** `research/s1_mature_20261018/README.md` and
   `FINAL_EXPORT_STATUS_20261018.md` still say "ADMISSION REFUSED / NO DISPATCH"; the study ran and
   was KILLed (CAMPAIGN_LOG.md). When the package can be edited, add a one-line OUTCOME banner
   linking CAMPAIGN_LOG; until then the catalog is the pointer.
2. **Broken doc link.** `research/runpod_fanout/runner.py` docstring cites
   `docs/research/runpod_fanout_runner_2026-10-03.md`, which does not exist. Recover it from git
   history or drop the reference.
3. **RunPod API version (owner/planner to confirm the v1 sunset date; not recorded in repo files).** `runpod_fanout/rp_client.py` callers (incl.
   `sequential_strict_template`) should be marked "unmaintained after v1 sunset" or migrated.
4. **Naming.** The Drive root is called `t16_colab` but is shared by colab_queue, Q1 and T23; consider
   documenting it as the generic legacy Drive root in docs/colab-cli.md.
5. **Superseded receipts.** S1 holds 5 ADMISSION_STATUS_* and 3 TRAIN_PREFLIGHT_* variants; a
   future `research/s1_mature_20261018/RECEIPTS_INDEX.md` (or a catalog section) should mark which
   one is final vs historical.
6. **Archive candidate.** Once Q1 is fully settled, `research/colab_queue` + the worker notebooks
   can be tagged legacy (e.g. moved under `research/_legacy/` together with their importers), not
   deleted.
