# S1 Mature-Schedule League vs Hero Fine-Tuning (`s1-mature`)

**Dates:** 2026-10-18 to 2026-10-19
**Verdict:** Both mature-schedule fine-tuning arms closed `KILL` on one Mac evaluation look (`-12.5` and `-14.9` pooled H5000 mass vs bare `b8-s3` across `5` paired seeds × `8M` decisions, vs `>= 20` and `LB90 > 0` GO bars; failure closes these packages only and cannot isolate schedule, collection, or optimizer-reset causality).

## What was tried
- Trained two `5`-seed × `8,000,000`-decision mature-schedule (`gamma = .998`, `epsilon = .01`, `anchor = .1` from step zero, fresh Adam, empty replay) fine-tunes from `b8-s3` on Colab L4: `N3` (`6` actors / `3` static-pool learner seats) and `N1` (`3` actors / `1` terminal-hero seat).
- Repaired a `48`-decision actor-ingestion barrier overshoot (`12,048` vs `12,000` decisions, bound `47`) in `train_m3c.py` by halting ingestion while a barrier is pending and returning drained packets to per-actor FIFOs (`0` overshoot).
- Audited full-capacity replay snapshot serialization (`500,000` replay + `300,000` label rows at `12,449 B`/row) and switched to uninterrupted final-only weight exports (`final.pth` + `target.pth`, `64 MiB` per-attempt cap) before running `10` L4 jobs and one `640`-world (`1,920`-episode) Mac look.

## What happened
- **Gameplay evaluation (`KILL` both arms):** While losing much less than T23's `-80.8`, both `N3` and `N1` regressed against bare `b8-s3` at **`-12.5` and `-14.9` pooled H5000 mass** (`64` scripted + `32` frozen + `32` mixed worlds/seed; audit PASS), missing `LB90(G) > 0` and mean `G >= 20`.
- **Snapshot size vs final-only export:** Full `torch.save` arrays measured **`11,178,149,683 B` (`N3`)** and **`10,490,632,471 B` (`N1`)**/snapshot (`867,602,770,220 B` across `4` snapshots × `10` jobs), whereas uninterrupted `30/60` min (`N3`) and `68/136` min (`N1`) runs exporting `final.pth` (`6,566,741 B`) + `target.pth` (`6,562,677 B`) shrank the archive to **`12,180,957 B` (`11.62 MiB`)**.
- **Preflight Mac eval timing:** Evaluating `10` disjoint H5000 worlds took `67.21 s` summed episode time (`2` 1-thread workers) / `63.68 s` serial (`154 s` command wall) with identical result bytes.

## Why
Per `PREREGISTRATION.md`, `S1` was designed without a restarted-schedule or warm-buffer control, so **cause was not established** between mature schedule, `N3` vs `N1` collection, fresh Adam/empty replay reset, or their interaction.

## What to reuse / what not to repeat
- Drain in-flight actor packets into learner-local FIFOs while a step barrier is pending so asynchronous multi-actor trainers hit exact decision targets (`0` overshoot).
- Price full replay-buffer checkpoint sizes (`~10.5–11.2 GB`/snapshot) before dispatch; for short (`30–68 min`) L4 runs under `~1 h`, export compact terminal weights (`~12.2 MB`) rather than hundreds of gigabytes of raw replay arrays.
- Do not repeat `8M`-step fresh-Adam / empty-replay DQN fine-tunes from `b8-s3` under static `N3` or `N1` pools (`-12.5` / `-14.9` mass).

## Full record
Original: `git show 90b3db94:docs/research/history/s1-mature.md` | Primary sources: [s1_mature_20261018/README.md](../../../research/s1_mature_20261018/README.md), [s1_mature_20261018/PREREGISTRATION.md](../../../research/s1_mature_20261018/PREREGISTRATION.md), [CAMPAIGN_LOG.md](../final_run_20261019/CAMPAIGN_LOG.md)
