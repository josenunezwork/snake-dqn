# GPU-Resident Long-Horizon PPO on A100 (`train01`, Packets 05–16)

**Dates:** 2026-10-27 to 2026-10-30
**Verdict:** Closed with zero strict-gate nominations after `384` GPU-resident PPO updates (`100,663,296` actor rows) from `b8-s3` peaked at `+14.23` and finished at `+8.09` equal-mix H5000 mass against the `>= +40` material-gain bar (insufficient evidence to advance, not a proof of zero benefit).

## What was tried
- Fine-tuned a fully trainable `b8-s3`-initialized `ego2s` actor with Proximal Policy Optimization (`PPO`), a frozen-B8 Kullback-Leibler (`KL`) anchor, and a separate frame-clocked PopArt return-normalized critic (`E360` complete-life preparation: `270` fit + `90` holdout worlds).
- Ran `384` updates (`E1024` GPU-resident worlds × `T256` / `256`-step rollouts = `100,663,296` actor rows across equal `scripted`/`frozen`/`mixed` opponent mixes) on an A100 GPU under Resume-by-Default, evaluating checkpoints on a `96`-world-per-mix paired H5000 development proxy against a `>= +40` pooled mass bar.
- Repaired four engineering blockers across packets `05`–`15`: NumPy slice-view memory retention, `30`-s `git fetch` staging RPC timeouts, first-update `+inf` KL divergence at calibrated temperature `T = 0.008439731318504158` (`T ≈ 0.00844`), and `300 s` upload timeouts on `1.33 GB` resume snapshots.

## What happened
- **Critic prep & trajectory (`COMPLETE`):** Critic preparation passed (`held_out_late_ev = 0.4404 >= 0.10`), and packet `15` completed all `384` updates (`100,663,296` actor rows in `13,745.224 s` on completion attempt c2).
- **Gameplay screen (`Closed, no nomination`):** Endpoint (update `384`) equal-mix paired H5000 mass vs bare `b8-s3` was `+8.09` (`CI95 [-18.66, +34.85]`; `scripted` `+26.32`, `frozen` `+2.85`, `mixed` `-4.90`), peaking at `+14.23` (`[-11.86, +40.32]`; `+41.16`/`+4.60`/`-3.06`) at update `320` after `-0.14`, `-7.99`, `-2.10`, `-3.81` at updates `64`/`128`/`192`/`256`.
- **Decision hedge:** Both endpoint and peak missed `>= +40`, so zero checkpoints crossed to the Mac strict gate; per closeout, this single-trajectory screen is insufficient evidence to advance, not a proof of zero benefit.
- **Engineering fixes:** Copying `12,443 B` slices cut the `160,000`-row buffer upper bound from `47.78 GB` (`298,632 B` E24 views) to `1.99 GB`; `background-sparse-v1` (`191` files, `925,149 B`) installed source in `0.025–0.035 s` vs `32.379 s` `git fetch`; float64 `log_softmax` stopped `+inf` KL; and `xz -6` shrank `1,327,620,931 B` (`1.33 GB`) snapshots `~151x` to `8,804,064 B` (`8.8 MB`).

## Why
Dividing a B8 advantage gap (`≈ 0.877`) by sharp temperature `T ≈ 0.00844` pushed legal float32 log-probabilities below `-103.97` (`<= -103.972084`), where float32 `exp()` underflows to zero and categorical KL returns `+inf`. For the `+8.09` plateau (`mixed` `-4.90`), deeper learning saturation had its cause not established in this single-trajectory screen.

## What to reuse / what not to repeat
- Compute categorical `log_softmax`, KL divergence, and entropy in `float64` whenever temperature scaling (`T ≈ 0.00844`) can push legal logits past float32 `exp()` underflow (`-103.97`).
- Clone observation rows before buffering so slices do not pin full batch tensors (`47.78 GB` vs `1.99 GB`), and compress multi-gigabyte resume snapshots with `xz -6` (`1.33 GB → 8.8 MB`) plus sparse source archives to stay under `300 s` / `30`-s staging caps.
- Do not continue single-seed frozen-input `b8-s3` PPO trajectories expecting large `mixed`-field gains after `100,663,296` transitions plateaued near `+8.09`.

## Full record
Original: `git show 90b3db94:docs/research/history/train01.md` | Primary sources: [TRAIN01.md](../../../research/sandbox_a_resident/TRAIN01.md), [TRAIN01_14.md](../../../research/sandbox_a_resident/TRAIN01_14.md), [TRAIN01_15.md](../../../research/sandbox_a_resident/TRAIN01_15.md)
