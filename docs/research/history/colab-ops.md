# Colab CLI Supervisor, Legacy Queue Worker, RunPod/H100 Operations, and Small-Bench Fast-Path

**Dates:** 2026-10-02 to 2026-10-31
**Verdict:** Qualified and live (`research/colab_cli/` + `tools/ops/`, `ACCOUNT_ASSIGNMENT_CAP = 6`, `ACCOUNT_L4_CAP = 3`): retired RunPod strict gates after `~1` of `50` (`1/54`) `H10000` float32 Q-tie flips, lifted A100 self-play to `15,927 tr/s` (`3.0×` Mac) and L4 `N=3` to `7,625 tr/s` (`T17`/`T22`/`T24`), and repaired `UPLOAD_CAP` staging via `metadata_archive.py` and kernel-UUID supervision.

## What was tried
- Tested RunPod `cpu3c` x86 vs Mac M5 Pro consistency (`28` H5000 episodes + `diag-divergence`) and `runpod_fanout`, then froze legacy Drive queue `research/colab_queue` (`2026-10-18`).
- Built `uv` Python `3.12` Colab bundles (`T16`/`T16b`), added multi-thread learner gather (`--gather-threads 3/4 --no-obs-pack`, `T17`/`T17b`), and wired Numba `v8` + shared body-release maps into static-pool actors (`T22`/`T24`) across Mac MPS, L4 (`12` vCPU), and A100.
- Built and qualified the Mac OAuth2 supervisor around `google-colab-cli 0.7.4` (`research/colab_cli`, `2`-poll grace & carry-over, kernel-UUID readiness, multi-tree qualification registry, and `ready_queue.py` auto-retry).
- Diagnosed `4` Oct 23–24 `UPLOAD_CAP` staging failures (`colab_upload_cap_diagnosis_2026-10-24.md`), built `research/metadata_archive.py` (`<= 2 MiB` tar), and audited `249.27 GiB` across `250` `snake-dqn-artifacts/` folders.

## What happened
- **RunPod retired for strict gates:** `cpu3c` ran `3.66×` slower (`1,813 s` vs `508 s`) with `4.8e-5`–`8.4e-5` (`2,176–3,072` ULP) Q-diffs, and `diag-divergence` proved `arm64` vs `x86_64` float32 Q ties flip in **`~1` of `50` H10000 episodes (`1/54` in S2)**.
- **A100 learner & host-gather optimization (`T17b`):** Cut `B=256` gather `3.024 -> 1.214 ms` and lifted A100 self-play `11,341 -> 15,927 tr/s` (`62.2 upd/s`, `3.0×` Mac MPS `5,459 tr/s`), missing the `84 upd/s` (`4×`) bar because isolated fp32 learner updates topped out at `78 upd/s`.
- **Static-pool speedups (`T22`+`T24`) & L4 economics:** Cut frozen-pool CPU `-30.2%` (`41.109 -> 28.687 ms`) and `L=1200` `v8` `-26.5%` (`22.155 -> 16.280 ms`), giving `12`-vCPU L4 **`7,625 tr/s` (`1.46×` Mac, `~3.8k` vs A100 `~2.1k tr/s` per unit-hour)**; `S1` L4 runs closed `KILL` (`-12.5` / `-14.9` mass).
- **`UPLOAD_CAP` staging & disk audit:** In P1-04 (`28` files, `74,138 B` in `113.760 s`), **`22` tiny sidecars (`1.6%` of bytes) took `80.6%` (`91.679 s`)** of transfer time alongside slow kernel/control RPCs (`0` PUT failures), fixed by single-tar `metadata_archive.py` and `STAGING_REPAIR_20261024.md`. Disk audit flagged `34` copies of one `4.06 GB` demo bundle (`~138 GB` of `249.27 GiB`).

## Why
Host Python/RPC overhead—single-thread replay gather, CPU `v8` wrappers, and per-file Colab kernel/Contents RPCs after `google-colab-cli 0.7.4` dropped its keep-alive daemon—dominated Colab runtimes rather than GPU FLOPs.

## What to reuse / what not to repeat
- Dispatch all Colab work via `research/colab_cli/` + `tools/ops/ready_queue.py` (`ACCOUNT_ASSIGNMENT_CAP = 6`, `L4 <= 3`), using L4 (`~3.8k tr/s` per unit-h) first for `12`-vCPU jobs and tar-bundling small sidecars (`metadata_archive.py`).
- Keep strict release gates on Mac (`1/54` x86 float32 tie flips), and never upload dozens of tiny metadata sidecars individually or retain duplicate `4.06 GB` staging bundles.

## Full record
Original: `git show 90b3db94:docs/research/history/colab-ops.md` | Primary sources: [colab-cli.md](../../colab-cli.md), [QUALIFICATION.md](../../../research/colab_cli/QUALIFICATION.md), [README.md](../../../tools/ops/README.md)
