# T16 Colab long-run recovery notebook

[`notebooks/colab_long_run.ipynb`](../../notebooks/colab_long_run.ipynb) is an
owner-run recovery harness for an already pre-registered M3-C training recipe.  It is
separate from the bounded throughput smoke notebook.  It deliberately ships with no
training recipe, run name, commit, or cadence selected: fill the clearly marked
**TO BE PRE-REGISTERED** parameter block only after those values have been registered.

The notebook mounts Drive, creates a clean Python 3.12 environment, and selects the
PyTorch CUDA wheel from the assigned GPU: `cu128` for compute capability 10.0 or
higher, otherwise `cu126`.  Its kernel interpreter is only an orchestrator; training
runs in that isolated environment.

For each `RUN_NAME`, snapshots live in:

```
<Drive root>/<RUN_NAME>/full_state/
```

Before starting `train_m3c.py`, the notebook calls
`discover_snapshots(..., verify=True)`.  This verifies hashes and ignores incomplete
temporary directories.  When a complete snapshot exists, the command receives
`--resume-full-state <that directory>`; the trainer selects its newest valid snapshot.
The resume event and whether replay was included are recorded by the trainer summary.

The command always supplies the async-checkpoint contract:

```
--full-state-every N
--full-state-replay none|full
--ckpt-local-dir /content/...
--ckpt-durable-dir /content/drive/...
--resume-full-state /content/drive/...       # only when a verified snapshot exists
```

Use `--full-state-replay none` when a restart can use the registered replay warm-up.
That mode is explicitly marked `resumed-without-replay` by the trainer.  `full` stores
the replay ring arrays.  Its size can be substantial and scales with replay capacity,
stored observation/action fields, and dtype; inspect each snapshot manifest's
`files[].size_bytes` before choosing a Drive quota or cadence.  The notebook has no
Drive copy in its learner process; the trainer writes locally first and transfers in
the background.

The notebook never releases a runtime after a failed/incomplete command.  If
`RELEASE_AFTER_SUCCESS` remains enabled, it releases only after the trainer exits
successfully and `discover_snapshots(..., verify=True)` confirms a durable, complete
snapshot.  A failed durable confirmation leaves the runtime assigned for diagnosis or
manual recovery.
