# Offline supervisor producer goldens

These twelve captures were made before changing continuation admission from
`e971d41f` (`training_resume.py` SHA-256
`c6aed498752109a8adadcc4ac4dfd84f902aaa96b955db28807a47f50df97473`).
The qualified `research.colab_cli.supervisor.Broker` and runtime source were not
edited. Every capture records producer source hashes, exact injections, raw
retained output and a complete file-size/SHA inventory in `provenance.json`.
No receipt was assembled by an admission fixture helper.

| Capture | Native result | Continuation |
| --- | --- | --- |
| clean-completion | PASS, real runtime scanner, zero child exit | Refuse: already completed |
| wall-cap | SUPERVISOR, whole-session wall cap | Admit |
| helper-deadline | SUPERVISOR, deadline before helper | Admit |
| cli-deadline | SUPERVISOR, deadline before CLI call | Admit |
| runaway-guard | ACCOUNTING, priced session unit cap | Admit |
| calibration-reserve | SUPERVISOR, study reserve during calibration | Admit |
| lease-renewal-timeout | RPC_EXEC_POLL_TIMEOUT, telemetry recovery deadline from `renew_lease` | Admit |
| rpc-exec-poll-timeout | RPC_EXEC_POLL_TIMEOUT, telemetry recovery deadline from `read_state` | Admit |
| runtime-timeout | CHILD_NONZERO_EXIT, native terminal `reason=timeout` | Admit |
| runtime-lease-expiry | CHILD_NONZERO_EXIT, native terminal `reason=lease_expired` | Admit |
| child-nonzero | CHILD_NONZERO_EXIT, exit 23, no reason | Refuse: unexplained failure |
| provider-loss-no-receipt | Fake provider sessions removed; broker process exits 91 before `finally` | Refuse ordinary admission: no receipt |

The subprocess fake is the existing `tests/fixtures/fake_colab.py`, configured by
the existing supervisor test helpers. Clock/deadline injections exercise actual
Broker/CLI guards. RPC faults are real `subprocess.TimeoutExpired` injections at
the selected fake-provider subprocess boundary; neither `execute` nor receipt
construction is replaced with canned errors. The runaway capture advances the
clock after the post-ready sessions call, between the loop wall check and the
native accounting check. The calibration reserve capture changes fake-provider
balance immediately after allocation.

The clean, runtime-timeout and runtime-lease-expiry captures also execute the
pinned `runtime.supervise` on a real CPU child. Runtime seams only relocate the
`/content` validation and replace credential/provider release operations. The
child runs the production `train.run` orchestration with the existing small
actor, critic, learner, task and preparation seams from
`test_training_production.py`. Proxy evaluation, pairing, serialization,
snapshot publication, runtime export scanning, state/terminal construction,
Broker transfer verification and final receipt construction are real producers.
For mechanical runtime stops, the child waits in the iteration-16 completion
callback after publishing its proxy-bearing immutable snapshot and before
writing a completed summary. These are format/transport qualification fixtures,
not scientific evidence or an actual B8 training qualification.

The provider-loss capture deliberately combines fake backend disappearance and
host loss. A surviving Broker normally writes a terminal receipt in `finally`;
absence of a receipt is not a new mechanical receipt class. This capture proves
that ordinary admission fails closed. A separate independently verified recovery
certificate is required by the recovery tooling; the capture creates none.

The original helper uses a private alternate Git index and creates temporary
commit objects. Their trees are byte-identical to the qualified baseline tree.
Each `pinned-commit.raw` retains the exact tiny commit object so tests can restore
it after Git garbage collection using `git hash-object -w -t commit`. This changes
no Git refs, checkout files or index. Admission still checks the actual manifest
pin against its staged source plan/archive. Paths inside captured receipts are
original capture paths; capture scratch has been deleted.

To reproduce into a new directory, use the project venv and thread limits:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  nice -n 15 /absolute/project/venv/bin/python \
  -m research.sandbox_a_resident.capture_supervisor_goldens /absolute/new/capture-dir
```

This tool selects only the explicit offline fake binary and private fake account
root. It never calls an installed Colab binary or accesses a real account. It
refuses existing capture destinations and removes its private scratch. Initial
failed discovery captures were retained separately under the task artifact
folder's `golden-discovery/`; none were substituted for successful cases.
