# RunPod runtime self-delete credential

Updated 2026-10-17.

`research/redesign_m3c_20261008/gpu_run_v2.py` and
`research/perf_podbench_20261013/pod_bench.py` now support an owner-created
**RESTRICTED RunPod API key with pods read/write permissions only**. The owner
stores it as a generic password in macOS Keychain with service
`runpod-selfdelete` and account `runpod` (use Keychain Access to enter the value;
do not put it in a shell command, shell history, environment variable, or file).
This is separate from the Mac account-management key under `runpod-api`.

After authenticated agent health succeeds and before any uploads or jobs, the
Mac runner reads the optional item through a captured Keychain stdout pipe. It
sends the value in the JSON body of authenticated `POST /self-delete-key` through
the existing RunPod HTTPS proxy with `X-M3-Token`. TLS verification is required
and credential redirects are refused. The key is never included in the pod
create body, pod environment, start command, template, argv, archive, or local
state. The agent holds it only in process memory; jobs do not inherit it.

The agent immediately probes `GET https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID`
with the supplied key. A **200 remains required** unless the runner has an
explicit `--accept-no-self-delete` waiver. The same in-memory key is used for
`DELETE` when the deadline or dead-man timer fires. The endpoint accepts only a
credential, not a target pod ID: lifecycle calls target `RUNPOD_POD_ID`. It returns
only armed/probe status, never the credential or provider response body. Agent
logs redact the key, transport errors are sanitized, and the Mac refuses agent
responses containing it before saving logs or pulling results.

If the Keychain item is absent, behavior is unchanged: probe with the injected
key and abort on failure unless the waiver is present. Keychain access errors or
credential delivery errors fail closed before uploads/jobs. An agent restart
loses the owner key; existing Mac guards and watchdog remain necessary. No API
migration is included here: the probe and DELETE still use the existing v1 URLs.

Validation uses fake credentials and mocked provider requests only; pytest
disables real Keychain reads by default. Offline tests cover authenticated
delivery, probe/delete selection, deadline/dead-man behavior, missing-key
fallback, child environment isolation, and absence from pod bodies, argv, logs,
state, ledgers, receipts, and pulled results. This change has not been verified
on a live pod and does not authorize creating one.
