# I2 anonymous GitHub authority follow-up

Implementation baseline: `626350e7ed98b10703a2f43f3b14e273813bde57`, clean
`snake-dqn-f2-i2` worktree, branch `sol-f2-i2-fix5-20261024`.

## Behavior

- Prefer `GITHUB_TOKEN`, otherwise use the existing sanitized, bounded
  `gh auth token --hostname github.com` credential discovery. Missing CLI,
  timeout, unsuccessful discovery or empty output selects anonymous mode.
  Malformed nonempty tokens and decoding/configuration failures remain
  unavailable. No login, Git credential helper or direct Keychain operation.
- Anonymous captures first fetch the pinned repository metadata over the same
  verified HTTPS transport. Exact `full_name`, boolean `private: false` and
  `visibility: public` are required. Private or unverifiable repositories are
  unavailable. Anonymous requests never contain Authorization.
- Public status is fresh per capture, shared only by nested calls within that
  capture. The request ledger counts every attempted GET, including the public
  check, exact-commit ledger and sidecar fetches, ancestry comparisons and
  activity pagination. It records `auth_mode`, `repository_public`,
  `request_count`, `request_budget` and `rate_limit_remaining` in observations.
  Evidence exports snapshot the final accounting before serialization.
- Anonymous captures permit at most **50 requests**, preserving quota for other
  readers of the shared source IP. Token captures permit **20,110**, covering
  the previous complete 100-page activity scan and its ancestry/blob requests.
  Token requests retain their Authorization header and require no public probe.
  Neither mode retries requests. Exhausted quota or capture budget is unavailable.
  Anonymous responses also require a valid `X-RateLimit-Remaining`; both modes
  reject malformed or zero remaining values when provided. The reader does not
  wait for resets, bypass limits or turn an incomplete scan into evidence.
- `check-publication --ledger <canonical ledger>` is a read-only dry probe: local
  publication identity, server ledger and both sidecars at the exact commit,
  authoritative main ancestry and the complete push-activity scan. It writes no
  evidence receipt and never reads a sealed F2 statistic.
- Live probing exposed an existing HTTP EOF issue: `HTTPResponse.read1` may close
  the socket when the declared body is consumed. The reader now recognizes
  `response.isclosed()` before trying to set a timeout on the closed socket again.
  Response-size and deadline checks still apply to every consumed chunk.

Fixed-host/port TLS peer checks, explicit system trust roots, environment/config
override rejection, no proxies, no redirects, response/deadline bounds, fresh
GitHub Date/skew checks, server blob identities, both sidecars, ancestry and
complete pagination remain enforced. Publication unavailability still takes the
existing conservative path; it does not authorize a claim or a local-time fallback.

## Live read-only results

Commands use `./venv/bin/python`, nice 15, CPU and all six thread limits set to 1:
`OMP_NUM_THREADS`, `OMP_THREAD_LIMIT`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
`NUMBA_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`.

```bash
nice -n 15 ./venv/bin/python -m research.bl_residual.github_authority check-time
nice -n 15 ./venv/bin/python -m research.bl_residual.github_authority check-publication \
  --ledger research/bl_residual/ledger/f2-inputs-20261024.json
```

1. **check-time: exit 0**, `auth_mode=anonymous`, `repository_public=true`,
   **2 requests / 50 budget**, rate remaining **55**. GitHub Date
   `Sat, 24 Oct 2026 20:26:33 GMT`; main-ref request ID
   `D698:2A3DBF:62E095:6CE33B:6ADD1479`; `api.github.com:443`, verified TLS 1.3.
2. **publication dry check: exit 1, unavailable**, reason
   `GitHub capture request budget exhausted`, `auth_mode=anonymous`,
   `repository_public=true`, **50 requests / 50 budget**, rate remaining **5**.
   The full publication proof did not complete; no publication evidence receipt
   was written. No second scan was launched.

Before the EOF fix, one failed check-time capture and one diagnostic capture each
made **1 request**, leaving rate remaining 58 and 57 respectively. Diagnostics
printed only exception type/line and whitelisted nonsecret headers. Total live
GETs during this follow-up: **54 across four captures**. All prior failed captures
remain failures; only the successful check-time result above is trusted time.

## Verification and protected state

Regressions exercise anonymous success and fresh public checks; private/foreign/
unverifiable repositories; absent/malformed/exhausted rate headers; 200/403/429
exhaustion; nested request budgets; unchanged token Authorization; missing/empty/
failed credential discovery; sanitized decoding failures; anonymous transport,
clock and redirect refusal; closed-socket EOF; read-only publication probing;
and final request accounting in serialized publication evidence.

The independent read-only reviewer found a decoding-error sanitization gap in
the initial diff. That boundary now preserves `AuthorityUnavailable` and converts
unexpected value/decoding failures to a sanitized unavailable error. A zero-request
regression covers it. The final re-review found no remaining actionable findings.

Final serial regression run: **480 passed, 7 deselected in 148.04 seconds**.
The excluded tests invoke the sealed F2 runner. The final authority module
contributes **114 passing tests** to this run. An earlier focused run passed
111 tests before the EOF/serialization/decoding additions. Pytest used the
preloaded repository OpenMP path to avoid the macOS conftest relaunch wrapper:
one pytest process, at most one serial helper, no xdist workers.

```bash
nice -n 15 env DYLD_LIBRARY_PATH=/Users/josenunez/Projects/ml/snake-dqn/venv/lib/python3.12/site-packages/torch/lib \
  ./venv/bin/python -m pytest -q tests/test_bl_residual*.py \
  -k 'not sealed_runner and not success_writes and not no_payload_or_fallback and not unpublished_input_refused'
nice -n 15 ./venv/bin/python -m black --check --line-length 100 \
  research/bl_residual/github_authority.py tests/test_bl_residual_github_authority.py
nice -n 15 ./venv/bin/python -m isort --check-only --profile black \
  research/bl_residual/github_authority.py tests/test_bl_residual_github_authority.py
nice -n 15 ./venv/bin/python -m flake8 \
  research/bl_residual/github_authority.py tests/test_bl_residual_github_authority.py
git diff --check
```

Black, isort, flake8 and whitespace checks passed. Full F2 source bytes equal
`626350e7`; SHA256
`d36eaa778dceca49e46a69f8103d6ee745087ee5d0ea9526e5cab06c0a4142f8`.
The exact statistic block, `def selected_survival(` up to `def _validate_plan(`,
retains SHA256
`23c86e2b3086d55a2fc26bc6a9bc5be4d0b46dced9cf11683ea12a2c466c39d1`.

## Git handoff

Committed in `sol-f2-i2-fix5-20261024` and rebased cleanly onto
`origin/main` at `c5cb448184112ac8f4e181ae236ff2a597bf35d6`. The intervening
main commit changes only `research/gpu_sim_spike/`; this follow-up's three
files and all F2 source bytes are unchanged by the rebase. Working tree clean
after commit.

**Push blocked by authentication under the owner's no-credential-access rule.**
Anonymous fetch succeeded with credential helpers and prompts disabled. The
fast-forward HTTPS push with those same restrictions failed with
`could not read Username for 'https://github.com': terminal prompts disabled`.
An SSH read-only route probe, with no custom configuration, batch mode,
`UseKeychain=no` and strict host-key checking, failed with
`Permission denied (publickey)`. The effective existing SSH configuration has
no alternate host/agent/identity route. No login, helper/Keychain access,
credential changes, remote rewrite or force push was attempted. The local
commit is ready for an authenticated fast-forward push; it is not on main yet.

Untouched: live D1024 factory; tmux; `exp/mac-lease.json*`;
`t34-round0-study-01/`; `research/bl_labels/*`; `research/colab_cli/*`; released
checkpoints; serving pins; existing hash-bound artifacts. No F2 `run-sealed`,
R-A execution, protected payload read, Colab action or serving change occurred.
