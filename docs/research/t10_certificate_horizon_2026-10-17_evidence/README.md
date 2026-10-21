# T10 certificate and horizon evidence (2026-10-17)

Development diagnostics only. This is not policy-training, rollout, gate or serving evidence.

**Cited by:** [t10_certificate_horizon_2026-10-17.md](../t10_certificate_horizon_2026-10-17.md).

**What it supports:**

1. The certificate verdict. Dense lookahead and B8 both hold the tail-by-L flag on almost every
   eligible decision (99.970% vs 99.915%). Holding the flag more often therefore does not explain
   lookahead's better H5000 survival.
2. The horizon verdict. Death-prediction AUC falls from D30 to D300, and how fast it falls depends
   on the inputs. The decline is not monotonic, so the data show no universal ceiling near 0.77.

| File | Contents |
| --- | --- |
| `certificate.json` | Full-cohort certificate result (run status `PASS`), summary, episode and manifest SHA-256. |
| `certificate-manifest.json` | Run manifest: CPU, 1 process / 1 thread, input source hashes; authority "diagnostic only". |
| `horizon.json` | Combined horizon-stratified fits (`t10-horizon-combined-v1`), data and fold hashes, 2,000 whole-world bootstrap draws. |
| `audit.json` | Saved-artifact audit of the certificate and horizon outputs; status `PASS`. |
| `verification.json` | Record of pytest, flake8, `git diff --check`, source hashes, environment and review; status `PASS`. |
