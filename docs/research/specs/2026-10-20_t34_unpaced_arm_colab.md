# T34 retention — move the unpaced descriptive arm to Colab (operator ruling, pre-data, 2026-10-20 ~14:15 PT)

Standing owner rule (docs/STATUS.md, 2026-10-20): use Colab credits whenever they speed up work. The fixes2 cost
smoke (research/t34_pathb/VERIFICATION_RETENTION_FIXES2.md) projects 38.45 h vs the 40 h cap, of which 17.72 h is the
UNPACED descriptive arm. That arm has no timing role and no decision role (descriptive only).
Ruling:
1. The unpaced descriptive arm runs on Colab lanes (CPU-bound; choose L4 or CPU-only runtime by measured cost per
   world — record nproc/RAM; <=3 concurrent lanes incl. any other Colab job), sharded over the SAME registered banks
   (terminal worlds + product sessions), through research/colab_cli (qualified bytes only — if this needs any
   colab_cli change, STOP and report: acceptance would be required) with public inputs (s1-inputs-20261019),
   create-only exports per shard, per-world hashes. Its results are descriptive; cross-platform float-tie
   divergence vs the Mac is expected (~1/50 episodes) — report per-world divergence of the shared bare prefix.
2. The Mac registered run executes only bare + paced arms + timing gate; caps unchanged (40 h / 60 CPU-h Mac,
   pre-check 8 h). Add a Colab unit cap for the unpaced arm (compute from the smoke: per-world cost x worlds,
   priced at 2.0 u/h for L4 or the measured CPU rate, + 20% reserve) to the prereg and registration.
3. The analyzer reads the unpaced exports as a separate descriptive input; missing/failed unpaced shards never
   change GO/KILL/INCONCLUSIVE (they are reported as missing descriptive rows).
4. Update PREREGISTRATION_RETENTION_V2.md (pre-data amendment with this reason), registration draft, README
   (exact Colab manifest-build + dispatch commands for the operator), tests.
Rules: Mac only; no live Colab/tmux; <=2 CPU threads; no efficacy rows; Mac preflight of the new Colab entrypoint
(2 worlds, short frames) per colab-jobs rules; commit/push often; merge to main; report commit SHA.
