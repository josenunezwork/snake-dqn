# E7 operational seed retirement (before scientific data)

During the current-E8 compatibility timing check, operational E6 continuations
on burned preflight source roots used 1102882751176 and 1100083338059 as future
environment RNG seeds. These two identities belong to the audited exploration
pool, rather than its preflight pool. Scientific source worlds were not played;
all outcome rows from the check were discarded and only timings were exported.
Evidence: exp/bl-e6e7/production-preflight-v2-e8-check/e6_current_e8_timing.json.

Retire both identities from EVERY scientific role prospectively. E7 manifests
select the first 3072 remaining exploration identities in the frozen audited
order; 3998 remain, so N and all design/decision rules stay unchanged. The
original SEED_AUDIT.json bytes and audit allocation stay immutable. A separate
runtime/admission exclusion enforces these operational burns. This amendment
is committed before any scientific collection or dispatch. It does not
reinterpret the operational timings as scientific evidence or authorize retries.
