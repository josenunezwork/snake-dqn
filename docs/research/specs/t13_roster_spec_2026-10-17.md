# Spec: T13 product-roster measurement (descriptive, not a gate), 2026-10-17

Read AGENTS.md, `docs/research/advice/2026-10-17_what_next.md` item 2 (binding), and the T12 qualification harness
(`research/t12_b8_serving_20261017/`, Watch mode via the real served session path). Implementer: Codex.
Using the T12 Watch harness (real served session, H5000, 12 snakes as served), 25 episodes each, the SAME seeds across
rosters (seed-disjoint from every bank in the T7/T12 freshness audits): (a) all b8-s3 (released default), (b) all
frp3-s12 + v8 (SNAKE_SERVE_CHECKPOINT=frp3-s12), (c) all champion + v8 (=champion), (d) mixed: b8-s3 hero with frp3-s12
+ v8 opponents (add a minimal default-off harness/config hook for a mixed served roster if none exists; serving defaults
unchanged). Report per roster: hero deaths per 1,000 decisions, hero length distribution (median, p95, max), H5000 mass,
kills made and taken, death-cause split (head-on, body, wall, self), boost rate, food eaten per 1,000 decisions; and the
same aggregated over all snakes in the arena. Pairwise differences with bootstrap CIs by seed. Doc
`docs/research/t13_product_roster_2026-10-17.md` with a plain-language summary: did Watch get better or worse for a viewer,
and is roster composition a lever. CPU, at most 6 processes. Never touch tmux sessions or processes you did not start.
Git: branch t13-20261017; commit often; after each commit `git fetch origin && git merge origin/main`,
`git push origin HEAD:main`, `git push origin t13-20261017`. Final message: the roster table, the summary, sha.
