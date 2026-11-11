# Standing A100 Development Sandbox (`B1`/`B2`/`B3` Probes) and Prospective Child Collector v2

**Dates:** 2026-10-26 to 2026-10-27
**Verdict:** No candidate passed across `60` B-1 configs, `240` B-2 config-threshold rows, and `432` B-3 points (`0/432` met `>= 0.70` rescue retention at `<= 0.01` ordinary flips and `<= 1` harm, peaking around `~4.5%` retention at `<= 1%` flips), prompting the owner to close `F5` with no fit while keeping `tools/sandbox/` active.

## What was tried
- Admitted the filtered `D1024` counterfactual rescue corpus (`730` lives, `291` parents, `65,825` rows split into `54,771` train / `11,054` validation rows with `6,201` decisive and `936` ordinary validation rows after excluding `248` F4/census/R30 parents) under `tools/sandbox/{admission,preflight}.py`.
- Ran `B-1` (`60` configs, `15,000` updates on A100): swept linear, length/margin-gated 2-layer MLP, and cosine-normalized residual heads on frozen 256-D B8 features (`phi`) across `5` ordinary KL weights `{1,2,4,8,16}` and `4` survival-pair weights `{1,2,4,8}`.
- Ran `B-2` (`48` configs × `5` switch margins = `240` rows, `48,000` updates): swept `normed_linear`, `cosine_fastgain`, and `supervised_gated_normed` heads across `kl_pairs` vs `hinge_top`, ordinary:survival ratios `{0.5,2,8,32}`, and row-uniform vs life-balanced sampling.
- Ran `B-3` (`432` points, `6,000` updates): tested cosine `k`-NN retrieval (`k in {5,15,50}`) plus `12` direct survival-veto configs adding `[phi, q_base]` context and `12`-D local crop geometry (`7` crop values + `5` v8 scalars, fixing neck-TTL blocking), alongside `tools/sandbox/supervisor_v2.py` (`wait4` child CPU collector).

## What happened
- **`B-1` (`No candidate`, `68.18 s`):** Rescue retention reached `0.552`, but every configuration caused `>= 7` ordinary-harm-proxy cases on validation.
- **`B-2` (`0/240` promising, `296.88 s`, `0.33 u`):** Best rescue retention `0.587` required a `0.925` ordinary flip rate; at ordinary flip rate `<= 0.01`, best retention was `0.038` of `6,201` decisive rows.
- **`B-3` (`0/432` promising, `83.07 s`, `0.29 u`):** Best retention `0.571` came at `0.997` ordinary flips; at `<= 0.01` (`<= 1%`) ordinary flips, best retention was `0.045` (`~4.5%`), missing `>= 0.70` (`>= 70%`) retention with `<= 1` harm and forming the basis to close `F5` with no fit.
- **Child Collector v2 (`ACCEPT` offline):** Passed `621` compatibility and `263` focused tests (`12,680,000 ns` CPU captured in local smoke), remaining `NOT_ADMISSIBLE` for live served panels pending host integration qualification.

## Why
Single-frame 256-D B8 features (`phi`), `[phi, q_base]`, and `31 × 31` local crop geometry overlap heavily between future self-traps and safe coiling states because full-arena multi-snake enclosures and off-crop tail release happen outside the `31 × 31` view, forcing every head onto one retention-vs-false-override curve.

## What to reuse / what not to repeat
- Reuse `tools/sandbox/{admission,preflight,supervisor_v2}.py` (`wait4` integer-nanosecond child CPU accounting without touching the qualified `research/colab_cli` tree).
- Do not repeat post-hoc linear, MLP, retrieval, or local-crop veto heads on static frozen `ego2s-bv8` features expecting selective rescue retention (`>= 70%` at `<= 1%` ordinary flips).

## Full record
Original: `git show 90b3db94:docs/research/history/sandbox-b.md` | Primary sources: [sandbox_b_heads/README.md](../../../research/sandbox_b_heads/README.md), [B2.md](../../../research/sandbox_b_heads/B2.md), [B3.md](../../../research/sandbox_b_heads/B3.md)
