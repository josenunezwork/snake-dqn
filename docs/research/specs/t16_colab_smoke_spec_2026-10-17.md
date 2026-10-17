# Spec: T16 Colab H100 environment + throughput smoke notebook (2026-10-17)

Read AGENTS.md and `docs/research/advice/2026-10-17_colab_h100.md` (step 1, binding). Implementer: Codex. The owner will run
the notebook in their own Colab account (H100/A100/L4); you cannot run Colab. Build it so one "Run all" produces a results
file.
1. `notebooks/colab_throughput_smoke.ipynb`: clones github.com/josenunezwork/snake-dqn at a pinned commit (parameter cell),
   installs the pinned stack (numba 0.68.0, llvmlite 0.50.0, numpy 1.26.4 and the repo's other pins; a CUDA torch matching
   the runtime), prints GPU/CPU info, runs the grid-sim parity and ego-live identity tests (fast subset), then times the
   trainer with `--device cuda` (and `--compile-learner reduce-overhead` if a C compiler is present) for: (a) the hero-only
   B8 recipe at 128 worlds with 2, 4, 8 actors; (b) a self-play configuration in which all snakes are driven by the
   learning network, at 64 and 128 worlds, if the trainer supports it; if it does not, add the minimal default-off
   trainer support needed (all snakes B8-driven, transitions collected for every driven snake) with tests, and note it.
   Each timing run ~2-3 min; checkpoints/results to Google Drive; resumable; total <= ~20 min.
2. Inputs it needs that are not in git (B8 finals, demo data, any bank): list them; add a small script to export them from
   the Mac into one tarball with sha256s that the owner uploads to Drive, and have the notebook verify the hashes.
3. Output: `colab_throughput_results.json` (transitions/s per config, GPU, CPU count, versions, commit) that the owner
   copies back; add `research/t16_colab/compare.py` that compares it with the Mac's 5.4k/s baseline and states whether each
   config clears 4x.
4. Doc `docs/research/t16_colab_smoke_2026-10-17.md`: exact owner steps (upload tarball, open notebook, pick runtime, Run
   all, download results).
Tests for any new code; flake8. Do not change serving or defaults. Git: branch t16-20261017; commit often; after each
commit `git fetch origin && git merge origin/main`, `git push origin HEAD:main`, `git push origin t16-20261017`.
Final message: notebook path, tarball command + size, owner steps, sha.
