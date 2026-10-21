# Reorg proposals — scope 22 (`web/`)

Proposals only; nothing below was changed (web/ is read-only except README).

1. **Stale docstring in `web/backend/safety_veto_serving.py`** (~line 33): says "`v2` (the default
   while `VARIANT_RELEASED_DEFAULT` is `"v2"`)" but the constant is `VARIANT_V8`. Reword to
   "historical default". Same for `web/backend/__init__.py` ("GameState + ApexPolicy"), which
   predates raster/ego2s serving (b8-s3 is not an ApexPolicy).
2. **`app.py` module docstring** gives `uvicorn ... --reload --port 8000` as the run command; the
   supported launcher is `web/serve.py` (pins torch threads). Point the docstring at `serve.py`.
3. **Split `session.py` (1564 lines)**: extract served-checkpoint/roster resolution and
   ego2s/b8 release-monitoring glue into `web/backend/serving_build.py`; keep `GameSession`
   for game/controls.
4. **Group serving-policy adapters** into `web/backend/policies/` (`raster_policy.py`,
   `ego2s_policy.py`) and Path B into `web/backend/pathb/` (`pathb.py`, `pathb_controller.py`).
   Needs import updates in tests; do only with the test suite run.
5. **Hard-coded absolute artifact paths** (`/Users/josenunez/...`) in `served_checkpoint.py` and
   `served_checkpoint_pins.json` tie serving to this Mac. Consider a `SNAKE_ARTIFACTS_ROOT`
   prefix (pins are hash-bound, so relocation would not weaken verification).
6. **One env-var table** owned in code (e.g. `web/backend/env.py`) that the README/WEB_MAP
   can be generated from; currently the `SNAKE_SERVE_*` vars are spread over four modules.
