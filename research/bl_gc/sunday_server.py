"""Launch real web/serve.py with a qualification-only Watch roster size."""

from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
from dataclasses import replace


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, choices=(6, 8, 12), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    import web.backend.session as session
    from research.apex_veto_serving_20261001.serving_run import set_world_seed

    original = session.load_config

    def local_count(path: str):
        config = original(path)
        return replace(config, game=replace(config.game, num_snakes=args.count))

    # Instrument only the harness child: the same GameSession constructor runs
    # after the real lifespan initializer, then emits its actual d.0 facts.
    from research.bl_gc import common

    original_init = session.GameSession.__init__

    def attested_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        print(
            "bl-gc-sunday-attestation: " + json.dumps(common.attest(self, {})),
            file=sys.stderr,
            flush=True,
        )

    session.GameSession.__init__ = attested_init
    session.load_config = local_count
    set_world_seed(args.seed)
    os.environ["PORT"] = str(args.port)
    sys.argv = ["web/serve.py"]
    runpy.run_path("web/serve.py", run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
