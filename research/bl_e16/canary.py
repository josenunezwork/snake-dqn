"""Prepare a real-entrypoint canary manifest; never allocate.

See research/colab_cli/REQUALIFICATION.md for the operator protocol.
"""

from research.colab_cli.canary import main

if __name__ == "__main__":
    main("bl_e16")
