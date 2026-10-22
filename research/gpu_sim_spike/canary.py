"""Prepare an excluded real-entrypoint target-platform canary.

See research/colab_cli/REQUALIFICATION.md.
"""

from research.colab_cli.canary import main

if __name__ == "__main__":
    main("gpu_sim_spike")
