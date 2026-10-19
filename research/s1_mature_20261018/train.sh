#!/usr/bin/env bash
set -euo pipefail

: "${S1_ARM:?missing S1_ARM}"
: "${S1_SEED:?missing S1_SEED}"
: "${COLAB_CLI_ROOT:?supervisor did not provide COLAB_CLI_ROOT}"

exec python -m research.s1_mature_20261018.train \
  --arm "$S1_ARM" --seed "$S1_SEED" --root "$COLAB_CLI_ROOT"
