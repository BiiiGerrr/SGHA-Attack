#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
"${PYTHON:-python}" -m sgha_attack.generate_anchors \
  --targets "${TARGETS:-data/metadata/targets_1000.txt}" \
  --seeds "${ANCHOR_SEEDS_FILE:-configs/anchor_seeds.json}" \
  --output-root "${ANCHOR_ROOT:-outputs/anchors}" "$@"
