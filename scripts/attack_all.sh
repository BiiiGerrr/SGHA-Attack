#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
# Seed used by the paper single-surrogate attacks.
seed="${SEED:-2023}"
for surrogate in vit_b vit_l vit_g; do
  "${PYTHON:-python}" -m sgha_attack.attack --surrogate "$surrogate" \
    --clean-root "${CLEAN_ROOT:-data/clean}" \
    --targets "${TARGETS:-data/metadata/targets_1000.txt}" \
    --anchor-record "${ANCHOR_RECORD:-outputs/anchors/seed_record.txt}" \
    --output "${ATTACK_ROOT:-outputs/attacks}/$surrogate/seed_$seed" \
    --seed "$seed" "$@"
done
