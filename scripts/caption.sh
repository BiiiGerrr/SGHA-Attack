#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
model="${1:?model}"; images="${2:?adversarial image directory}"; output="${3:?caption .txt output}"
if [[ "$model" == unidiffuser ]]; then default_seed=1234; else default_seed=2023; fi
seed="${4:-$default_seed}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
"${PYTHON:-python}" -m sgha_attack.caption --model "$model" --images "$images" --output "$output" --seed "$seed"
