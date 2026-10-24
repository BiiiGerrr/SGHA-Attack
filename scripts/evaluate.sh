#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
captions="${1:?caption file}"; output="${2:?output prefix}"
targets="${TARGETS:-data/metadata/targets_1000.txt}"
original="${ORIGINAL_CAPTIONS:?Set ORIGINAL_CAPTIONS to the shared clean MiniGPT-4 descriptions}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
"${PYTHON:-python}" -m sgha_attack.eval_clip --captions "$captions" --targets "$targets" --output "${output}_clip.json"
"${API_PYTHON:-python}" -m sgha_attack.eval_gpt --captions "$captions" --targets "$targets" --original "$original" --output "${output}_gpt.json" --model "${JUDGE_MODEL:-gpt-4.1-mini}"
