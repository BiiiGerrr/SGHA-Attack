#!/usr/bin/env bash
# Download directly on the machine where the experiments will run.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
backend="${1:?usage: bash scripts/fetch_backends.sh anchors|unidiffuser|lavis|minigpt4|llava|llava_next}"
case "$backend" in
  anchors) name=stable-diffusion; url=https://github.com/CompVis/stable-diffusion.git ;;
  unidiffuser) name=unidiffuser; url=https://github.com/thu-ml/unidiffuser.git ;;
  lavis|blip2|instructblip) name=LAVIS; url=https://github.com/salesforce/LAVIS.git ;;
  minigpt4) name=MiniGPT-4; url=https://github.com/Vision-CAIR/MiniGPT-4.git ;;
  llava) name=LLaVA; url=https://github.com/haotian-liu/LLaVA.git ;;
  llava_next) name=LLaVA-NeXT; url=https://github.com/LLaVA-VL/LLaVA-NeXT.git ;;
  *) echo "Unknown backend: $backend" >&2; exit 2 ;;
esac
mkdir -p vendor
if [[ -e "vendor/$name" ]]; then
  echo "Already present: vendor/$name (left unchanged)"
else
  git clone "$url" "vendor/$name"
  default_ref="$("${PYTHON:-python}" -c 'import json,sys; print(json.load(open("configs/backend_refs.json")).get(sys.argv[1], ""))' "$name")"
  if [[ -n "${BACKEND_REF:-$default_ref}" ]]; then git -C "vendor/$name" checkout --detach "${BACKEND_REF:-$default_ref}"; fi
fi
# Set BACKEND_REF to a commit/tag when reproducing an archived environment.
if [[ -n "${BACKEND_REF:-}" ]]; then git -C "vendor/$name" checkout --detach "$BACKEND_REF"; fi
git -C "vendor/$name" rev-parse HEAD > "vendor/${name}.revision.txt"
"${PYTHON:-python}" scripts/patch_backend.py --backend "$name" --directory "vendor/$name"
echo "Code fetched. Install this backend's dependencies as described in README.md."
