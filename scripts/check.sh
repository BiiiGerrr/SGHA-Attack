#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
"${PYTHON:-python}" -m compileall -q sgha_attack configs
"${PYTHON:-python}" -m unittest discover -s tests -v
"${PYTHON:-python}" scripts/audit_release.py
for script in scripts/*.sh; do bash -n "$script"; done
