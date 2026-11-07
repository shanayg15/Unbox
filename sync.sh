#!/usr/bin/env bash
set -e
git add -A
git commit -m "${1:-sync}" || true
git pull --rebase origin main
pytest -q tests/test_isolation.py || echo "WARNING: isolation tests failing"
git push origin main
