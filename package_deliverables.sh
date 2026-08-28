#!/usr/bin/env bash
set -euo pipefail

OUT=kuairand_deliverables.zip
rm -f "$OUT"
zip -r "$OUT" README.md results_summary.md run_logs baseline_scores.json baseline.py data.py evaluate.py agent.py features.py models.py submit.py || true
echo "Created $OUT"
