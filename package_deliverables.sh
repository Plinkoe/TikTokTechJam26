#!/usr/bin/env bash
set -euo pipefail

OUT=kuairand_deliverables.zip
rm -f "$OUT"
# NOTE: agent.py is only a CLI wrapper around agent_architecture.py — shipping
# one without the other produces a deliverable that can't actually run.
# history_model.py and requirements.txt were also missing before: without
# history_model.py, finalize() for the champion (history_deepfm) can't run at
# all, which is exactly the bug that caused the shipped submission to silently
# be the FM baseline instead of the champion (see README "Bugs found and fixed").
zip -r "$OUT" \
  README.md results_summary.md requirements.txt \
  run_logs baseline_scores.json \
  baseline.py data.py evaluate.py \
  agent.py agent_architecture.py history_model.py finalize_and_score.py \
  features.py models.py submit.py \
  || true
echo "Created $OUT"
