#!/usr/bin/env bash
# package root: override with REPRO_ROOT=/path/to/code if needed
REPRO_ROOT="${REPRO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

echo "This runner preserves the historical absolute paths used by the analysis."
echo "Restore the packaged PROJECT and DATA snapshots to:"
echo "  "${REPRO_ROOT}/GBM_EXTERNAL_EVIDENCE_20260917""
echo "  "${REPRO_ROOT}/gbm_canonical_recovery_20260917""
echo "before executing this script."

python "$HERE/SCRIPTS/postprocess_completed_g132_v28.py"
python "$HERE/SCRIPTS/postprocess_completed_g116_v28.py"
python "$HERE/SCRIPTS/postprocess_completed_ivy_v28.py"
python "$HERE/SCRIPTS/final_measurement_audit_and_aggregate_v28.py"
python "$HERE/SCRIPTS/run_gbm_m0_m1_m2_downstream_v29.py"
python "$HERE/SCRIPTS/build_and_run_gbm_m3_v29.py"
python "$HERE/SCRIPTS/audit_and_summarize_gbm_downstream_v29.py"
python "$HERE/SCRIPTS/gbm_downstream_fold_consistency_v29.py"
python "$HERE/SCRIPTS/analyze_gbm_downstream_results_v29.py"
python "$HERE/verify_key_results.py"
