#!/bin/bash
# package root: override with REPRO_ROOT=/path/to/code if needed
REPRO_ROOT="${REPRO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
set -euo pipefail
Q="${REPRO_ROOT}/09_cross_model_qwen/multi_dataset_swap"
PY=${PYTHON:-python3}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
run_renal(){ "$PY" "$Q/scripts/renal_select_lambda_foldlocal_gpt.py"; "$PY" "$Q/scripts/renal_finalize_pre_external_gpt.py"; "$PY" "$Q/scripts/renal_external_eval_gpt.py"; }
run_sepsis(){ "$PY" "$Q/scripts/sepsis_run_top30_top50_nested_gpt.py"; }
run_breast(){ "$PY" "$Q/scripts/breast_run_nested_m3_gpt.py"; "$PY" "$Q/scripts/breast_finalize_development_m3_gpt.py"; "$PY" "$Q/scripts/breast_sealed_eval_gpt.py"; }
run_credit(){ "$PY" "$Q/scripts/credit_analyze_llm_measurement.py"; "$PY" "$Q/scripts/credit_select_gamma_development.py"; "$PY" "$Q/scripts/credit_build_final_selected_sets.py"; "$PY" "$Q/scripts/credit_evaluate_final_holdout_once.py"; }
run_hospital(){ "$PY" "$Q/scripts/hospital_run_batch1_downstream_gpt.py"; "$PY" "$Q/scripts/hospital_freeze_and_batch2_eval_gpt.py"; }
run_darmanis(){ "$PY" "$Q/scripts/darmanis_build_run_m3_gpt.py"; }
run_renal >"$Q/logs/main_renal.log" 2>&1 & p1=$!
run_sepsis >"$Q/logs/main_sepsis.log" 2>&1 & p2=$!
run_breast >"$Q/logs/main_breast.log" 2>&1 & p3=$!
run_credit >"$Q/logs/main_credit.log" 2>&1 & p4=$!
run_hospital >"$Q/logs/main_hospital.log" 2>&1 & p5=$!
run_darmanis >"$Q/logs/main_darmanis.log" 2>&1 & p6=$!
rc=0
for p in $p1 $p2 $p3 $p4 $p5 $p6; do wait "$p" || rc=1; done
exit $rc
