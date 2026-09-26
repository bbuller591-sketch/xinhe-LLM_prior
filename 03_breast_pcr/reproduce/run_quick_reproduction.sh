#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
conda run -n zcy-xinhe python REPRODUCE/verify_package.py
conda run -n zcy-xinhe python REPRODUCE/reproduce_m3_measurement_postprocess.py
conda run -n zcy-xinhe python REPRODUCE/reproduce_m1m2_bt.py
conda run -n zcy-xinhe python REPRODUCE/reproduce_final_sealed.py
echo "QUICK_REPRODUCTION_PASS"
