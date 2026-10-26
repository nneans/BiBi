#!/usr/bin/env bash
# Restarts cv_extra_units.py up to 5 times (finished folds are kept). Runs on the E-cores: bibi_reviews/cpu.py.
cd "$(dirname "$0")/../.."
export PYTHONPATH=src USE_TF=0 USE_JAX=0 TRANSFORMERS_NO_TF=1 PYTHONIOENCODING=utf-8
for start in $(seq 1 5); do
  timeout 3600 python 5_감성분류/분류기/cv_extra_units.py >> "5_감성분류/분류기/run-cv-extra${UNIT_VARIANT:+-$UNIT_VARIANT}.log" 2>&1 && { echo "done after $start starts"; exit 0; }
  echo "restart $start"
done
