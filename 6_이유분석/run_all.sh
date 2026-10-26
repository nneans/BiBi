#!/usr/bin/env bash
# Runs every unfinished product in products.json in one process (run_product.py --all). Jobs pin themselves to the
# E-cores (bibi_reviews/cpu.py: this PC's P-cores crash); a crash or a 40-minute hang restarts the process, at most
# 5 times, and run_product.py resumes from its saved stages. More restarts would only hide a real bug.
cd "$(dirname "$0")/.."
export PYTHONPATH=src USE_TF=0 USE_JAX=0 TRANSFORMERS_NO_TF=1 PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false  # thread counts and E-core pinning: bibi_reviews/cpu.py
RUN=v3 VERSION=5  # must match RUN and REPORT_VERSION in run_product.py
mkdir -p 6_이유분석/results-$RUN
remaining() {
  python -c "import json,os;R='6_이유분석/results-$RUN';ok=lambda p:os.path.exists(f'{R}/{p}/report.json') and json.load(open(f'{R}/{p}/report.json',encoding='utf-8')).get('report_version')==$VERSION;print(sum(not ok(p['id']) for p in json.load(open('6_이유분석/products.json',encoding='utf-8'))))"
}
for start in $(seq 1 5); do
  left=$(remaining)
  echo "== start $start $(date +%H:%M:%S) | $left left"
  [ "$left" = 0 ] && break
  timeout 2400 python 6_이유분석/run_product.py --all >> 6_이유분석/results-$RUN/run.log 2>&1
  echo "   exit $?"
done
echo "all done $(date +%H:%M:%S) | $(remaining) left"
