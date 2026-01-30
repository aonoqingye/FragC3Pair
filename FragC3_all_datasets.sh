#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

GPU=3
DATASETS=("ONeil" "DrugComb" "ALMANAC")
CV_GROUPS=("Cell" "Drug" "none")

echo "[INFO] Start sequential runs on GPU=${GPU}"
echo "[INFO] Datasets: ${DATASETS[*]}"
echo "[INFO] Groups: ${CV_GROUPS[*]}"
echo

for data in "${DATASETS[@]}"; do
  for grp in "${CV_GROUPS[@]}"; do
    echo "============================================================"
    echo "[INFO] $(date '+%F %T') | GPU=${GPU} | dataset=${data} | groups=${grp}"
    echo "============================================================"
    CUDA_VISIBLE_DEVICES=${GPU} python train_final.py --dataset "${data}" --groups "${grp}" --encoder "FragC3"
    echo "[INFO] $(date '+%F %T') | dataset=${data} | groups=${grp} finished"
    echo
  done
done

echo "[INFO] All runs finished."
