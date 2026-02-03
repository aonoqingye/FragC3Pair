#!/bin/bash
#SBATCH -J ONeil_StageA
#SBATCH -p i64m1tga40ue
#SBATCH -o logs/ONeil_StageA_%A_%a.out
#SBATCH -e logs/ONeil_StageA_%A_%a.err
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --array=0-11

# 环境准备
module load cuda/12.2
source /hpc2ssd/softwares/anaconda3/bin/activate pytorch_gpu_2.0.1
conda activate EGNN
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

set -euo pipefail

# -----------------------------
# Stage A 搜索空间（仅结构项）
# -----------------------------
TRI_VARIANTS=("scale_dot" "add" "dot" "trilinear")     # 4
CV_MODES=("mul" "add" "bilinear")                      # 3

DATASET="ONeil"
GROUP="Drug"
ENCODER="FragC3"

# 省时设置（按需调整）
EPOCHS=80
EARLY_STOP=25

# -----------------------------
# 0..11 -> tri_idx(0..3) + cv_idx(0..2)
# IDX = tri_idx*(#CV_MODES) + cv_idx
# -----------------------------
IDX="${SLURM_ARRAY_TASK_ID}"
TRI_IDX=$(( IDX / ${#CV_MODES[@]} ))          # /3
CV_IDX=$(( IDX % ${#CV_MODES[@]} ))          # %3

TRI="${TRI_VARIANTS[$TRI_IDX]}"
CVM="${CV_MODES[$CV_IDX]}"

echo "============================================================"
echo "[INFO] job=${SLURM_JOB_ID} array_task=${SLURM_ARRAY_TASK_ID}"
echo "[INFO] $(date '+%F %T') | dataset=${DATASET} | groups=${GROUP} | encoder=${ENCODER}"
echo "[INFO] tri_variant=${TRI} | cv_mode=${CVM}"
echo "[INFO] epochs=${EPOCHS} | early_stopping=${EARLY_STOP}"
echo "[INFO] node=${SLURMD_NODENAME:-unknown} | cpus=${SLURM_CPUS_ON_NODE:-8} | gres=${SLURM_JOB_GPUS:-1}"
echo "============================================================"

python train_final.py \
  --dataset "${DATASET}" \
  --groups "${GROUP}" \
  --encoder "${ENCODER}" \
  --tri_variant "${TRI}" \
  --cv_mode "${CVM}" \
  --epochs "${EPOCHS}" \
  --early_stopping "${EARLY_STOP}" \
  --resume

echo "[INFO] $(date '+%F %T') | finished"