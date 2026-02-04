#!/usr/bin/env bash
set -euo pipefail
# =========================
# 可直接改的参数区（你只改这里）
# =========================
SESSION="gpu4"
# 进入项目目录（按需修改）
WORKDIR="$HOME/code/FragC3Pair"

# 可选：每个窗口跑之前要执行的环境初始化（按需修改/删掉）
# 例如 module load / conda activate / export CC/CXX 等
ENV_SETUP=$'conda activate fragc3'

# 四个任务：你把 python 脚本 + 参数直接写在 CMD 里
# 注意：这里不需要写 CUDA_VISIBLE_DEVICES，会自动分配 0/1/2/3
CMD0='python train_final.py --dataset ONeil --group none --encoder FragC3'
CMD1='python train_final.py --dataset ONeil --group Drug --encoder FragC3'
CMD2='python train_final.py --dataset ONeil --group Cell --encoder FragC3'
CMD3='python train_final.py --dataset ONeil --group Drug --encoder FragC3 --train_batch_size 256'

# 每个任务使用哪张 GPU（默认 0,1,2,3；你也可以改成别的映射）
GPU0=0
GPU1=1
GPU2=2
GPU3=3

# =========================
# 下面一般不用改
# =========================

run_in_tmux_window () {
  local win="$1"
  local gpu="$2"
  local cmd="$3"

  # 在 tmux window 里执行：cd -> env -> 绑 GPU -> 跑命令
  # 用 bash -lc 确保能执行 module/conda 等 shell 初始化逻辑
  tmux send-keys -t "${SESSION}:${win}" \
    "bash -lc 'cd \"${WORKDIR}\" \
      && ${ENV_SETUP} \
      && export CUDA_VISIBLE_DEVICES=${gpu} \
      && echo \"[INFO] $(date) | window=${win} | GPU=${gpu} | cmd=${cmd}\" \
      && ${cmd}'" C-m
}

# 如果会话已存在，直接 attach（避免误开多份）
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "[WARN] tmux session '${SESSION}' already exists. Attaching..."
  tmux attach -t "${SESSION}"
  exit 0
fi

# 新建会话（后台）
tmux new-session -d -s "${SESSION}" -c "${WORKDIR}"

# window0
tmux rename-window -t "${SESSION}:0" "gpu${GPU0}"
run_in_tmux_window 0 "${GPU0}" "${CMD0}"

# window1-3
tmux new-window -t "${SESSION}" -n "gpu${GPU1}" -c "${WORKDIR}"
run_in_tmux_window 1 "${GPU1}" "${CMD1}"

tmux new-window -t "${SESSION}" -n "gpu${GPU2}" -c "${WORKDIR}"
run_in_tmux_window 2 "${GPU2}" "${CMD2}"

tmux new-window -t "${SESSION}" -n "gpu${GPU3}" -c "${WORKDIR}"
run_in_tmux_window 3 "${GPU3}" "${CMD3}"

# 附加进入会话
tmux attach -t "${SESSION}"
