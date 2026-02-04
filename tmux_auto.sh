#!/usr/bin/env bash
set -euo pipefail

# =========================
# 可直接改的参数区（你只改这里）
# =========================
SESSION="gpu4"

CMD0='python train_final.py --dataset ONeil --group none --encoder FragC3'
CMD1='python train_final.py --dataset ONeil --group Drug --encoder FragC3'
CMD2='python train_final.py --dataset ONeil --group Cell --encoder FragC3'
CMD3='python train_final.py --dataset ONeil --group Drug --encoder FragC3 --train_batch_size 256'

GPU0=0
GPU1=1
GPU2=2
GPU3=3

# =========================
# 下面一般不用改
# =========================

# 脚本所在目录（保证所有任务都在这里运行）
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

run_in_tmux_window () {
  local win="$1"
  local gpu="$2"
  local cmd="$3"

  tmux send-keys -t "${SESSION}:${win}" \
    "cd \"${SCRIPT_DIR}\" \
     && export CUDA_VISIBLE_DEVICES=${gpu} \
     && echo \"[INFO] $(date) | window=${win} | GPU=${gpu} | cmd=${cmd}\" \
     && ${cmd}" C-m
}


# 如果会话已存在，直接 attach（避免误开多份）
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "[WARN] tmux session '${SESSION}' already exists. Attaching..."
  tmux attach -t "${SESSION}"
  exit 0
fi

# 新建会话（后台），并把默认目录设为脚本目录（更稳）
tmux new-session -d -s "${SESSION}" -c "${SCRIPT_DIR}"

# window0
tmux rename-window -t "${SESSION}:0" "gpu${GPU0}"
run_in_tmux_window 0 "${GPU0}" "${CMD0}"

# window1-3
tmux new-window -t "${SESSION}" -n "gpu${GPU1}" -c "${SCRIPT_DIR}"
run_in_tmux_window 1 "${GPU1}" "${CMD1}"

tmux new-window -t "${SESSION}" -n "gpu${GPU2}" -c "${SCRIPT_DIR}"
run_in_tmux_window 2 "${GPU2}" "${CMD2}"

tmux new-window -t "${SESSION}" -n "gpu${GPU3}" -c "${SCRIPT_DIR}"
run_in_tmux_window 3 "${GPU3}" "${CMD3}"

tmux attach -t "${SESSION}"
