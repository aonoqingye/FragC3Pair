#!/bin/bash

set -euo pipefail
unset LD_LIBRARY_PATH

SESSION="gpu2"
WORKDIR="${WORKDIR:-$PWD}"

# 下面四条命令改成你自己的脚本/参数（也可以换成 bash xxx.sh）
CMD0='python train_final.py --dataset ONeil --groups none'
CMD1='python train_final.py --dataset ONeil --groups Drug'
# CMD2='python train_final.py --dataset ONeil --groups Cell'
# CMD3='python train_final.py --dataset ONeil --groups Drug --train_batch_size 256'

# ACTIVATE='conda activate fragc3 && export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"'
ACTIVATE='source /home/kimi/miniconda3/etc/profile.d/conda.sh && conda activate fragc3'

# 如果 session 已存在，直接提示并 attach（避免重复起）
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "[WARN] tmux session '$SESSION' already exists. Attaching..."
  tmux attach -t "$SESSION"
  exit 0
fi

tmux new-session -d -s "$SESSION" -c "$WORKDIR"
tmux set-option -t "$SESSION" remain-on-exit on

# window 0 (GPU0)
tmux rename-window -t "$SESSION:0" "gpu2"
tmux send-keys -t "$SESSION:0" "cd '$WORKDIR'" C-m
tmux send-keys -t "$SESSION:0" "$ACTIVATE" C-m
tmux send-keys -t "$SESSION:0" "export CUDA_VISIBLE_DEVICES=2" C-m
tmux send-keys -t "$SESSION:0" "$CMD0" C-m

# window 1 (GPU1)
tmux new-window -t "$SESSION" -n "gpu3" -c "$WORKDIR"
tmux send-keys -t "$SESSION:1" "cd '$WORKDIR'" C-m
tmux send-keys -t "$SESSION:1" "$ACTIVATE" C-m
tmux send-keys -t "$SESSION:1" "export CUDA_VISIBLE_DEVICES=3" C-m
tmux send-keys -t "$SESSION:1" "$CMD1" C-m

## window 2 (GPU2)
#tmux new-window -t "$SESSION" -n "gpu2" -c "$WORKDIR"
#tmux send-keys -t "$SESSION:2" "cd '$WORKDIR'" C-m
#tmux send-keys -t "$SESSION:2" "$ACTIVATE" C-m
#tmux send-keys -t "$SESSION:2" "export CUDA_VISIBLE_DEVICES=2" C-m
#tmux send-keys -t "$SESSION:2" "$CMD2" C-m
#
## window 3 (GPU3)
#tmux new-window -t "$SESSION" -n "gpu3" -c "$WORKDIR"
#tmux send-keys -t "$SESSION:3" "cd '$WORKDIR'" C-m
#tmux send-keys -t "$SESSION:3" "$ACTIVATE" C-m
#tmux send-keys -t "$SESSION:3" "export CUDA_VISIBLE_DEVICES=3" C-m
#tmux send-keys -t "$SESSION:3" "$CMD3" C-m

echo "[OK] Started tmux session '$SESSION' with 2 windows (gpu2..gpu3)."
echo "     Attach: tmux attach -t $SESSION"
echo "     List  : tmux ls"
echo "     Kill  : tmux kill-session -t $SESSION"

tmux attach -t "$SESSION"
