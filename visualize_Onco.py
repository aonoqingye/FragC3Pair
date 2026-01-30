import random
import argparse
import numpy as np

from sklearn import metrics
from collections import defaultdict
from torch_geometric.loader import DataLoader

from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    recall_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    accuracy_score,
    precision_score,
)

from utils import *            # 依赖 save_AUCs 等工具函数（与原流程一致）
from const import *            # 依赖 DATAS_DIR / RESULTS_DIR 等常量（如未使用，仍可保留）
from model import *            # 依赖 MultiSyn 等模型实现（与原流程一致）
from pair_dataset import PairDataset
from tools.process_folds import process_folds
from explainer import save_cross_attn_pairlines, save_group_montage_by_pair

# -----------------------------
# tqdm 配置 & 日志
# -----------------------------
TQDM_KW = dict(dynamic_ncols=True, mininterval=1.0, smoothing=0.0)

def log(msg: str):
    """统一日志出口，避免与 tqdm 冲突（不要用 print）。"""
    tqdm.write(str(msg))


# -----------------------------
# 工具函数
# -----------------------------
def set_seed(seed: int):
    """固定随机种子（保持确定性设置）。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_device(device_arg: str):
    """根据参数选择设备，保持与原逻辑一致（优先 CUDA:0）。"""
    if device_arg and device_arg.lower() in {"cpu", "cuda"}:
        if device_arg.lower() == "cuda" and torch.cuda.is_available():
            log("The code uses GPU...")
            return torch.device("cuda:0")
        elif device_arg.lower() == "cuda" and not torch.cuda.is_available():
            log("CUDA 不可用，回退至 CPU。")
            return torch.device("cpu")
        else:
            log("The code uses CPU!!!")
            return torch.device("cpu")
    # auto
    if torch.cuda.is_available():
        log("The code uses GPU...")
        return torch.device("cuda:0")
    log("The code uses CPU!!!")
    return torch.device("cpu")


# -----------------------------
# 训练 / 评估
# -----------------------------
def train(
    args,
    model,
    device,
    train_loader,
    optimizer,
    loss_fn,
    epoch: int,
    log_interval: int,
    show_batch_pbar: bool = False,
):
    """
    模型训练过程：
    - 同步迭代两路药物图数据（drug1/drug2）与标签；
    - 前向 -> 反向 -> 更新；
    - 可选 batch 级 tqdm 进度条（默认关闭，减少残留与刷新开销）。
    """
    model.train()
    total_loss = 0.0

    total_batches = len(train_loader)
    iterator = train_loader

    if show_batch_pbar:
        iterator = tqdm(
            train_loader,
            total=total_batches,
            desc=f"Train epoch {epoch}",
            position=2,
            leave=False,
            **TQDM_KW,
        )

    for batch_idx, data in enumerate(iterator):
        data = data.to(device)
        if args.dataset == "ONeil":
            y = data.y.view(-1, 1).long().to(device).squeeze(1)
        else:
            y = data.y.view(-1, 1).float().to(device)  # 回归用 float

        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, y)
        if torch.isnan(loss):
            print(f"NaN loss at batch {batch_idx}")
            continue
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if show_batch_pbar and (batch_idx % max(1, log_interval) == 0):
            # 动态展示当前 loss
            iterator.set_postfix(loss=f"{loss.item():.6f}")

    if show_batch_pbar and hasattr(iterator, "clear"):
        iterator.clear()   # 确保不残留 batch 进度条行

    avg_loss = total_loss / max(1, total_batches)
    return avg_loss


@torch.no_grad()
def predicting(
    args,
    model,
    device,
    test_loader,
    show_batch_pbar: bool = False,
    epoch: int = None,
):
    """
    模型预测过程（评估阶段使用）：
    返回 真值标签、预测分数（正类概率）、预测标签。
    """
    model.eval()
    total_scores = []
    total_labels = []
    total_pred_labels = []

    total_batches = len(test_loader)
    iterator = test_loader

    if show_batch_pbar:
        tag = f"Eval epoch {epoch}" if epoch is not None else "Eval"
        iterator = tqdm(
            test_loader,
            total=total_batches,
            desc=tag,
            position=3,
            leave=False,
            **TQDM_KW,
        )

    for data in iterator:
        data = data.to(device)
        output = model(data)

        if args.dataset == "ONeil":
            ys = F.softmax(output, dim=1).to("cpu").data.numpy()
            pred_labels = np.argmax(ys, axis=1).tolist()
            pred_scores = [row[1] for row in ys]  # 正类概率
        else:
            ys = output.squeeze(1).detach().cpu().numpy()
            pred_labels = ys  # 预测值即标签
            pred_scores = ys

        total_scores.extend(pred_scores)
        total_pred_labels.extend(pred_labels)
        total_labels.extend(data.y.view(-1, 1).cpu().numpy().flatten().tolist())

    if show_batch_pbar and hasattr(iterator, "clear"):
        iterator.clear()

    return (
        np.asarray(total_labels).flatten(),
        np.asarray(total_scores).flatten(),
        np.asarray(total_pred_labels).flatten(),
    )


def compute_performance(T, S, Y, best_auc, file, epoch):
    """
    计算多项分类指标，并在 AUC 更优时写盘（保持原 save_AUCs 调用与写入格式）。
    - T: 真实标签
    - S: 预测分数（正类概率）
    - Y: 预测标签
    - best_auc: 历史最佳 AUC（用于比较）
    - file: 写入 CSV 的路径
    - epoch: 当前轮次（写入时保留）
    """
    AUC = roc_auc_score(T, S)
    precision, recall, _ = metrics.precision_recall_curve(T, S)
    PR_AUC = metrics.auc(recall, precision)
    BACC = balanced_accuracy_score(T, Y)
    tn, fp, fn, tp = confusion_matrix(T, Y).ravel()
    TPR = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    PREC = precision_score(T, Y, zero_division=0)
    ACC = accuracy_score(T, Y)
    KAPPA = cohen_kappa_score(T, Y)
    REC = recall_score(T, Y, zero_division=0)
    F1 = f1_score(T, Y, zero_division=0)

    AUCs_row = [epoch, AUC, PR_AUC, ACC, BACC, PREC, TPR, KAPPA, REC, F1]
    save_AUCs(AUCs_row, file)  # 依赖 utils.save_AUCs（与原逻辑一致）
    if best_auc < AUC:
        best_auc = AUC
    return best_auc, AUC

def compute_performance_regression(T, P, best_mse, file, epoch):
    mse_val = np.mean((np.array(T) - np.array(P))**2)
    rmse_val = rmse(P, T)
    mae_val = mae(P, T)
    print(f'Evaluation MSE: {mse_val}')


# -----------------------------
# 主流程
# -----------------------------
def main():
    args = parse_args()

    # 1) 固定随机性
    set_seed(args.seed)

    # 2) 设备
    device = build_device(args.device)

    # 3) 数据集加载（两路药物图，保证与原数据组织一致）
    work_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(work_dir, "datas")  # 保持你当前写法
    dataset = PairDataset(root=data_dir, dataset=args.dataset)

    if hasattr(dataset, "cell2") and dataset.cell2 is not None:
        cell_dim = dataset.cell1.size(1) + dataset.cell2.size(1)
    else:
        cell_dim = dataset.cell1.size(1)

    log(f"Parameters:  {args}")

    # 结果目录与每折 CSV 初始化
    out_dir = os.path.join(work_dir, "results")
    os.makedirs(out_dir, exist_ok=True)

    # —— 构造 DataLoader（保持默认不打乱，与原流程一致） —— #
    test_loader  = DataLoader(dataset,  batch_size=args.test_batch_size)

    # —— 构建模型/损失/优化器 —— #
    if args.task == "classification":
        loss_fn = nn.CrossEntropyLoss()
        n_output = 2
    else:  # regression
        loss_fn = nn.MSELoss()
        n_output = 1

    model = MultiFrag(
        n_output=n_output,
        cell_dim=cell_dim,
        hid_dim=args.hidden,
        heads=args.heads,
        ffn_expansion=args.ffn_expansion,
        cell_attn=args.cell_attn,
        dropout=args.dropout,
        frag_list=args.frag_list,
        share_encoder=args.share_encoder,
        frag_agg=args.frag_agg,
    ).to(device)

    out_info = f'{args.dataset}_Frags_'
    for i in range(len(args.frag_list)):
        out_info += f"{args.frag_list[i]}_"
    # if args.share_encoder:
    #     out_info += 'share_encoder_'
    # else:
    #     out_info += 'separate_'
    out_info += f"{args.frag_agg}_agg_Group_{args.groups}"
    if args.cell_attn:
        out_info += '_withCell'
    else:
        out_info += '_woCell'

    model_path = os.path.join(args.model_path, f"{out_info}_fold_0.pt")
    # ====== 如果存在已训练权重，优先加载 ======
    T, S, _ = load_checkpoint(model, model_path, map_location=device)

    T, S, _ = predicting(
        args, model, device, test_loader,
        show_batch_pbar=args.show_batch_pbar, epoch=None
    )
    compute_performance_regression(T, S, np.inf, file=None, epoch=None)

    # -----------------------------
    # 测试后：按任务类型筛选“最接近真实标签”的前 visualize_n 个样本并画图
    # -----------------------------
    length = len(T)
    # ---- 计算误差（归一化相对误差）----
    T_arr = np.asarray(T, dtype=float)
    S_arr = np.asarray(S, dtype=float)
    epsilon = 1e-8
    abs_err = np.abs(S_arr - T_arr)  # |ŷ - y|
    rel_err = abs_err / (np.abs(T_arr) + epsilon)  # |ŷ - y| / (|y|+eps)

    # ---- 构造“药物对”分组键 (drug1_id, drug2_id) ----
    drug_ids_a_all = np.asarray(dataset.drug1_id) if hasattr(dataset, "drug1_id") else None
    drug_ids_b_all = np.asarray(dataset.drug2_id) if hasattr(dataset, "drug2_id") else None

    def _pair_for_gidx(gi: int):
        if drug_ids_a_all is not None and drug_ids_b_all is not None:
            return (str(drug_ids_a_all[gi]), str(drug_ids_b_all[gi]))
        return (str(gi), str(gi))

    # ====== 新增：在样本选择前先检查是否已有索引文件 ======
    attn_out_dir = os.path.join(work_dir, "attn_pair")
    os.makedirs(attn_out_dir, exist_ok=True)
    sel_idx_path = os.path.join(attn_out_dir, "selected_indices.txt")

    selected_global_idx = None
    top_pairs = []  # 仅用于后续写统计文件时引用

    if os.path.exists(sel_idx_path) and args.cell_attn == False:
        # 已有索引文件，直接读取并跳过选择
        try:
            loaded = np.loadtxt(sel_idx_path, dtype=int, ndmin=1)
            # 兼容只有单个值时的形状
            selected_global_idx = np.asarray(loaded, dtype=np.int64).reshape(-1)
            log(f"Loaded {len(selected_global_idx)} indices from existing file: selected_indices.txt")
        except Exception as e:
            log(f"Failed to load existing indices, will re-select. Reason: {type(e).__name__}: {e}")

    # 若没有预存索引，才执行分组筛选逻辑
    if selected_global_idx is None:
        # ---- 按药物对分组 & 剔除样本量 < 2 的组 ----
        group = defaultdict(list)  # pair -> list[(global_idx, rel_e)]
        for i in range(length):
            pair = _pair_for_gidx(int(i))
            group[pair].append((int(i), float(rel_err[i])))

        # 对组内样本按 rel_err 升序排序, 取前 <= 4 个样本
        group = {pair: sorted(items, key=lambda x: x[1])[:4] for pair, items in group.items() if len(items) >= 2}

        # ---- 选择组内平均相对误差最小的前 visualize_n 个组 ----
        if len(group) == 0:
            fallback_k = min(args.visualize_n, len(rel_err))
            selected_global_idx = np.argsort(rel_err)[:fallback_k]
            log(f"No groups with >=2 samples. Fallback to top-{fallback_k} individual samples.")
        else:
            stats = []  # (pair, mean_rel_err, size)
            for pair, items in group.items():
                mean_rel = float(np.mean([e for _, e in items]))
                stats.append((pair, mean_rel, len(items)))
            stats.sort(key=lambda x: x[1])

            top_pairs = stats[: min(args.visualize_n, len(stats))]
            selected = []
            errs = []
            for pair, _, _ in top_pairs:
                selected.extend([i for i, _ in group[pair]])
                errs.extend([e for _, e in group[pair]])
            selected_global_idx = np.asarray(selected, dtype=np.int64)
            selected_rel_errs = np.asarray(errs, dtype=np.float16)

            log_lines = [
                "pair(mean_rel_err,size): " + ", ".join(
                    [f"{p[0]}|{p[1]}({m:.4f},{s})" for (p, m, s) in top_pairs])
            ]
            log(f"Selected {len(top_pairs)} groups, total {len(selected_global_idx)} samples.\n" + "\n".join(
                log_lines))

        # 仅在“重新选择”的情况下写统计与索引文件
        group_stat_path = os.path.join(attn_out_dir, "selected_groups.txt")
        with open(group_stat_path, "w", encoding="utf-8") as f:
            f.write("pair_a,pair_b,mean_err,size\n")
            for pair, mean_err, size in (top_pairs if len(top_pairs) > 0 else []):
                f.write(f"{pair[0]},{pair[1]},{mean_err:.6f},{size}\n")

        np.savetxt(sel_idx_path, selected_global_idx.reshape(-1, 1), fmt="%d", header="global_idx")

    log(f"Visualizing {len(selected_global_idx)} samples from top groups...")

    # ===== 批量版：把 top-k 样本一次性打成 batch，单次前向拿注意力 =====
    # # 1) 收集单样本图
    # g_list = [dataset[gi] for gi in selected_global_idx]
    # # 2) 合批（兼容 PyG / DGL）
    # def _batch_list(graph_list):
    #     from torch_geometric.data import Batch as PygBatch
    #     if graph_list and graph_list[0].__class__.__name__ in ("Data", "HeteroData"):
    #         return PygBatch.from_data_list(graph_list).to(device)

    data_batch = dataset[selected_global_idx]
    batch_loader = DataLoader(data_batch, batch_size=len(data_batch))
    for _, batch in enumerate(batch_loader):
        output_b, extra_b, agg_attn = model(batch.to(device), return_attn=True)

    # 4) 组装批量标签与标识
    y_b = data_batch.y.view(-1)  # [B]
    cell_id_b = [str(dataset.cell_id[i]) if hasattr(dataset, "cell_id") else ""
                 for i in selected_global_idx]
    drug_id_a_b = [str(dataset.drug1_id[i]) for i in selected_global_idx]
    drug_id_b_b = [str(dataset.drug2_id[i]) for i in selected_global_idx]

    # 6) 一次性可视化（visualize_n=批量大小）
    group_ae, group_re = save_cross_attn_pairlines(
        y=y_b,  # [B]
        output=output_b,  # [B, C] 或 [B, 1]
        extra_per_view=extra_b,
        agg_attn=agg_attn,
        cell_id=cell_id_b,  # len=B
        drug_id_a=drug_id_a_b,  # len=B
        drug_id_b=drug_id_b_b,  # len=B
        idx=selected_global_idx,
        visualize_n=len(selected_global_idx),
        with_cell=args.cell_attn,
        out_dir=attn_out_dir
    )
    log(f"Batched visualization done (B={len(selected_global_idx)}).\n")
    # 若开启按药物对聚合拼图：把同组样本上下并排到一张图里（每个药物对一张）
    if args.group_visualize:
        # --- 构建 test_idx 到误差数组位置的映射 ---
        # idx_to_pos = {int(g): pos for pos, g in enumerate(test_idx_np)}

        # --- 按 selected_global_idx 顺序提取对应误差 ---
        # abs_err_batch = np.array([abs_err[idx_to_pos[int(g)]] for g in selected_global_idx])
        # rel_err_batch = np.array([rel_err[idx_to_pos[int(g)]] for g in selected_global_idx])
        save_group_montage_by_pair(
            drug_id_a=drug_id_a_b,
            drug_id_b=drug_id_b_b,
            cell_id=cell_id_b,
            idx=selected_global_idx,
            group_ae=group_ae,  # << 新增：绝对误差
            group_re=group_re,  # << 新增：相对误差
            out_dir=attn_out_dir,
            with_cell=args.cell_attn,
            base_height=2400,
            col_width=1600, # 列宽，列数=组内样本数
            dpi=200
        )
        log(f"Group montage images saved.")

    # —— 全部折完成后：汇总为 Excel —— #
    process_folds(args, out_dir, out_info)

# -----------------------------
# 参数
# -----------------------------
def parse_args():
    """统一管理可调超参数与配置。"""
    p = argparse.ArgumentParser(description="Two-drug graph training & evaluation (K-fold) with tqdm")
    # 训练/评估超参数
    p.add_argument("--train_batch_size", type=int, default=1024, help="训练批大小")
    p.add_argument("--test_batch_size", type=int, default=1024, help="测试批大小")
    p.add_argument("--lr", type=float, default=1e-4, help="学习率")
    p.add_argument("--epochs", type=int, default=100, help="训练轮数")
    p.add_argument("--log_interval", type=int, default=20, help="训练日志打印间隔（按 batch）")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--folds", type=int, default=5, help="随机划分折数（默认 5）")
    p.add_argument("--groups", type=str, default="none",
                   choices=["Cell", "DrugPair", "none"], help="k fold分组依据")
    p.add_argument("--info", type=str, default="", help="输出文件信息")
    # 模型参数
    p.add_argument("--hidden", type=int, default=300, help="隐层维度")
    # 片段参数
    p.add_argument("--frag_list", nargs="+", default=["brics", "fg", "murcko"],
                   help='"brics", "fg", "murcko", "ringpaths"')
    p.add_argument("--share_encoder", type=bool, default=False, help="开启Bi2Frag编码")
    p.add_argument("--frag_agg", type=str, default="cell_attn",
                   choices=["mlp", "gate", "cell_attn"], help="多视角融合机制")
    # Bi2Frag参数
    p.add_argument("--bi2frag", type=bool, default=True, help="开启Bi2Frag编码")
    p.add_argument("--heads", type=int, default=4, help="注意力头数")
    p.add_argument("--ffn_expansion", type=int, default=4, help="FFN扩张倍数")
    p.add_argument("--cell_attn", type=bool, default=False, help="开启cell line注意力")
    p.add_argument("--dropout", type=float, default=0.2)
    # 可视化
    p.add_argument('--visualize_attn', type=bool, default=True, help='Save fragment-level cross-attention heatmaps')
    p.add_argument('--visualize_n', type=int, default=8, help='Number of test pairs to visualize')
    p.add_argument('--group_visualize', type=bool, default=True, help='在批量单图绘制后，按药物对把同组样本上下拼接成一张图')
    # 模型保存/读取
    p.add_argument("--save_model", type=bool, default=True, help="是否在验证最佳时保存模型参数")
    p.add_argument("--load_model", type=bool, default=True, help="是否在验证最佳时保存模型参数")
    p.add_argument("--model_path", type=str, default="save", help="模型参数保存/读取路径")
    # 数据与设备
    p.add_argument("--dataset", type=str, default="OncoPolyPharmacology",
                   choices=["ONeil", "OncoPolyPharmacology"], help="数据集前缀名")
    p.add_argument("--task", type=str, default="regression",
                   choices=["classification", "regression"], help="任务类型：classification 或 regression")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="计算设备选择")
    # 可选：是否显示 batch 级进度条（若训练很快可关掉减少开销/刷新）
    p.add_argument("--show_batch_pbar", type=bool, default=True, help="显示 batch 级别 tqdm 进度条")
    return p.parse_args()


if __name__ == "__main__":
    # sleep(1500)
    main()
