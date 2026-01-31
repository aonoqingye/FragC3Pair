import random
import argparse
import numpy as np

from tqdm import tqdm
from sklearn.model_selection import KFold, GroupShuffleSplit, ShuffleSplit

from sklearn import metrics
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    recall_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    accuracy_score,
    precision_score,
)
from torch_geometric.loader import DataLoader

from tools.fidelity_utils import eval_fidelity
from utils import *  # 依赖 save_AUCs 等工具函数（与原流程一致）
from model.FragC3 import *
from dataset import PairDataset

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


def save_checkpoint(save_path: str, model, optimizer, epoch: int, best_val: float, args, extra: dict = None):
    """保存最佳模型参数到磁盘（state_dict + 训练元信息）。"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "epoch": int(epoch),
        "best_val": float(best_val),
        "args": vars(args),
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, save_path)


def compute_performance_classification(T, S, Y, best_auc, file, epoch):
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

    row = [epoch, AUC, PR_AUC, ACC, BACC, PREC, TPR, KAPPA, REC, F1]
    save_AUCs(row, file)
    if best_auc < AUC:
        best_auc = AUC
    return best_auc, AUC


def compute_performance_regression(T, P, best_mse, file, epoch):
    mse_val = np.mean((np.array(T) - np.array(P)) ** 2)
    rmse_val = rmse(P, T)
    mae_val = mae(P, T)
    row = [epoch, float(mse_val), float(rmse_val), float(mae_val)]
    save_AUCs(row, file)  # 依赖 utils.save_AUCs（与原逻辑一致）
    if best_mse > mse_val:
        best_mse = mse_val
    return best_mse, mse_val


# -----------------------------
# 训练 / 评估
# -----------------------------
def train(
        args,
        model,
        device,
        task,
        loader_train,
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

    total_batches = len(loader_train)
    iterator = loader_train

    if show_batch_pbar:
        iterator = tqdm(
            iterator,
            total=total_batches,
            desc=f"Train epoch {epoch}",
            position=2,
            leave=False,
            **TQDM_KW,
        )

    for batch_idx, data in enumerate(iterator):
        data = data.to(device)
        # 自动任务：ONeil 为二分类，其余为回归
        if task == "classification":
            y = data.y.view(-1, 1).long().to(device).squeeze(1)
        else:
            y = data.y.view(-1, 1).to(device).float()

        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if show_batch_pbar and (batch_idx % max(1, log_interval) == 0):
            # 动态展示当前 loss
            iterator.set_postfix(loss=f"{loss.item():.6f}")

    if show_batch_pbar and hasattr(iterator, "clear"):
        iterator.clear()  # 确保不残留 batch 进度条行

    avg_loss = total_loss / max(1, total_batches)
    return avg_loss


@torch.no_grad()
def predicting(
        args,
        model,
        device,
        task,
        loader_test,
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

    total_batches = len(loader_test)
    iterator = loader_test

    if show_batch_pbar:
        tag = f"Eval epoch {epoch}" if epoch is not None else "Eval"
        iterator = tqdm(
            iterator,
            total=total_batches,
            desc=tag,
            position=3,
            leave=False,
            **TQDM_KW,
        )

    for data in iterator:
        data = data.to(device)
        output = model(data)

        if task == "classification":
            ys = F.softmax(output, dim=1).to("cpu").data.numpy()
            pred_labels = np.argmax(ys, axis=1).tolist()
            pred_scores = [row[1] for row in ys]  # 正类概率
        else:
            ys = output.squeeze(1).detach().cpu().numpy()
            pred_labels = ys  # 回归：预测值即标签
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


# -----------------------------
# 主流程
# -----------------------------
def main():
    global out_info
    args = parse_args()

    # 1) 固定随机性
    set_seed(args.seed)

    # 2) 设备
    device = build_device(args.device)

    # 3) 数据集加载（两路药物图，保证与原数据组织一致）
    work_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(work_dir, "datas")  # 保持你当前写法
    dataset = args.dataset
    pairs_data = PairDataset(root=data_dir, dataset=dataset)

    # ONeil 数据可能同时包含 cell1/cell2（与 train_ONeil.py 对齐）；其他数据保持兼容
    cell_dim = pairs_data.data.cell1.shape[1]

    length = len(pairs_data)
    log(f"Parameters:  {args}")

    # 设定“组”的定义：二选一（按你的评测目标）
    # drug1_id = drug1_data.data.drug_id
    # drug2_id = drug2_data.data.drug_id

    if args.groups == 'Cell':
        groups = np.asarray(pairs_data.data.cell_id)
        cv = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=args.seed)
    elif args.groups == 'Drug':
        # groups = np.asarray([f'{a} + {b}' for a, b in zip(drug1_id, drug2_id)])
        groups = np.asarray(pairs_data.data.drug1_id)
        cv = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=args.seed)
    else:
        groups = None
        cv = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        print(f"Groups not implement")

    # 结果目录与每折 CSV 初始化
    out_dir = os.path.join(work_dir, "results")
    os.makedirs(out_dir, exist_ok=True)

    # —— K 折外层进度条 —— #
    fold_iter = tqdm(enumerate(cv.split(np.zeros(length), None, groups), 1),
                     desc=f"{args.folds}-Fold CV", position=0, leave=True, **TQDM_KW)

    for fold, (trainval_idx, test_idx) in fold_iter:
        # —— 构造 DataLoader（保持默认不打乱，与原流程一致） —— #
        test_idx = test_idx.astype(np.int64)
        data_test = pairs_data[test_idx]
        loader_test = DataLoader(data_test, batch_size=args.test_batch_size)

        task_type = ("classification" if args.dataset.lower().startswith("o") else "regression")
        if task_type == "classification":
            n_output = 2
        else:
            n_output = 1

        model = FragC3(
            n_output=n_output,
            cell_dim=cell_dim,
            hid_dim=args.hidden,
            heads=args.heads,
            ffn_expansion=args.ffn_expansion,
            use_C3Attn=args.use_C3Attn,
            tri_attn=args.tri_attn,
            tri_variant=args.tri_variant,
            cv_mode=args.cv_mode,
            tokenizer=args.tokenizer,
            dropout=args.dropout,
            cell_agg=args.cell_agg,
            cell_pred=args.cell_pred,
            Lc=args.Lc,
            frag_list=args.frag_list,
            frag_agg=args.frag_agg,
        ).to(device)

        out_info = f'{args.dataset}_{args.encoder}_Frags{"_".join(args.frag_list)}_Group{args.groups}'
        ckpt_path = os.path.join(args.ckpt_dir, f"{out_info}_fold_{fold}_best.pt")
        if not (os.path.exists(ckpt_path) and os.path.isfile(ckpt_path)):
            raise FileNotFoundError(
                f"[CKPT Not Found] Expected checkpoint at:\n  {ckpt_path}\n"
                f"Tips: 先完成该 fold 训练并保存 *_best.pt"
            )
        ckpt = torch.load(ckpt_path, map_location="cpu")
        # robust load: accept either {"model": state_dict} or raw state_dict
        state_dict = ckpt.get("model_state_dict")
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        print(f"[Loaded] ckpt={ckpt_path}")

        eval_fidelity(
            model=model,
            out_info=out_info,
            loader=loader_test,
            device=device,
            task=task_type,
            sparsity_list=[float(x) for x in args.sparsity],
        )

# -----------------------------
# 参数
# -----------------------------
def parse_args():
    """统一管理可调超参数与配置。"""
    p = argparse.ArgumentParser(description="Two-drug graph training & evaluation (K-fold) with tqdm")
    # 训练/评估超参数
    p.add_argument("--train_batch_size", type=int, default=512, help="训练批大小")
    p.add_argument("--test_batch_size", type=int, default=512, help="测试批大小")
    p.add_argument("--lr", type=float, default=1e-4, help="学习率")
    p.add_argument("--epochs", type=int, default=100, help="训练轮数")
    p.add_argument("--early_stopping", type=int, default=20, help="early stopping patience")
    p.add_argument("--log_interval", type=int, default=20, help="训练日志打印间隔（按 batch）")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--folds", type=int, default=5, help="随机划分折数（默认 5）")
    p.add_argument("--groups", type=str, default="Drug", choices=["Cell", "Drug", "none"], help="分组依据")
    # 模型参数
    p.add_argument("--hidden", type=int, default=300, help="隐层维度")
    p.add_argument("--encoder", type=str, default="PRODeepSyn",
                   choices=["FragC3", "SDDS", "MultiSyn", "AttenSyn", "PRODeepSyn",
                            "DeepDDS_GCN", "DeepDDS_GAT", "MatchMaker", "GCN", "GAT"])
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--frag_list", nargs="+", default=["brics", "fg", "murcko"],
                   help='"brics", "fg", "murcko"')
    p.add_argument("--frag_agg", type=str, default="cell_attn",
                   choices=["mlp", "gate", "cell_attn"], help="多视角融合机制")
    # C3Attn参数
    p.add_argument("--use_C3Attn", type=bool, default=True, help="开启Bi2Frag编码")
    p.add_argument("--tri_attn", type=bool, default=True, help="开启cell line注意力")
    p.add_argument("--tri_variant", type=str, default="trilinear",
                   choices=['scale_dot', 'add', 'dot', 'trilinear'])
    p.add_argument("--cv_mode", type=str, default="mul",
                   choices=["mul", "add", "bilinear"])
    p.add_argument("--tokenizer", type=str, default="conv",
                   choices=["conv", "linear"])
    p.add_argument("--heads", type=int, default=4, help="注意力头数")
    p.add_argument("--ffn_expansion", type=int, default=4, help="FFN扩张倍数")
    # p.add_argument("--cell_hid", type=int, default=512)
    p.add_argument("--cell_agg", type=int, default=512)
    p.add_argument("--cell_pred", type=int, default=128)
    p.add_argument("--Lc", type=int, default=128)
    # 数据与设备
    p.add_argument("--dataset", type=str, default="ONeil",
                   choices=["ALMANAC", "DrugComb", "ONeil"], help="数据集前缀名（ONeil 自动切换为二分类）")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="计算设备选择")
    # 可选：是否显示 batch 级进度条（若训练很快可关掉减少开销/刷新）
    p.add_argument("--show_batch_pbar", type=bool, default=False, help="显示 batch 级别 tqdm 进度条")
    # checkpoint加载
    p.add_argument("--ckpt_dir", type=str, default="saves", help="checkpoint加载目录")
    p.add_argument("--sparsity", nargs="+", type=float, default=[0.5, 0.6, 0.7, 0.8, 0.9],
                   help="sparsity levels; keep ratio = 1 - sparsity")
    return p.parse_args()


if __name__ == "__main__":
    main()