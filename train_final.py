import time
import random
import argparse
import numpy as np

from tqdm import tqdm
from sklearn import metrics
from sklearn.model_selection import KFold, GroupShuffleSplit, ShuffleSplit
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    recall_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    accuracy_score,
    precision_score,
    roc_auc_score,
)

from utils import *  # 依赖 save_AUCs 等工具函数（与原流程一致）
from model.FragC3 import *
from dataset import PairDataset
from torch_geometric.loader import DataLoader
from tools.process_folds import process_folds

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
        # 内层：再从 trainval 中“按组留出”验证集
        if args.groups != "none":
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed + fold)
            tv_groups = groups[trainval_idx]
            tv_train_rel, tv_valid_rel = next(gss.split(np.arange(len(trainval_idx)), None, groups=tv_groups))
        else:
            ss = ShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed + fold)
            tv_train_rel, tv_valid_rel = next(ss.split(np.arange(len(trainval_idx)), None, None))
        train_idx = trainval_idx[tv_train_rel]
        valid_idx = trainval_idx[tv_valid_rel]
        # → 得到 train / valid / test 三份互斥且不“拆组”的索引

        # —— 构造 DataLoader（保持默认不打乱，与原流程一致） —— #
        train_idx = train_idx.astype(np.int64)
        valid_idx = valid_idx.astype(np.int64)
        test_idx = test_idx.astype(np.int64)
        data_train = pairs_data[train_idx]
        data_valid = pairs_data[valid_idx]
        data_test = pairs_data[test_idx]

        loader_train = DataLoader(data_train, batch_size=args.train_batch_size, shuffle=True, drop_last=True)
        loader_valid = DataLoader(data_valid, batch_size=args.test_batch_size)
        loader_test = DataLoader(data_test, batch_size=args.test_batch_size)

        task_type = ("classification" if args.dataset.lower().startswith("o") else "regression")
        if task_type == "classification":
            loss_fn = nn.CrossEntropyLoss()
            n_output = 2
        else:
            loss_fn = nn.MSELoss()
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

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # —— 初始化每折结果 CSV —— #
        out_info = (f'{args.dataset}_Group{args.groups}_Frags{"_".join(args.frag_list)}'
                    f'Batch{args.train_batch_size}_Tri{args.tri_variant}_CV{args.cv_mode}_Lc{args.Lc}'
                    f'_H{args.heads}_FFN{args.ffn_expansion}_CA{args.cell_agg}_CP{args.cell_pred}')
        csv_path = os.path.join(out_dir, f"{out_info}_fold_{fold}.csv")

        with open(csv_path, "w") as f:
            if task_type == "classification":
                f.write("Epoch,AUC_dev,PR_AUC,ACC,BACC,PREC,TPR,KAPPA,RECALL,F1\n")
            else:
                f.write("Epoch,MSE,RMSE,MAE\n")

        best_val = (-np.inf if task_type == "classification" else np.inf)
        patience = int(args.early_stopping)
        bad_epochs = 0
        ckpt_path = os.path.join(args.ckpt_dir, f"{out_info}_fold_{fold}_best.pt")

        # —— Epoch 内层进度条 —— #
        epoch_iter = tqdm(
            range(1, args.epochs + 1),
            desc=f"Fold {fold}/{args.folds}",
            position=1,
            leave=False,
            **TQDM_KW,
        )
        for epoch in epoch_iter:
            avg_train_loss = train(
                args=args,
                model=model,
                device=device,
                task=task_type,
                loader_train=loader_train,
                optimizer=optimizer,
                loss_fn=loss_fn,
                epoch=epoch,
                log_interval=args.log_interval,
                show_batch_pbar=args.show_batch_pbar,
            )

            if task_type == "classification":
                T, S, Y = predicting(
                    args, model, device, task_type, loader_valid,
                    show_batch_pbar=args.show_batch_pbar, epoch=epoch
                )
                prev_best = best_val
                best_val, current_val = compute_performance_classification(
                    T=T, S=S, Y=Y, best_auc=best_val, file=csv_path, epoch=epoch
                )
                improved = best_val > prev_best + 1e-12
                epoch_iter.set_postfix(
                    auc=f"{current_val:.4f}",
                    best_auc=f"{best_val:.4f}",
                    train_loss=f"{avg_train_loss:.4f}",
                )
                if improved:
                    bad_epochs = 0
                    save_checkpoint(
                        save_path=ckpt_path,
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        best_val=best_val,
                        args=args,
                        extra={"fold": fold, "out_info": out_info, "task_type": task_type},
                    )
                else:
                    bad_epochs += 1
                if patience > 0 and bad_epochs >= patience:
                    log(f"[Fold {fold}] Early stopping at epoch {epoch}: "
                        f"no improvement for {patience} epochs. best_auc={best_val:.6f}")
                    break
            else:
                T, P, _ = predicting(
                    args, model, device, task_type, loader_valid,
                    show_batch_pbar=args.show_batch_pbar, epoch=epoch
                )
                prev_best = best_val
                best_val, current_val = compute_performance_regression(
                    T, P, best_val, file=csv_path, epoch=epoch
                )
                improved = best_val < prev_best - 1e-12  # 回归：更小更好
                epoch_iter.set_postfix(
                    mse=f"{current_val:.4f}",
                    best_mse=f"{best_val:.4f}",
                    train_loss=f"{avg_train_loss:.4f}"
                )
                if improved:
                    bad_epochs = 0
                    save_checkpoint(
                        save_path=ckpt_path,
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        best_val=best_val,
                        args=args,
                        extra={"fold": fold, "out_info": out_info, "task_type": task_type},
                    )
                else:
                    bad_epochs += 1
                if patience > 0 and bad_epochs >= patience:
                    log(f"[Fold {fold}] Early stopping at epoch {epoch}: "
                        f"no improvement for {patience} epochs. best_mse={best_val:.6f}")
                    break
        # 清理 epoch 进度条，避免残留
        if hasattr(epoch_iter, "clear"):
            epoch_iter.clear()

        # 训练结束：恢复最佳权重 → 评 test（只此一次）
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt["model_state_dict"], strict=True)
            best_val_loaded = float(ckpt.get("best_val", best_val))
            if task_type == "classification":
                log(f"[Fold {fold}] Loaded best checkpoint from: {ckpt_path} (best_auc={best_val_loaded:.6f})")
            else:
                log(f"[Fold {fold}] Loaded best checkpoint from: {ckpt_path} (best_mse={best_val_loaded:.6f})")
        else:
            raise RuntimeError(f"[Fold {fold}] best checkpoint not found: {ckpt_path}")

        if task_type == "classification":
            T, S, Y = predicting(
                args, model, device, task_type, loader_test,
                show_batch_pbar=args.show_batch_pbar
            )
            _, test_auc = compute_performance_classification(
                T=T, S=S, Y=Y, best_auc=best_val, file=csv_path, epoch="test"
            )
            print(f"Test AUC: {test_auc:.4f}")
        else:
            T, P, _ = predicting(
                args, model, device, task_type, loader_test,
                show_batch_pbar=args.show_batch_pbar
            )
            _, test_mse = compute_performance_regression(
                T, P, best_val, file=csv_path, epoch="test"
            )
            print(f"Test MSE: {test_mse:.4f}")
    # —— 全部折完成后：汇总为 Excel —— #
    process_folds(args, out_dir, out_info)


# -----------------------------
# 参数
# -----------------------------
def parse_args():
    """统一管理可调超参数与配置。"""
    p = argparse.ArgumentParser(description="Two-drug graph training & evaluation (K-fold) with tqdm")
    # 训练/评估超参数
    p.add_argument("--train_batch_size", type=int, default=512, help="训练批大小")
    p.add_argument("--test_batch_size", type=int, default=512, help="测试批大小")
    p.add_argument("--lr", type=float, default=2e-4, help="学习率")
    p.add_argument("--epochs", type=int, default=100, help="训练轮数")
    p.add_argument("--early_stopping", type=int, default=20, help="early stopping patience")
    p.add_argument("--log_interval", type=int, default=20, help="训练日志打印间隔（按 batch）")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--folds", type=int, default=5, help="随机划分折数（默认 5）")
    p.add_argument("--groups", type=str, default="Drug", choices=["Cell", "Drug", "none"], help="分组依据")
    # 模型参数
    p.add_argument("--hidden", type=int, default=300, help="隐层维度")
    p.add_argument("--encoder", type=str, default="FragC3",
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
    p.add_argument("--tri_variant", type=str, default="scale_dot",
                   choices=['scale_dot', 'add', 'dot', 'trilinear'])
    p.add_argument("--cv_mode", type=str, default="bilinear",
                   choices=["mul", "add", "bilinear"])
    p.add_argument("--tokenizer", type=str, default="conv",
                   choices=["conv", "linear"])
    p.add_argument("--heads", type=int, default=2, help="注意力头数")
    p.add_argument("--ffn_expansion", type=int, default=8, help="FFN扩张倍数")
    # p.add_argument("--cell_hid", type=int, default=512)
    p.add_argument("--cell_agg", type=int, default=256)
    p.add_argument("--cell_pred", type=int, default=128)
    p.add_argument("--Lc", type=int, default=32)
    # 数据与设备
    p.add_argument("--dataset", type=str, default="ONeil",
                   choices=["ALMANAC", "DrugComb", "ONeil"], help="数据集前缀名（ONeil 自动切换为二分类）")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="计算设备选择")
    # 可选：是否显示 batch 级进度条（若训练很快可关掉减少开销/刷新）
    p.add_argument("--show_batch_pbar", type=bool, default=False, help="显示 batch 级别 tqdm 进度条")
    # checkpoint 保存
    p.add_argument("--ckpt_dir", type=str, default="saves", help="checkpoint保存目录")
    return p.parse_args()


if __name__ == "__main__":
    main()