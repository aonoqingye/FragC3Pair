"""
Quantitative interpretability (Fidelity) evaluation for FragC3:
- Fidelity+ : mask (remove) the Top-K important fragments -> prediction performance should drop
- Fidelity- : keep only the Top-K important fragments -> prediction performance should be largely preserved
"""
from __future__ import annotations

import argparse
import copy
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

import dgl

from dataset import PairDataset
from model.FragC3 import FragC3



# -----------------------------
# small utils (keep script standalone)
# -----------------------------
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_device(device_arg: str) -> torch.device:
    if device_arg and device_arg.lower() in {"cpu", "cuda"}:
        if device_arg.lower() == "cuda" and torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")
    # auto
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


# -----------------------------
# metrics
# -----------------------------
def _safe_div(a: float, b: float, eps: float = 1e-12) -> float:
    return float(a) / float(b + eps)


def compute_auc_aupr(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    # lazy import so script still runs for regression envs without sklearn
    from sklearn.metrics import roc_auc_score, average_precision_score
    auc = float(roc_auc_score(y_true, y_score))
    aupr = float(average_precision_score(y_true, y_score))
    return auc, aupr


def compute_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


# -----------------------------
# Top-K selection from attention
# -----------------------------
def attn_to_importance(attn: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      imp_A: [B, L_A]
      imp_B: [B, L_B]
    """
    b2a = attn["B2A"]  # [B,H,LA,LB]
    a2b = attn["A2B"]  # [B,H,LB,LA]
    # importance of B fragments: sum over heads and A queries
    imp_B = b2a.sum(dim=1).sum(dim=1)  # [B,LB]
    # importance of A fragments: sum over heads and B queries
    imp_A = a2b.sum(dim=1).sum(dim=1)  # [B,LA]
    return imp_A, imp_B


def topk_indices_per_sample(
    imp: torch.Tensor,
    sizes: torch.Tensor,
    k: int,
    largest: bool = True,
) -> List[torch.Tensor]:
    """
    imp: [B, Lmax] (may include padded positions)
    sizes: [B] number of valid positions for each sample
    returns list of length B; each element is a 1D LongTensor of indices (local positions)
    """
    out: List[torch.Tensor] = []
    B = imp.shape[0]
    for i in range(B):
        n = int(sizes[i].item())
        n = max(1, n)
        kk = min(max(1, k), n)
        scores = imp[i, :n]
        idx = torch.topk(scores, k=kk, largest=largest).indices
        out.append(idx)
    return out


# -----------------------------
# Graph editing helpers (DGL)
# -----------------------------
def _clone_graph_list(graphs: List[dgl.DGLHeteroGraph]) -> List[dgl.DGLHeteroGraph]:
    # DGLGraph.clone() is shallow for python objects but copies structure + features, which we want.
    return [g.clone() for g in graphs]


def _remove_p_nodes(
    graphs: List[dgl.DGLHeteroGraph],
    remove_idx_list: List[torch.Tensor],
) -> List[dgl.DGLHeteroGraph]:
    """
    Remove fragment nodes (ntype='p') for each graph in graphs according to local indices.
    """
    assert len(graphs) == len(remove_idx_list)
    new_graphs: List[dgl.DGLHeteroGraph] = []
    for g, rm in zip(graphs, remove_idx_list):
        # rm 必须与图在同一设备，否则 DGL 会报 device 不一致
        rm = rm.detach().to(g.device)
        if rm.numel() == 0:
            new_graphs.append(g)
            continue

        n_p = g.num_nodes("p")
        rm = rm[(rm >= 0) & (rm < n_p)]
        # Ensure at least 1 fragment node remains (some components assume non-empty)
        if rm.numel() >= n_p:
            rm = rm[: max(0, n_p - 1)]
        if rm.numel() == 0:
            new_graphs.append(g)
            continue

        g2 = dgl.remove_nodes(g, rm, ntype="p")
        # 确保边特征 schema 一致：当边为空时，显式设定形状 (0, feat_dim)
        etype = ("p", "r", "p")
        if etype in g2.etypes:
            # 确定特征维度
            if g.num_edges(etype) > 0 and etype in g.edata and "x" in g.edata[etype]:
                if g.edata[etype]["x"].ndim == 2:
                    feat_dim = g.edata[etype]["x"].shape[1]
                else:
                    feat_dim = 34  # 默认 BrICS/Ring 边特征维度
            else:
                feat_dim = 34  # 默认 BrICS/Ring 边特征维度
            
            # 处理边特征
            if g2.num_edges(etype) == 0:
                # 空边：确保维度为 (0, feat_dim)
                if etype not in g2.edata:
                    g2.edata[etype] = {}
                g2.edata[etype]["x"] = torch.empty((0, feat_dim), dtype=torch.float32, device=g2.device)
            elif etype in g2.edata and "x" in g2.edata[etype]:
                # 非空边：确保维度为 (num_edges, feat_dim)
                if g2.edata[etype]["x"].ndim == 1:
                    g2.edata[etype]["x"] = g2.edata[etype]["x"].reshape(-1, feat_dim)
                elif g2.edata[etype]["x"].ndim == 2 and g2.edata[etype]["x"].shape[1] != feat_dim:
                    # 如果维度不匹配，可能需要截断或填充（这里假设是截断）
                    current_dim = g2.edata[etype]["x"].shape[1]
                    if current_dim > feat_dim:
                        g2.edata[etype]["x"] = g2.edata[etype]["x"][:, :feat_dim]
                    else:
                        padding = torch.zeros((g2.edata[etype]["x"].shape[0], feat_dim - current_dim), 
                                             dtype=torch.float32, device=g2.device)
                        g2.edata[etype]["x"] = torch.cat([g2.edata[etype]["x"], padding], dim=1)
        new_graphs.append(g2)
    return new_graphs


def build_masked_views(
    data,
    view_to_remove: Dict[str, Dict[str, List[torch.Tensor]]],
) -> object:
    """
    Returns a shallow-copied data object with mask information instead of removing nodes.
    This avoids DGL schema inconsistency issues.
    view_to_remove: {view: {"A": [idx...], "B": [idx...]}}
    
    Instead of removing nodes, we store mask information in the data object,
    which will be used to mask attention in the model.
    """
    d2 = copy.copy(data)

    # Initialize mask dictionaries if they don't exist
    if not hasattr(d2, 'frag_masks'):
        d2.frag_masks = {}
    
    for view, sides in view_to_remove.items():
        # Store mask information instead of removing nodes
        # The mask will be applied to valid_a/valid_b in the model
        d2.frag_masks[view] = {
            "A": sides["A"],  # List of tensors, each tensor contains indices to mask
            "B": sides["B"]
        }
    
    # Keep original graphs unchanged (no node removal)
    # The model will use frag_masks to mask attention

    return d2


# -----------------------------
# evaluation loop
# -----------------------------
@torch.no_grad()
def eval_fidelity(
    model: torch.nn.Module,
    out_info: str,
    loader: DataLoader,
    device: torch.device,
    task: str,
    sparsity_list: List[float],
):
    """
    For each sparsity s in sparsity_list:
      K = ceil((1 - s) * L)  (keep ratio)
      Fidelity+  : remove Top-K, compute metric_drop
      Fidelity-  : keep only Top-K, compute metric_keep
    """
    rows = []
    model.eval()

    # ----- baseline predictions -----
    y_true_all: List[float] = []
    base_score_all: List[float] = []  # probability for cls; value for reg

    # Cache per-batch attention to avoid re-forward baseline multiple times per sparsity
    cached_batches = []

    for data in loader:
        data = data.to(device)

        out, extra_per_view, _ = model(data, return_attn=True)

        if task == "classification":
            prob = F.softmax(out, dim=1)[:, 1].detach().cpu()  # [B]
            base_score = prob.numpy()
            y_true = data.y.view(-1).detach().cpu().numpy().astype(int)
        else:
            base_score = out.view(-1).detach().cpu().numpy()
            y_true = data.y.view(-1).detach().cpu().numpy()

        y_true_all.append(y_true)
        base_score_all.append(base_score)

        cached_batches.append((data, extra_per_view, base_score, y_true))

    y_true_all_np = np.concatenate(y_true_all, axis=0)
    base_score_all_np = np.concatenate(base_score_all, axis=0)

    if task == "classification":
        base_auc, base_aupr = compute_auc_aupr(y_true_all_np, base_score_all_np)
        print(f"[Baseline] AUROC={base_auc:.4f}  AUPR={base_aupr:.4f}")
    else:
        base_mse = compute_mse(y_true_all_np, base_score_all_np)
        print(f"[Baseline] MSE={base_mse:.6f}")

    # ----- sparsity loop -----
    for s in sparsity_list:
        assert 0.0 < s < 1.0, f"sparsity should be in (0,1), got {s}"
        masked_scores_all = []
        kept_scores_all = []
        y_true_all2 = []

        for (data, extra_per_view, base_score, y_true) in cached_batches:
            B = len(y_true)
            view_to_remove_mask = {}
            view_to_remove_keep = {}

            # For each view, compute Top-K separately (fragments differ across views)
            for view, attn_data in extra_per_view.items():
                # attn_data 可能是两种格式：
                # 1. 新格式: {"attn": {"B2A":..., "A2B":...}, "valid_a": ..., "valid_b": ...}
                # 2. 旧格式: {"B2A":..., "A2B":...} 或 None
                if attn_data is None:
                    continue
                
                # 提取注意力字典
                if isinstance(attn_data, dict) and "attn" in attn_data:
                    attn = attn_data["attn"]  # 新格式
                else:
                    attn = attn_data  # 旧格式
                
                if attn is None or not isinstance(attn, dict):
                    continue

                # We need sizes (#fragment nodes) for each sample in the current batch under THIS view.
                # Use the actual DGL graphs, not the padded length.
                if view == "brics":
                    sizes_A = torch.tensor([g.num_nodes("p") for g in data.graph1], device=device)
                    sizes_B = torch.tensor([g.num_nodes("p") for g in data.graph2], device=device)
                elif view == "fg":
                    sizes_A = torch.tensor([g.num_nodes("p") for g in data.graph1_fg], device=device)
                    sizes_B = torch.tensor([g.num_nodes("p") for g in data.graph2_fg], device=device)
                elif view == "murcko":
                    sizes_A = torch.tensor([g.num_nodes("p") for g in data.graph1_murcko], device=device)
                    sizes_B = torch.tensor([g.num_nodes("p") for g in data.graph2_murcko], device=device)
                else:
                    raise KeyError(view)

                # derive importance from attention
                imp_A, imp_B = attn_to_importance(attn)

                # K is based on keep ratio (1 - sparsity)
                # Use Lmax from imp shape, but clamp per sample by sizes_{A,B}.
                # Here we use K computed from the *max* fragment count of this view in batch to keep
                # "sparsity" roughly comparable; still clamped per-sample.
                Lmax_A = int(imp_A.shape[1])
                Lmax_B = int(imp_B.shape[1])
                K_A = max(1, int(np.ceil((1.0 - s) * max(1, Lmax_A))))
                K_B = max(1, int(np.ceil((1.0 - s) * max(1, Lmax_B))))

                topA = topk_indices_per_sample(imp_A, sizes_A, K_A, largest=True)
                topB = topk_indices_per_sample(imp_B, sizes_B, K_B, largest=True)

                # mask-topK: remove topK indices
                view_to_remove_mask[view] = {"A": topA, "B": topB}

                # keep-only: remove everything except topK
                keep_rm_A = []
                keep_rm_B = []
                for i in range(B):
                    nA = int(sizes_A[i].item())
                    nB = int(sizes_B[i].item())
                    setA = set(topA[i].detach().cpu().tolist())
                    setB = set(topB[i].detach().cpu().tolist())
                    rmA = torch.tensor([j for j in range(nA) if j not in setA], dtype=torch.long)
                    rmB = torch.tensor([j for j in range(nB) if j not in setB], dtype=torch.long)
                    keep_rm_A.append(rmA)
                    keep_rm_B.append(rmB)
                view_to_remove_keep[view] = {"A": keep_rm_A, "B": keep_rm_B}

            # build masked/kept data and forward
            data_masked = build_masked_views(data, view_to_remove_mask).to(device)
            data_kept = build_masked_views(data, view_to_remove_keep).to(device)

            out_masked = model(data_masked)
            out_kept = model(data_kept)

            if task == "classification":
                masked_scores = F.softmax(out_masked, dim=1)[:, 1].detach().cpu().numpy()
                kept_scores = F.softmax(out_kept, dim=1)[:, 1].detach().cpu().numpy()
            else:
                masked_scores = out_masked.view(-1).detach().cpu().numpy()
                kept_scores = out_kept.view(-1).detach().cpu().numpy()

            masked_scores_all.append(masked_scores)
            kept_scores_all.append(kept_scores)
            y_true_all2.append(y_true)

        y_true_np = np.concatenate(y_true_all2, axis=0)
        masked_np = np.concatenate(masked_scores_all, axis=0)
        kept_np = np.concatenate(kept_scores_all, axis=0)

        if task == "classification":
            m_auc, m_aupr = compute_auc_aupr(y_true_np, masked_np)
            k_auc, k_aupr = compute_auc_aupr(y_true_np, kept_np)

            # Fidelity+ : bigger is better (more drop after masking Top-K)
            # Fidelity- : bigger is better (better preserved after keeping Top-K)
            # For "higher is better" metrics (AUROC/AUPR):
            f_plus_auc = _safe_div((base_auc - m_auc), base_auc)
            f_minus_auc = _safe_div(k_auc, base_auc)

            f_plus_aupr = _safe_div((base_aupr - m_aupr), base_aupr)
            f_minus_aupr = _safe_div(k_aupr, base_aupr)

            print(
                f"[sparsity={s:.2f}] "
                f"masked AUROC={m_auc:.4f} AUPR={m_aupr:.4f} | "
                f"kept AUROC={k_auc:.4f} AUPR={k_aupr:.4f} | "
                f"F+ (AUROC)={f_plus_auc:.3f}  F- (AUROC)={f_minus_auc:.3f}"
            )

            rows.append({
                "sparsity": s,
                "baseline_auroc": base_auc,
                "baseline_aupr": base_aupr,
                "masked_auroc": m_auc,
                "masked_aupr": m_aupr,
                "kept_auroc": k_auc,
                "kept_aupr": k_aupr,
                "fidelity_plus_auroc": f_plus_auc,
                "fidelity_minus_auroc": f_minus_auc,
                "fidelity_plus_aupr": f_plus_aupr,
                "fidelity_minus_aupr": f_minus_aupr,
            })

        else:
            m_mse = compute_mse(y_true_np, masked_np)
            k_mse = compute_mse(y_true_np, kept_np)

            # For "lower is better" metric (MSE):
            # Fidelity+ : (masked - base)/base (increase after masking Top-K)
            # Fidelity- : base/kept (how close kept is to base, upper bounded near 1 when equal)
            f_plus = _safe_div((m_mse - base_mse), base_mse)
            f_minus = _safe_div(base_mse, k_mse)

            print(
                f"[sparsity={s:.2f}] "
                f"masked MSE={m_mse:.6f} | kept MSE={k_mse:.6f} | "
                f"F+={f_plus:.3f}  F-={f_minus:.3f}"
            )

            rows.append({
                "sparsity": s,
                "baseline_mse": base_mse,
                "masked_mse": m_mse,
                "kept_mse": k_mse,
                "fidelity_plus": f_plus,
                "fidelity_minus": f_minus,
            })

    # write csv
    import pandas as pd
    df = pd.DataFrame(rows)
    out_csv = os.path.join("FSE", f"{out_info}_fidelity.csv")
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[Saved] {out_csv}")