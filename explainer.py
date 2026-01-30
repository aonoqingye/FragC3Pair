import io
import os
import csv
import math
import json
import torch
import matplotlib
import numpy as np

from tools.fragment_namer import describe_fragment
from tools.fg_rules_loader import build_fg_smart_db
from prepare_onco_data import GetFragmentFeats, _get_mol_fragment_sets_fg, _get_mol_fragment_sets_murcko, \
    _get_mol_fragment_sets_ringpaths
from tools.prompter import _call_llm

# --- 固定非交互后端，必须在 pyplot 之前 ---
matplotlib.use("Agg")

from rdkit import Chem
from rdkit.Chem import AllChem
from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib.patches import FancyArrowPatch
from rdkit.Chem.Draw import rdMolDraw2D
from typing import Dict, List, Tuple, Optional
from functools import lru_cache
from collections import defaultdict

FG_RULES = None  # 懒加载

# =========================
# 基础：加载 drug_id -> SMILES
# =========================
def load_id2smiles(csv_path: str, drug_col: str = "drug", smi_col: str = "smiles") -> Dict[str, str]:
    """从CSV加载 drug_id->SMILES 映射（一次性读取，去除空白行）。"""
    mapping = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = str(row.get(drug_col, "")).strip()
            val = str(row.get(smi_col, "")).strip()
            if key and val:
                mapping[key] = val
    return mapping


# =========================
# BRICS 片段成员（与你数据口径一致）
# =========================
def _invert_atom2frag(atom2frag: Dict[int, int]) -> Dict[int, List[int]]:
    frag2atoms: Dict[int, List[int]] = {}
    for a, p in atom2frag.items():
        frag2atoms.setdefault(p, []).append(a)
    return frag2atoms


@lru_cache(maxsize=4096)
def _brics_members_from_smiles(smiles: str, view_order) -> Tuple[Chem.Mol, Dict[int, List[int]]]:
    """返回 RDKit Mol(去氢) 与 frag_id -> [atom_idx] 的成员映射（缓存加速重复SMILES）。"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    # 生成2D坐标（先加氢再去氢通常能得到更稳定的布局）
    mol_H = Chem.AddHs(mol)
    AllChem.Compute2DCoords(mol_H)
    mol_noH = Chem.RemoveHs(mol_H)
    AllChem.Compute2DCoords(mol_noH)

    # BRICS：沿用 GetFragmentFeats
    frag2atoms = {}
    for view in view_order:
        if view == "brics":
            atom2frag, _ = GetFragmentFeats(mol)  # {atom_idx: frag_id}
        else:
            atom2frag = {}
            if view == "fg":
                sets, edges = _get_mol_fragment_sets_fg(mol) or []
            elif view == "murcko":
                sets, edges = _get_mol_fragment_sets_murcko(mol) or []
            elif view == "ringpaths":
                sets, edges = _get_mol_fragment_sets_ringpaths(mol)
            for fi, aset in enumerate(sets):
                for a in aset:
                    atom2frag[a] = fi
        frag2atoms[view] = _invert_atom2frag(atom2frag)

    return mol_noH, frag2atoms


# =========================
# 注意力工具
# =========================
def _reduce_heads(attn_3d: torch.Tensor, how: str = "mean") -> torch.Tensor:
    """[H, Lq, Lk] -> [Lq, Lk]"""
    if attn_3d.dim() != 3:
        raise ValueError(f"attn must be [H,Lq,Lk], got shape={tuple(attn_3d.shape)}")
    return attn_3d.max(dim=0).values if how == "max" else attn_3d.mean(dim=0)


def _crop_valid(A2B_3d: torch.Tensor, valid_q: torch.Tensor, valid_k: torch.Tensor) -> torch.Tensor:
    """裁剪掉 padding：A2B_3d:[H,Lq,Lk], valid_*:[Lq]/[Lk](bool)"""
    H, Lq, Lk = A2B_3d.shape
    Lq_real = int(valid_q.sum().item())
    Lk_real = int(valid_k.sum().item())
    return A2B_3d[:, :Lq_real, :Lk_real]


def _topk_edges(W: np.ndarray, k: int) -> List[Tuple[int, int, float]]:
    """从矩阵 W 中选出 |权重| 最大的 k 条边（返回 (row, col, value) 列表，按 |value| 降序）"""
    flat = W.ravel()
    if flat.size == 0 or k <= 0:
        return []
    k = int(min(k, flat.size))
    idx = np.argsort(-np.abs(flat))[:k]
    Lq, Lk = W.shape
    return [(int(id_ // Lk), int(id_ % Lk), float(W[int(id_ // Lk), int(id_ % Lk)])) for id_ in idx]


def _to_numpy(x):
    import numpy as _np
    import torch as _torch
    if x is None:
        return None
    if isinstance(x, _torch.Tensor):
        return x.detach().cpu().numpy()
    return _np.asarray(x)

def _norm01(arr_like, eps=1e-12):
    import numpy as _np
    arr = _np.asarray(list(arr_like), dtype=float)
    if arr.size == 0:
        return []
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < eps:
        return [0.6] * arr.size   # 退化时给个中性值，避免全 0
    return ((arr - lo) / (hi - lo)).tolist()


# =========================
# 用 RDKit Drawer 的像素坐标：绘分子图并返回片段原子屏幕坐标 + 质心 + 像素缩放
# =========================
def _draw_mol_and_centroids_on_ax(
    ax, mol: Chem.Mol, frag2atoms: Dict[int, List[int]], box, canvas: int = 800
) -> Tuple[np.ndarray, Dict[int, np.ndarray], float, float]:
    """
    返回：
      - pts_centroid: [num_frags, 2]
      - frag2xy_screen: {frag_id: [[x,y], ...]} 屏幕坐标（与 ax 相同坐标系）
      - sx, sy: 每个像素在 x/y 方向上对应的轴坐标长度
    """
    w = h = int(canvas)
    d2d = rdMolDraw2D.MolDraw2DCairo(w, h)
    rdMolDraw2D.PrepareMolForDrawing(mol)
    d2d.DrawMolecule(mol)
    d2d.FinishDrawing()

    png = d2d.GetDrawingText()
    img = plt.imread(io.BytesIO(png), format="png")

    x0, x1, y0, y1 = box
    ax.imshow(img, extent=[x0, x1, y0, y1], zorder=0)

    # 像素 -> 轴坐标缩放系数
    sx = (x1 - x0) / w
    sy = (y1 - y0) / h

    pts = []
    frag2xy_screen: Dict[int, np.ndarray] = {}
    for pid, atoms in frag2atoms.items():
        if not atoms:
            pts.append([np.nan, np.nan])
            frag2xy_screen[pid] = np.zeros((0, 2), dtype=float)
            continue
        xy = []
        for a in atoms:
            p = d2d.GetDrawCoords(int(a))  # 像素坐标原点左上
            sx_ax = x0 + p.x * sx
            sy_ax = y0 + (h - p.y) * sy
            xy.append([sx_ax, sy_ax])
        xy = np.asarray(xy, dtype=float)
        c = xy.mean(axis=0)
        pts.append([c[0], c[1]])
        frag2xy_screen[pid] = xy
    return np.asarray(pts, dtype=float), frag2xy_screen, sx, sy


# =========================
# 几何辅助
# =========================
def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Andrew's monotone chain，points: [N,2] -> hull 顺时针（去重+排序）"""
    pts = np.unique(points, axis=0)
    n = len(pts)
    if n <= 2:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))
    hull = np.array(lower[:-1] + upper[:-1], dtype=float)
    return hull



def _inflate_polygon_px(poly_ax: np.ndarray, margin_px: float, sx: float, sy: float) -> np.ndarray:
    """在像素空间外扩 margin_px（>0 向外，<0 向内），再换算回轴坐标。poly_ax 为轴坐标顶点。

    关键修复：保证法线方向为“外侧”。做法：在像素空间下计算每个顶点的平均法线，
    并用 (p - centroid) 的像素向量与法线点积判断方向；若为负则翻转法线方向。
    """
    if len(poly_ax) <= 2 or abs(margin_px) <= 0:
        return poly_ax

    # 轴坐标 -> 像素坐标的辅助
    def ax_to_px(v_ax: np.ndarray) -> np.ndarray:
        return np.array([v_ax[0] / sx, v_ax[1] / sy], dtype=float)

    def px_to_ax(v_px: np.ndarray) -> np.ndarray:
        return np.array([v_px[0] * sx, v_px[1] * sy], dtype=float)

    # 顶点像素法线（左法线）
    def left_normal_px(v_px: np.ndarray) -> np.ndarray:
        n = np.array([-v_px[1], v_px[0]], dtype=float)
        l = np.linalg.norm(n) + 1e-9
        return n / l

    centroid_ax = np.mean(poly_ax, axis=0)
    centroid_px = ax_to_px(centroid_ax)

    out = []
    k = len(poly_ax)
    for i in range(k):
        p_prev_ax = poly_ax[(i - 1) % k]
        p_ax = poly_ax[i]
        p_next_ax = poly_ax[(i + 1) % k]

        # 邻边向量（像素尺度）
        v1_px = ax_to_px(p_ax - p_prev_ax)
        v2_px = ax_to_px(p_next_ax - p_ax)

        # 邻边法线并求平均（像素尺度）
        n1_px = left_normal_px(v1_px)
        n2_px = left_normal_px(v2_px)
        n_avg_px = n1_px + n2_px
        ln = np.linalg.norm(n_avg_px)
        if ln < 1e-9:
            n_avg_px = n1_px
        else:
            n_avg_px = n_avg_px / ln

        # 保证“外侧”：与 (p - centroid) 在像素空间的点积应为正
        to_vert_px = ax_to_px(p_ax - centroid_ax)
        if float(np.dot(n_avg_px, to_vert_px)) < 0:
            n_avg_px = -n_avg_px

        delta_ax = px_to_ax(n_avg_px * margin_px)
        out.append(p_ax + delta_ax)

    return np.array(out, dtype=float)


def _smooth_polygon(poly_ax: np.ndarray, iters: int = 2) -> np.ndarray:
    """Chaikin 平滑"""
    if len(poly_ax) < 3 or iters <= 0:
        return poly_ax
    P = poly_ax.copy()
    for _ in range(iters):
        Q = []
        n = len(P)
        for i in range(n):
            p = P[i]
            q = P[(i + 1) % n]
            Q.append(0.75 * p + 0.25 * q)
            Q.append(0.25 * p + 0.75 * q)
        P = np.asarray(Q, dtype=float)
    return P


# =========================
# 片段覆盖层：统一颜色 + 权重控制显著性
# =========================
def _overlay_fragments_on_ax(
    ax,
    frag2xy_screen: Dict[int, np.ndarray],
    sx: float,
    sy: float,
    base_color: str,
    frag_weights: Optional[Dict[int, float]] = None,
    alpha_min: float = 0.12,
    alpha_max: float = 0.85,
    sat_min: float = 0.35,
):
    """
    仅绘制“填充型”覆盖层：
      - 单/双原子：填充椭圆；
      - >=3 原子：凸包 -> 像素外扩(0) -> Chaikin 平滑。
    颜色饱和度与透明度按 frag_weights 映射（权重越大越显著）。
    """
    import matplotlib.colors as mcolors

    if frag_weights is None:
        frag_weights = {}

    # 像素级尺寸常量（可按需微调）
    PX_E_RAD   = 32.0
    PX_E_MIN_W = 42.0
    PX_E_MIN_H = 32.0

    def mix_with_white(rgb, s: float):
        r, g, b = rgb
        return (1 - (1 - r) * s, 1 - (1 - g) * s, 1 - (1 - b) * s)

    def pxw(wpx): return wpx * sx
    def pxh(hpx): return hpx * sy

    base_rgb = mcolors.to_rgb(base_color)

    for pid, xy in frag2xy_screen.items():
        if xy.shape[0] == 0:
            continue
        w = float(frag_weights.get(pid, 0.0))
        if not math.isfinite(w):
            w = 0.0
        w = min(1.0, max(0.0, w))

        fill_alpha = alpha_min + (alpha_max - alpha_min) * w
        sat        = sat_min + (1.0 - sat_min) * w
        fill_rgb   = mix_with_white(base_rgb, sat)

        if xy.shape[0] == 1:
            cx, cy = xy[0]
            e = patches.Ellipse((cx, cy), width=2 * pxw(PX_E_RAD), height=2 * pxh(PX_E_RAD),
                                facecolor=fill_rgb, edgecolor="none", linewidth=0.0, alpha=fill_alpha)
            ax.add_patch(e)
            continue

        if xy.shape[0] == 2:
            p1, p2 = xy[0], xy[1]
            cx, cy = (p1 + p2) / 2
            d_ax = float(np.linalg.norm(p2 - p1))
            d_px = d_ax / max(sx, 1e-9)
            rx_px = max(PX_E_MIN_W / 2, d_px * 1.2)
            ry_px = max(PX_E_MIN_H / 2, d_px * 0.6)
            e = patches.Ellipse((cx, cy), width=2 * pxw(rx_px), height=2 * pxh(ry_px),
                                facecolor=fill_rgb, edgecolor="none", linewidth=0.0, alpha=fill_alpha)
            ax.add_patch(e)
            continue

        hull = _convex_hull(xy)
        if hull.shape[0] >= 3:
            inflated = _inflate_polygon_px(hull, margin_px=36.0, sx=sx, sy=sy)
            smooth = _smooth_polygon(inflated, iters=2)
            poly = patches.Polygon(smooth, closed=True, facecolor=fill_rgb,
                                   edgecolor="none", linewidth=0.0, alpha=fill_alpha)
            ax.add_patch(poly)
            continue

        # 兜底
        cx, cy = xy.mean(axis=0)
        e = patches.Ellipse((cx, cy), width=pxw(44), height=pxh(32),
                            facecolor=fill_rgb, edgecolor="none", linewidth=0.0, alpha=fill_alpha)
        ax.add_patch(e)

def extract_target_and_pred(y_i: torch.Tensor, out_i: torch.Tensor) -> Tuple[float, float, float, float]:
    """回归显示：返回 (y_true, y_pred, |err|, |rel_err|)。"""
    y_true = float(y_i.detach().cpu())
    y_pred = float(out_i.detach().cpu())
    abs_err = abs(y_pred - y_true)
    rel_err = abs_err / (abs(y_true) + 1e-8)
    return y_true, y_pred, abs_err, rel_err

# =========================
# 主函数：并排双分子 + 双向注意力连线 + 片段高亮
# =========================
def save_cross_attn_pairlines(
    y: torch.Tensor,
    output: torch.Tensor,
    extra_per_view: dict,             # 期望 {view: {"attn": {"B2A": [B,H,La,Lb], "A2B": ...}, "valid_a": [B,La], "valid_b": [B,Lb]}}
    agg_attn: dict,
    cell_id: List[str],
    drug_id_a: List[str],
    drug_id_b: List[str],
    idx,
    visualize_n: int = 4,
    top_k: int = 10,
    with_cell: bool = False,
    out_dir: str = "attn_pair",
    smiles_csv: str = "datas/data/drug_id.csv",
    smiles_drug_col: str = "drug",
    smiles_smi_col: str = "smiles",
    head_reduce: str = "mean",
    # === 新增：Prompt 导出控制 ===
    call_llm: bool = True,
    model: str = 'gpt-4',
    prompt_thresholds: Dict = None,          # e.g., {"style":"one_sided","synergistic_if":-5.0}
    prompt_external_knowledge: str = "disallowed"
):
    """
    函数功能未变：
      - 并排绘制两个分子；
      - 以片段为节点，绘制 A→B / B→A 的 top-k 注意力连线（带权重、箭头、轻微错位）；
      - 在两侧分子上用统一基色填充片段区域，显著性与注意力强度一致；
      - 标题包含 y 与 ŷ。

    新增：
      - write_prompt=True 时，同时输出一份与单样本同名的 .prompt.txt，内容为已定制好的 LLM Prompt；
      - write_json 可控制是否继续写 JSON（默认保留，确保向后兼容）。
    """
    os.makedirs(out_dir, exist_ok=True)
    smiles_map = load_id2smiles(smiles_csv, smiles_drug_col, smiles_smi_col)

    # 兼容：如果传入的其实是旧格式（单视角），转成 dict 形式
    if isinstance(extra_per_view, dict) and "attn" in extra_per_view and "valid_a" in extra_per_view:
        extra_per_view = {"brics": extra_per_view}

    def _reduce_and_crop(B2A_i, A2B_i, va_i, vb_i):
        B2A_crop = _crop_valid(B2A_i, va_i, vb_i)
        A2B_crop = _crop_valid(A2B_i, vb_i, va_i)
        B2A_2d = _reduce_heads(B2A_crop, head_reduce).detach().cpu().numpy()
        A2B_2d = _reduce_heads(A2B_crop, head_reduce).detach().cpu().numpy()
        return B2A_2d, A2B_2d

    # ---- 画带箭头的有向连线 ----
    def _arrow(ax, x1, y1, x2, y2, color, wn, offset_sign):
        # 轻微上下错位
        dy = 0.005 * offset_sign
        y1o, y2o = y1 + dy, y2 + dy
        lw = 0.5 + 2.5 * wn
        alpha = 0.3 + 0.6 * wn
        arr = FancyArrowPatch(
            (x1, y1o), (x2, y2o),
            arrowstyle='-|>',
            mutation_scale=max(8.0, 10.0 * wn),   # 箭头尺寸随权重放大（设下限避免太小）
            linewidth=lw,
            color=color,
            alpha=alpha,
            zorder=5,
            shrinkA=3.0, shrinkB=3.0,     # 轻微收缩，避免扎在点心上
            connectionstyle="arc3,rad=0.0",
        )
        ax.add_patch(arr)
        # 在线段中点标注权重（两位小数）
        midx, midy = (x1 + x2) / 2, (y1o + y2o) / 2
        ax.text(midx, midy, f"{wn:.2f}", fontsize=7, color=color, alpha=0.85, ha="center")

    n = min(int(visualize_n), len(drug_id_a))

    view_order = []
    for k in extra_per_view.keys():
        view_order.append(k)

    group_ae = []
    group_re = []
    for i in range(n):
        ida, idb = str(drug_id_a[i]), str(drug_id_b[i])
        idc = str(cell_id[i]) if cell_id else ""
        smi_a, smi_b = smiles_map.get(ida), smiles_map.get(idb)
        if (smi_a is None) or (smi_b is None):
            continue

        # ===== 根据视角生成片段成员 =====
        mol_a, frag2atoms_a = _brics_members_from_smiles(smi_a, tuple(view_order))
        mol_b, frag2atoms_b = _brics_members_from_smiles(smi_b, tuple(view_order))

        # 画布：每个视角一行（高度 800 像素左右，宽度 1600）
        rows = max(1, len(view_order))
        fig = plt.figure(figsize=(1600 / 200, rows * 800 / 200), dpi=200)
        fig.subplots_adjust(hspace=0.28)

        # ====== 准备 sample_dict（供 JSON/Prompt 两用）======
        y_true, y_pred, abs_err, rel_err = extract_target_and_pred(y[i], output[i])
        group_ae.append(abs_err)
        group_re.append(rel_err)
        sample_dict = {
            "task": "regression",
            "y_true": y_true, "y_pred": y_pred, "abs_err": abs_err, "rel_err": rel_err,
            "cell_line": idc,
            "drugA": {"id": ida, "smiles": smi_a},
            "drugB": {"id": idb, "smiles": smi_b},
            "views": {}
        }
        global FG_RULES
        if FG_RULES is None:
            FG_RULES = build_fg_smart_db()  # 读 RDConfig.RDDataDir 下两文件

        def draw_one_view(i: int, row_idx: int, view_key: str):
            sub = extra_per_view[view_key]
            attn_dict = sub.get("attn", {})
            valid_a = sub.get("valid_a", None)
            valid_b = sub.get("valid_b", None)
            # 统一成 [B,H,Lq,Lk] 形状（这里只取单向互注意力）
            B2A = attn_dict["B2A"]
            A2B = attn_dict["A2B"]
            if B2A.dim() == 3:
                B2A = B2A.unsqueeze(0)
                A2B = A2B.unsqueeze(0)
                valid_a = valid_a.unsqueeze(0)
                valid_b = valid_b.unsqueeze(0)

            B2A_i, A2B_i = B2A[i], A2B[i]
            va_i, vb_i = valid_a[i].bool().cpu(), valid_b[i].bool().cpu()
            B2A_2d, A2B_2d = _reduce_and_crop(B2A_i, A2B_i, va_i, vb_i)
            if (B2A_2d is None) and (A2B_2d is None):
                return

            # 行面板
            top = 1.0 - (row_idx + 1) / rows
            height = 1.0 / rows - 0.01
            ax = fig.add_axes([0, top, 1, height])
            ax.set_axis_off()

            # 左右绘制区域（轴坐标）
            L = (0.02, 0.48, 0.20, 0.65)
            R = (0.52, 0.98, 0.20, 0.65)
            Pa, frag2xy_a, sx_a, sy_a = _draw_mol_and_centroids_on_ax(ax, mol_a, frag2atoms_a[view_key], L, canvas=800)
            Pb, frag2xy_b, sx_b, sy_b = _draw_mol_and_centroids_on_ax(ax, mol_b, frag2atoms_b[view_key], R, canvas=800)

            # === 覆盖强度权重（由已绘连线推导） ===
            pid_order_a = list(frag2xy_a.keys())
            pid_order_b = list(frag2xy_b.keys())

            maskA = va_i.cpu().numpy().astype(bool)
            maskB = vb_i.cpu().numpy().astype(bool)
            idx_a = np.flatnonzero(maskA)
            idx_b = np.flatnonzero(maskB)
            if idx_a.size == 0 or idx_b.size == 0:
                plt.close(fig)
                print(f"Invalid Masks in idx{idx[i]} view {view_key}: maskA:{idx_a} maskB:{idx_b}")

            Pa_eff, Pb_eff = Pa[idx_a], Pb[idx_b]
            La, Lb = B2A_2d.shape[0], B2A_2d.shape[1]
            La = min(La, Pa_eff.shape[0])
            Lb = min(Lb, Pb_eff.shape[0])
            B2A_2d, A2B_2d = B2A_2d[:La, :Lb], A2B_2d[:Lb, :La]
            Pa_eff, Pb_eff = Pa_eff[:La], Pb_eff[:Lb]

            k_eff = int(min(top_k, B2A_2d.size, A2B_2d.size))
            edges_ab = _topk_edges(B2A_2d, k_eff)
            edges_ba = _topk_edges(A2B_2d, k_eff)
            w_ab = _norm01([w for _, _, w in edges_ab])
            w_ba = _norm01([w for _, _, w in edges_ba])

            # ---- 连线 ----
            for (ia, jb, _), wn in zip(edges_ab, w_ab):
                x1, y1 = Pa_eff[ia]
                x2, y2 = Pb_eff[jb]
                _arrow(ax, x1, y1, x2, y2, color="C0", wn=wn, offset_sign=-1)  # A→B

            for (ib, ja, _), wn in zip(edges_ba, w_ba):
                x1, y1 = Pb_eff[ib]
                x2, y2 = Pa_eff[ja]
                _arrow(ax, x1, y1, x2, y2, color="C3", wn=wn, offset_sign=+1)  # B→A

            if Pa_eff.size > 0:
                ax.scatter(Pa_eff[:, 0], Pa_eff[:, 1], s=10, color="C0", alpha=0.7, zorder=6)
            if Pb_eff.size > 0:
                ax.scatter(Pb_eff[:, 0], Pb_eff[:, 1], s=10, color="C3", alpha=0.7, zorder=6)

            # —— 用绘线所用边反推覆盖强度（用于片段高亮）——
            edges_ab_np = edges_ab
            edges_ba_np = edges_ba
            w_ab_np = w_ab
            w_ba_np = w_ba

            frag_cov_A = defaultdict(float)
            frag_cov_B = defaultdict(float)

            if len(edges_ab_np) == len(w_ab_np):
                for (ia, ib, _), wab in zip(edges_ab_np, w_ab_np):
                    val = float(abs(wab))
                    if 0 <= int(ia) < len(pid_order_a):
                        frag_cov_A[pid_order_a[int(ia)]] += val
                    if 0 <= int(ib) < len(pid_order_b):
                        frag_cov_B[pid_order_b[int(ib)]] += val

            if len(edges_ba_np) == len(w_ba_np):
                for (ib, ia, _), wba in zip(edges_ba_np, w_ba_np):
                    val = float(abs(wba))
                    if 0 <= int(ia) < len(pid_order_a):
                        frag_cov_A[pid_order_a[int(ia)]] += val
                    if 0 <= int(ib) < len(pid_order_b):
                        frag_cov_B[pid_order_b[int(ib)]] += val

            covA_list = [frag_cov_A.get(pid, 0.0) for pid in pid_order_a]
            covB_list = [frag_cov_B.get(pid, 0.0) for pid in pid_order_b]
            covA_norm = _norm01(covA_list)
            covB_norm = _norm01(covB_list)
            frag_w_A = {pid: w for pid, w in zip(pid_order_a, covA_norm)}
            frag_w_B = {pid: w for pid, w in zip(pid_order_b, covB_norm)}

            _overlay_fragments_on_ax(
                ax, frag2xy_a, sx=sx_a, sy=sy_a, base_color="C1", frag_weights=frag_w_A,
                alpha_min=0.18, alpha_max=0.85, sat_min=0.35,
            )
            _overlay_fragments_on_ax(
                ax, frag2xy_b, sx=sx_b, sy=sy_b, base_color="C4", frag_weights=frag_w_B,
                alpha_min=0.18, alpha_max=0.85, sat_min=0.35,
            )

            # 行标题 + 聚合权重
            view_disp = {"brics":"BRICS","fg":"Functional Groups","murcko":"Bemis-Murcko scaffolds","ringpaths":"RingPaths"}.get(view_key,view_key)
            ax_title = fig.add_axes([0.02, top + height - 0.06, 0.30, 0.06])
            ax_title.set_axis_off()
            ax_title.text(0.0, 0.5, f"View: {view_disp}", fontsize=11, va="center", ha="left",
                          bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.9, edgecolor="#444",
                                    linewidth=1.0))

            view_index = view_order.index(view_key)
            weightA = agg_attn['A'][i, 0, view_index]
            weightB = agg_attn['B'][i, 0, view_index]
            ax.text(0.20, 0.0, f"Weight : {weightA:.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=12, color="black")
            ax.text(0.66, 0.0, f"Weight : {weightB:.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=12, color="black")

            # —— 组装 JSON/Prompt 所用的 top_pairs（每视角最多2条）——
            def _select_top_pairs_for_prompt(A2B_2d, B2A_2d, k=2):
                items = []
                if A2B_2d is not None and A2B_2d.size:
                    idxs = np.argsort(-np.abs(A2B_2d.ravel()))[:k]
                    La, Lb = A2B_2d.shape
                    for idxv in idxs:
                        ia, jb = divmod(int(idxv), Lb)
                        items.append(("A->B", ia, jb, float(A2B_2d[ia, jb])))
                if B2A_2d is not None and B2A_2d.size:
                    idxs = np.argsort(-np.abs(B2A_2d.ravel()))[:k]
                    Lb, La = B2A_2d.shape
                    for idxv in idxs:
                        jb, ia = divmod(int(idxv), La)
                        items.append(("B->A", ia, jb, float(B2A_2d[jb, ia])))
                # 合并 top2 by |value|
                items = sorted(items, key=lambda x: -abs(x[3]))[:k]
                return items

            pick = _select_top_pairs_for_prompt(A2B_2d, B2A_2d, k=2)
            vdict = {"top_pairs": []}
            for (direc, ia, jb, score) in pick:
                a_atoms = frag2atoms_a[view_key].get(ia, [])
                b_atoms = frag2atoms_b[view_key].get(jb, [])
                a_desc = describe_fragment(mol_a, a_atoms, prefer_name=(view_key == "fg"), rulebank=FG_RULES) if mol_a and a_atoms else {"name": "", "smiles": ""}
                b_desc = describe_fragment(mol_b, b_atoms, prefer_name=(view_key == "fg"), rulebank=FG_RULES) if mol_b and b_atoms else {"name": "", "smiles": ""}
                vdict["top_pairs"].append({
                    "dir": direc, "fragA_id": int(ia), "fragB_id": int(jb),
                    "attn": float(score), "fragA": a_desc, "fragB": b_desc
                })
            sample_dict["views"][view_key] = vdict

        # 绘制每一行
        for r, vk in enumerate(view_order):
            draw_one_view(i, r, vk)

        # 图例（与颜色一致）
        legend_elems = [
            patches.Patch(color="C0", label="A→B"),
            patches.Patch(color="C3", label="B→A"),
        ]
        fig.legend(handles=legend_elems, loc="lower right", fontsize=7, frameon=False)

        # 标注 y 与 ŷ
        try:
            y_disp, pred_disp = f"{y_true:.2f}", f"ŷ={y_pred:.2f}  |err|={abs_err:.2f}  |rel_err|={rel_err:.3f}"
            fig.text(
                0.5, 0.97,
                f"Regression | y = {y_disp}    {pred_disp}",
                ha="center", va="top", fontsize=11,
                bbox=dict(boxstyle="round,pad=0.45", facecolor="white", alpha=0.95, edgecolor="#1f77b4", linewidth=1.6)
            )
        except Exception as _e:
            fig.text(
                0.5, 0.97,
                f"Label/Pred display failed: {type(_e).__name__}",
                ha="center", va="top", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.45", facecolor="white", alpha=0.95, edgecolor="#ff7f0e", linewidth=1.2)
            )

        # 标题与保存
        if with_cell:
            fig.suptitle(f"Drug {ida}  ⇄  Drug {idb} with Cell Line {idc}", fontsize=12, y=0.99)
            base_name = f"Drug{ida}_x_Drug{idb}_with_Cell{idc}_{idx[i]}"
        else:
            fig.suptitle(f"Drug {ida}  ⇄  Drug {idb} without Cell Line {idc}", fontsize=12, y=0.99)
            base_name = f"Drug{ida}_x_Drug{idb}_without_Cell{idc}_{idx[i]}"
        fig_out_path = os.path.join(out_dir, f"{base_name}.png")
        fig.savefig(fig_out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        # === 写 Prompt（可选）===
        if call_llm:

            prompt, response = _call_llm(
                sample_dict,
                fig_out_path,
                model,# "gpt-4" version in gpt-4o-2024-08-06, "gpt-4-turbo" version in gpt-4-vision
                thresholds=prompt_thresholds,
                external_knowledge=prompt_external_knowledge
            )
            with open(os.path.join(out_dir, f"{base_name}.prompt.json"), "w", encoding="utf-8") as f:
                json.dump(prompt, f, ensure_ascii=False, indent=2)
            with open(os.path.join(out_dir, f"{base_name}.{model}.json"), "w", encoding="utf-8") as f:
                json.dump(response, f, ensure_ascii=False, indent=2)

    return np.asarray(group_ae, dtype=np.float16), np.asarray(group_re, dtype=np.float16)


def _topk_pairs_from_attn(A2B, B2A, k: int=2):
    items = []
    if A2B is not None and A2B.size:
        Lb, La = A2B.shape
        ids = np.argsort(-np.abs(A2B.ravel()))[:k]
        for idx in ids:
            i, j = divmod(idx, La)
            items.append((int(i), int(j), float(A2B[i, j]), "A->B"))

    if B2A is not None and B2A.size:
        La, Lb = B2A.shape
        ids = np.argsort(-np.abs(B2A.ravel()))[:k]
        for idx in ids:
            i, j = divmod(idx, Lb)
            items.append((int(j), int(i), float(B2A[i, j]), "B->A"))
    return sorted(items, key=lambda x: -abs(x[2]))[:k]


# ================ 按药物对聚合，纵向拼接出图（每组药物对一张图） ================
def save_group_montage_by_pair(
    drug_id_a: List[str],
    drug_id_b: List[str],
    cell_id: List[str],
    idx,
    group_ae,
    group_re,
    out_dir: str,
    with_cell: bool,
    base_height: int = 2400,
    col_width: int = 1600,
    dpi: int = 200,
):
    """
    在 save_cross_attn_pairlines 已经把单样本图片写盘之后调用：
    - 按 (drugA, drugB) 分组，把同组样本图纵向叠放成一张大图；
    - 主标题为药物对 + 平均误差。
    """
    pair2items = defaultdict(list)
    for ida, idb, cid, i, ae, re in zip(drug_id_a, drug_id_b, cell_id, idx, group_ae, group_re):
        tag = "with" if with_cell else "without"
        fname = f"Drug{ida}_x_Drug{idb}_{tag}_Cell{cid}_{i}.png"
        fpath = os.path.join(out_dir, fname)
        pair2items[(str(ida), str(idb))].append((str(cid), int(i), fpath, float(ae), float(re)))

    for (ida, idb), items in pair2items.items():
        items = [(cid, i, p, ae, re) for (cid, i, p, ae, re) in items if os.path.exists(p)]
        if not items:
            continue
        items.sort(key=lambda x: x[1])

        cols = len(items)
        fig_h = base_height / dpi
        fig_w = max(1, cols) * (col_width / dpi)

        fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
        top_pad = 0.05
        hgap = 0.02
        ax_h = 1.0 - top_pad
        ax_w = (1.0 - (cols - 1) * hgap) / cols

        mean_ae = []
        mean_re = []
        for r, (cid, i, fpath, ae, re) in enumerate(items):
            mean_ae.append(ae)
            mean_re.append(re)
            left = 1.0 - (r + 1) * ax_w - r * hgap
            ax = fig.add_axes([left, 0.0, ax_w, ax_h])
            ax.set_axis_off()
            img = plt.imread(fpath)
            ax.imshow(img)

        tag = "with_Cell" if with_cell else "without_Cell"
        title = f"Drug {ida}  ⇄  Drug {idb} {tag}"
        subtitle = f"Mean |err| = {np.mean(mean_ae):.3f}    Mean |rel_err| = {np.mean(mean_re):.3f}"

        fig.suptitle(f"{title}\n{subtitle}", fontsize=20, y=0.995)
        out_name = f"Pair_{ida}_x_{idb}_GROUP_{tag}.png"
        out_path = os.path.join(out_dir, out_name)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    print("This module defines save_cross_attn_pairlines(...) for visualization and LLM prompt export.")
