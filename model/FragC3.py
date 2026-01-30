import dgl
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.TriAttention import *
from utils import get_func
from dgl import function as fn
from functools import partial
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d


# =========================
# dgl graph utils & core ops
# =========================
def reverse_edge(tensor: torch.Tensor):
    """Reorder edge features so that msg on (u->v) aligns with (v->u).
    Expect even #edges (paired), but gracefully handle empty input."""
    n = tensor.size(0)
    if n == 0:
        return tensor  # 空张量直接返回

    assert n % 2 == 0, "reverse_edge expects even number of edges (paired)."

    device = tensor.device
    delta = torch.ones(n, dtype=torch.long, device=device)
    delta[1::2] = -1
    idx = torch.arange(n, device=device) + delta
    return tensor[idx]


def del_reverse_message(edge, field):
    """Subtract the corresponding reverse edge message from the edge"""
    return {"m": edge.src[field] - edge.data["rev_h"]}


def add_attn(node, field, attn):
    """Aggregate messages in the mailbox using the attention mechanism"""
    feat = node.data[field].unsqueeze(1)
    return {field: (attn(feat, node.mailbox["m"], node.mailbox["m"]) + feat).squeeze(1)}


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.):
        """Take in model size and number of heads."""
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, return_attn: bool=False):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        out = self.linears[-1](x)
        return (out, self.attn) if return_attn else out


# =========================
# Readout & Attention blocks
# =========================
class Node_readout(nn.Module):
    """graph readout. Implemented with dgl graph;
    Suitable for integrating embeddings of different types of nodes in the graph.
    read='atom' / 'frag'
    """
    def __init__(self, hid_dim, mix_heads=6, read='atom', dropout=0., bidirectional=True):
        super(Node_readout, self).__init__()
        self.hid_dim = hid_dim
        self.att_mix = MultiHeadedAttention(mix_heads, hid_dim, dropout)
        self.read = read

    def split_batch(self, bg, ntype, field, device):
        hidden = bg.nodes[ntype].data[field]
        node_size = bg.batch_num_nodes(ntype)
        start_index = torch.cat(
            [torch.tensor([0], device=device), torch.cumsum(node_size, 0)[:-1]]
        )
        max_num_node = max(node_size) if node_size.numel() > 0 else torch.tensor(1, device=device)

        hidden_lst = []
        for i in range(bg.batch_size):
            start, size = start_index[i], node_size[i]
            assert size != 0, size
            cur_hidden = hidden.narrow(0, start, size)
            cur_hidden = torch.nn.ZeroPad2d((0, 0, 0, max_num_node - cur_hidden.shape[0]))(cur_hidden)
            hidden_lst.append(cur_hidden.unsqueeze(0))
        hidden_lst = torch.cat(hidden_lst, 0)
        return hidden_lst

    def forward(self, bg, suffix="h"):
        self.suffix = suffix
        device = bg.device
        p_pharmj = self.split_batch(bg, "p", f"f_{suffix}", device)
        a_pharmj = self.split_batch(bg, "a", f"f_{suffix}", device)
        valid_a = (a_pharmj != 0).any(dim=-1)
        valid_p = (p_pharmj != 0).any(dim=-1)
        mask_pa = ~(valid_a.unsqueeze(-1) & valid_p.unsqueeze(1))
        mask_ap = ~(valid_p.unsqueeze(-1) & valid_a.unsqueeze(1))

        if self.read == 'atom':
            node_type = 'a'; valid = valid_a; mask = mask_pa
            h = self.att_mix(a_pharmj, p_pharmj, p_pharmj, mask) + a_pharmj
        elif self.read == 'frag':
            node_type = 'p'; valid = valid_p; mask = mask_ap
            h = self.att_mix(p_pharmj, a_pharmj, a_pharmj, mask) + p_pharmj
        else:
            raise ValueError(f"Mix read out not implement: {self.read}")

        node_size = bg.batch_num_nodes(node_type)
        return h, valid, node_size


def masked_layernorm(x, valid_mask, ln_module):
    B, L, D = x.shape
    x_flat = x.view(B * L, D)
    m_flat = valid_mask.view(B * L)
    if m_flat.any():
        x_sel = x_flat[m_flat]
        x_sel = ln_module(x_sel)
        x_flat[m_flat] = x_sel
    return x


class MaskedCrossAttn(nn.Module):
    def __init__(self, hid_dim: int, heads: int = 4, dropout: float = 0.):
        super().__init__()
        self.ln_a = nn.LayerNorm(hid_dim)
        self.ln_b = nn.LayerNorm(hid_dim)
        self.ln_out_a = nn.LayerNorm(hid_dim)
        self.ln_out_b = nn.LayerNorm(hid_dim)
        self.attn  = MultiHeadedAttention(heads, hid_dim, dropout=dropout)

    def forward(self, x_a, x_b, valid_a, valid_b, return_attn: bool=False):
        mask_B2A = ~(valid_a.unsqueeze(-1) & valid_b.unsqueeze(1))
        mask_A2B = ~(valid_b.unsqueeze(-1) & valid_a.unsqueeze(1))
        a  = masked_layernorm(x_a.clone(), valid_a, self.ln_a)
        b  = masked_layernorm(x_b.clone(), valid_b, self.ln_b)

        out_a, attn_B2A = self.attn(a, b, b, mask_B2A, return_attn=True)
        out_a = masked_layernorm(out_a + x_a, valid_a, self.ln_out_a)

        out_b, attn_A2B = self.attn(b, a, a, mask_A2B, return_attn=True)
        out_b = masked_layernorm(out_b + x_b, valid_b, self.ln_out_b)

        if return_attn:
            return out_a, out_b, {"B2A": attn_B2A, "A2B": attn_A2B}
        return out_a, out_b, None


class CellConditionedCrossContextAttn(nn.Module):
    def __init__(self, hid_dim: int, heads: int = 4, dropout: float = 0.,
                 ctx_in_dim: int = 512, Lc: int = 16,
                 tokenizer: str = 'conv',
                 tri_variant: str = 'scale_dot',  # ['scale_dot', 'add', 'dot', 'trilinear']
                 cv_mode: str = 'scale_dot'):  # ["mul", "add", "bilinear"]
        super().__init__()
        self.ln_a = nn.LayerNorm(hid_dim)
        self.ln_b = nn.LayerNorm(hid_dim)
        self.ln_out_a = nn.LayerNorm(hid_dim)
        self.ln_out_b = nn.LayerNorm(hid_dim)
        self.ctx_tokenizer = CellLineContextTokenizer(ctx_in_dim=ctx_in_dim, d_model=hid_dim, Lc=Lc, dropout=dropout)
        if tokenizer == 'linear':
            self.ctx_tokenizer = ContextLinearTokenizer(
                ctx_in_dim=ctx_in_dim, d_model=hid_dim, dropout=dropout
            )
        elif tokenizer == 'conv':
            self.ctx_tokenizer = CellLineContextTokenizer(
                ctx_in_dim=ctx_in_dim, d_model=hid_dim, Lc=Lc, dropout=dropout
            )
        else:
            raise ValueError(f"Unknown tokenizer: {tokenizer}")
        if tri_variant == 'scale_dot':
            self.tri_attn = TriEinsumAttention(d_model=hid_dim, num_heads=heads, attn_dropout=dropout, cv_mode=cv_mode)
        elif tri_variant == 'add':
            self.tri_attn = TriAddAttention(d_model=hid_dim, num_heads=heads, attn_dropout=dropout, cv_mode=cv_mode)
        elif tri_variant == 'dot':
            self.tri_attn = TriDotProductAttention(d_model=hid_dim, num_heads=heads, attn_dropout=dropout,
                                                   cv_mode=cv_mode)
        elif tri_variant == 'trilinear':
            self.tri_attn = TriTrilinearAttention(d_model=hid_dim, num_heads=heads, attn_dropout=dropout,
                                                  cv_mode=cv_mode)
        else:
            raise ValueError(f"Unknown tri_variant: {tri_variant}")

    def forward(self, x_a, x_b, valid_a, valid_b, cell_vec, return_attn: bool=False):
        mask_B2A = ~(valid_a.unsqueeze(-1) & valid_b.unsqueeze(1))
        mask_A2B = ~(valid_b.unsqueeze(-1) & valid_a.unsqueeze(1))
        a = masked_layernorm(x_a.clone(), valid_a, self.ln_a)
        b = masked_layernorm(x_b.clone(), valid_b, self.ln_b)

        C_tokens = self.ctx_tokenizer(cell_vec)  # [B,Lc,D]

        out_a, attn_flat_A = self.tri_attn(a, b, b, C_tokens, attn_mask_flat=mask_B2A)
        out_a = masked_layernorm(out_a + x_a, valid_a, self.ln_out_a)

        out_b, attn_flat_B = self.tri_attn(b, a, a, C_tokens, attn_mask_flat=mask_A2B)
        out_b = masked_layernorm(out_b + x_b, valid_b, self.ln_out_b)

        if not return_attn:
            return out_a, out_b, None

        def reduce_ctx(attn_flat, Lq, Lk, Lc):
            attn_5d = attn_flat.view(attn_flat.size(0), attn_flat.size(1), Lq, Lk, Lc)
            return attn_5d.sum(-1)

        B_, H_, La_, LbLc_ = attn_flat_A.shape
        Lc = C_tokens.size(1)
        Lb = LbLc_ // Lc
        attn_B2A = reduce_ctx(attn_flat_A, La_, Lb, Lc)

        B_, H_, Lb_, LaLc_ = attn_flat_B.shape
        La = LaLc_ // Lc
        attn_A2B = reduce_ctx(attn_flat_B, Lb_, La, Lc)

        return out_a, out_b, {"B2A": attn_B2A, "A2B": attn_A2B}


class PreNormFFN(nn.Module):
    def __init__(self, hid_dim: int, expansion: int = 4, dropout: float = 0.):
        super().__init__()
        self.ln = nn.LayerNorm(hid_dim)
        self.ln_out = nn.LayerNorm(hid_dim)
        act = nn.GELU()
        hidden = expansion * hid_dim
        self.ffn = nn.Sequential(
            nn.Linear(hid_dim, hidden),
            act,
            nn.Dropout(dropout),
            nn.Linear(hidden, hid_dim),
        )

    def forward(self, x: torch.Tensor, valid_mask):
        x_norm = masked_layernorm(x.clone(), valid_mask, self.ln)
        y = self.ffn(x_norm)
        out = x + y
        out = masked_layernorm(out, valid_mask, self.ln_out)
        return out


class C3Attn(nn.Module):
    def __init__(self, hid_dim, heads=6, ffn_expansion=4, tri_attn=False,
                 tri_variant: str = "scale_dot", cv_mode: str = "mul",
                 tokenizer: str = "conv", cell_dim=1722, Lc=16, dropout=0.):
        super(C3Attn, self).__init__()
        self.hid_dim = hid_dim
        self.cell_attn = tri_attn
        if tri_attn:
            print(f'Tri Mode = {tri_variant}, C-V Mode = {cv_mode}')
            self.attn = CellConditionedCrossContextAttn(hid_dim, heads, dropout, tokenizer=tokenizer,
                                                        tri_variant=tri_variant, cv_mode=cv_mode,
                                                        ctx_in_dim=cell_dim, Lc=Lc)
        else:
            self.attn = MaskedCrossAttn(hid_dim, heads, dropout)
        self.ffn = PreNormFFN(hid_dim, ffn_expansion, dropout)

    def forward(self, h_a, h_b, valid_a, valid_b, cell_vec=None, return_attn: bool=False, **kw):
        if self.cell_attn:
            assert cell_vec is not None, "cell_vec must be provided when tri_attn=True"
            cross_h_a, cross_h_b, attn = self.attn(h_a.clone(), h_b.clone(), valid_a, valid_b, cell_vec, return_attn)
        else:
            cross_h_a, cross_h_b, attn = self.attn(h_a.clone(), h_b.clone(), valid_a, valid_b, return_attn)
        out_a = self.ffn(cross_h_a, valid_a)
        out_b = self.ffn(cross_h_b, valid_b)
        return out_a, out_b, attn


class MVMP(nn.Module):
    def __init__(
        self,
        msg_func=add_attn,
        hid_dim=300,
        heads=4,
        depth=3,
        view="aba",
        suffix="apj",
        act=nn.ReLU(),
    ):
        """
        MultiViewMassagePassing
        view: a: atom, p: frag, j: use junction-enhanced membership edges
              例如 "apj" 表示用 a-b-a、p-r-p 以及 a-c-p/p-c-a（带 f_junc_*）三类通道
        """
        super(MVMP, self).__init__()
        self.view = view
        self.depth = depth
        self.suffix = suffix
        self.msg_func = msg_func
        self.act = act

        # 只存在两类节点：原子(a)与片段(p)
        self.node_types = ["a", "p"]

        # 同构边：原子-原子；若包含 'p'，再加片段-片段
        self.homo_etypes = [("a", "b", "a")]
        if "p" in view:
            self.homo_etypes.append(("p", "r", "p"))

        # 异构 membership 边：使用 prepare 中已有的 'c'
        # 原先误写成了 'j'，导致 DGL 报不存在
        self.hetero_etypes = []
        if "j" in view:
            self.hetero_etypes = [("a", "c", "p"), ("p", "c", "a")]

        # 注意力层（每种边一套）
        self.attn = nn.ModuleDict()
        for etype in self.homo_etypes + self.hetero_etypes:
            self.attn["".join(etype)] = MultiHeadedAttention(heads, hid_dim)

        # 同构边的中间层
        self.mp_list = nn.ModuleDict()
        for edge_type in self.homo_etypes:
            self.mp_list["".join(edge_type)] = nn.ModuleList(
                [nn.Linear(hid_dim, hid_dim) for _ in range(depth - 1)]
            )

        # 末端聚合层（按目标节点类型区分）
        self.node_last_layer = nn.ModuleDict({
            "a": nn.Linear(3 * hid_dim, hid_dim),
            "p": nn.Linear(3 * hid_dim, hid_dim),
        })

    # ---- 内部小工具 ----
    def update_edge(self, edge, layer):
        return {"h": self.act(edge.data["x"] + layer(edge.data["m"]))}

    def update_node(self, node, field, layer):
        # 将收到的 mail 与自身表示、静态特征拼接
        return {
            field: layer(
                torch.cat([node.mailbox["mail"].sum(dim=1), node.data[field], node.data["f"]], 1)
            )
        }

    def init_node(self, node):
        return {f"f_{self.suffix}": node.data["f"].clone()}

    def init_edge(self, edge):
        return {"h": edge.data["x"].clone()}

    # ---- 前向 ----
    def forward(self, bg):
        suffix = self.suffix

        # 初始化节点特征缓存 & 边特征
        for ntype in self.node_types:
            bg.apply_nodes(self.init_node, ntype=ntype)
        for etype in self.homo_etypes:
            bg.apply_edges(self.init_edge, etype=etype)

        # 若启用 'j' 视角，则把 junction 拼接后的特征放到 f_junc_{suffix}
        if "j" in self.view:
            bg.nodes["a"].data[f"f_junc_{suffix}"] = bg.nodes["a"].data["f_junc"].clone()
            bg.nodes["p"].data[f"f_junc_{suffix}"] = bg.nodes["p"].data["f_junc"].clone()

        # --- 第一阶段：同构边消息传递（多层） ---
        update_funcs_homo = {
            e: (
                fn.copy_e("h", "m"),
                partial(self.msg_func, attn=self.attn["".join(e)], field=f"f_{suffix}"),
            )
            for e in self.homo_etypes
        }

        for i in range(self.depth - 1):
            bg.multi_update_all(update_funcs_homo, cross_reducer="sum")

            # 反向边配对 + 边更新
            for edge_type in self.homo_etypes:
                E = bg.num_edges(edge_type)
                h = bg.edges[edge_type].data.get("h", None)
                if h is None:
                    continue
                if E == 0 or h.size(0) == 0:
                    bg.edges[edge_type].data["rev_h"] = h
                    continue
                bg.edges[edge_type].data["rev_h"] = reverse_edge(h)
                bg.apply_edges(partial(del_reverse_message, field=f"f_{suffix}"), etype=edge_type)
                bg.apply_edges(partial(self.update_edge, layer=self.mp_list["".join(edge_type)][i]), etype=edge_type)

        # --- 第二阶段：把同构边的消息聚合到节点（a/p 各自使用对应的 last_layer） ---
        update_funcs_last = {
            e: (
                fn.copy_e("h", "mail"),
                partial(self.update_node, field=f"f_{suffix}", layer=self.node_last_layer[e[0]]),
            )
            for e in self.homo_etypes
        }
        bg.multi_update_all(update_funcs_last, cross_reducer="sum")

        # --- 第三阶段：若启用 'j' 视角，再走异构 membership 边 （使用 f_junc_* 作为消息源）---
        if "j" in self.view and len(self.hetero_etypes) > 0:
            update_funcs_hetero = {}
            for e in self.hetero_etypes:
                # e: (src_ntype, 'c', dst_ntype)
                dst_ntype = e[2]                # 'a' or 'p'
                layer = self.node_last_layer[dst_ntype]
                update_funcs_hetero[e] = (
                    fn.copy_u(f"f_junc_{suffix}", "mail"),
                    partial(self.update_node, field=f"f_junc_{suffix}", layer=layer),
                )
            bg.multi_update_all(update_funcs_hetero, cross_reducer="sum")

# =========================
# FragC3: 多视角片段编码器
# =========================
class FragC3(nn.Module):
    """
    多视角片段融合的药物组合预测模型。

    参数：
      n_output: 分类或回归维度
      cell_dim: 细胞系原始输入维度
      hid_dim:  隐层维度
      heads / ffn_expansion / dropout: 与现有实现一致
      tri_attn: 是否在 Bi2Frag 中启用 cell 上下文注意力
      frag_list: 使用的片段视角列表，例如 ["brics","fg","murcko","ringpaths"] 的任意子集
      frag_agg:  片段视角聚合策略：'mlp' / 'gate' / 'tri_attn'
    """
    FRAG_KEYS = ["brics", "fg", "murcko"]

    def __init__(self, n_output=2, cell_dim=1722, hid_dim=300, heads=6, ffn_expansion=4, use_C3Attn=False,
                 tri_attn=False, tri_variant='scale_dot', cv_mode='mul', cell_pred=512, cell_agg=512, Lc=16,
                 tokenizer='conv', frag_list=("brics", "fg", "murcko"), frag_agg="mlp", dropout=0.2):
        super().__init__()
        self.act = get_func("ReLU")
        self.hid_dim = hid_dim
        self.tri_attn = tri_attn
        self.use_C3Attn = use_C3Attn

        # 规范化片段列表并检查
        frag_list = list(frag_list) if isinstance(frag_list, (list, tuple)) else [frag_list]
        for k in frag_list:
            assert k in self.FRAG_KEYS, f"Unknown fragment view: {k}"
        self.frag_list = frag_list

        # ——（1）每个片段视角独立的输入投影矩阵——
        # 维度：与原实现一致
        in_dims = dict(atom=45, bond=14, pharm=194, reac=34, junc=45+194)
        self.in_proj = nn.ModuleDict()
        for k in self.FRAG_KEYS:
            self.in_proj[k] = nn.ModuleDict({
                "w_atom": nn.Linear(in_dims["atom"], hid_dim),
                "w_bond": nn.Linear(in_dims["bond"], hid_dim),
                "w_pharm": nn.Linear(in_dims["pharm"], hid_dim),
                "w_reac": nn.Linear(in_dims["reac"], hid_dim),
                "w_junc": nn.Linear(in_dims["junc"], hid_dim),
            })

        def build_mvmp():
            return MVMP(
                msg_func=add_attn,
                hid_dim=hid_dim,
                heads=4,
                depth=3,
                view="apj",
                suffix="h",
                act=self.act,
            )
        self.mvmp = nn.ModuleDict({k: build_mvmp() for k in self.FRAG_KEYS})

        def build_C3A():
            return C3Attn(
                hid_dim=hid_dim,
                heads=heads,
                ffn_expansion=ffn_expansion,
                tri_attn=tri_attn,
                tri_variant=tri_variant,
                cv_mode=cv_mode,
                tokenizer=tokenizer,
                cell_dim=cell_dim,
                Lc=Lc,
                dropout=dropout)
        if use_C3Attn:
            self.C3A = nn.ModuleDict({k: build_C3A() for k in self.FRAG_KEYS})

        # 片段级读出与 GRU（共享即可；节点级输出维度一致）
        self.readout = Node_readout(hid_dim, 6, 'frag', dropout)
        self.gru = nn.GRU(hid_dim, hid_dim, batch_first=True, bidirectional=True)

        # cell features MLP
        self.aggregation = nn.Sequential(
            nn.Linear(cell_dim, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, cell_agg),
            nn.ReLU(),
        )
        self.prediction = nn.Sequential(
            nn.Linear(cell_dim, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, cell_pred),
            nn.ReLU(),
        )

        # ——（4）跨片段视角聚合头——
        self.frag_agg = frag_agg
        out_per_drug = hid_dim * 2  # 双向 GRU 的输出维度（pool 后）
        num_views = len(self.frag_list)

        if frag_agg == "mlp":
            self.agg = nn.Sequential(
                nn.Linear(out_per_drug * num_views, out_per_drug),
                nn.ReLU(),
            )
        elif frag_agg == "gate":
            # 可学习门控（对每个视角输出一个标量 gate，再 softmax）
            self.gate = nn.Linear(out_per_drug, 1)
        elif frag_agg == "cell_attn":
            # 用 cell 向量 Query，对各视角 Key 做注意力
            self.q = nn.Linear(cell_agg, out_per_drug)
            self.k = nn.Linear(out_per_drug, out_per_drug)
            self.v = nn.Linear(out_per_drug, out_per_drug)
        else:
            raise ValueError(f"Unknown frag_agg: {frag_agg}")

        # ——（5）最终预测头：concat(A,B,cell)——
        self.pred = nn.Sequential(
            nn.Linear(out_per_drug * 2 + cell_pred, 1024),
            BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, n_output),
        )

        self.initialize_weights()

    # 兼容旧 MultiSyn 名称（可选）
    # 你也可以在训练脚本中直接用 FragC3，并传 frag_list=["brics"] 来复现老结果
    # （保留 class MultiSyn = FragC3 的别名也行，见文末）
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GRU):
                for name, p in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(p.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(p.data)
                    elif 'bias' in name:
                        nn.init.zeros_(p.data)

    # ——按视角使用“独立输入投影矩阵”——
    def init_feature_for_view(self, bg, view_key: str):
        proj = self.in_proj[view_key]

        # a-b-a
        bg.nodes["a"].data["f"] = self.act(proj["w_atom"](bg.nodes["a"].data["f"]))
        # bond x 一般都在，如果担心极端情况，也可做同样的容错
        if "x" not in bg.edges[("a", "b", "a")].data:
            Ea = bg.num_edges(("a", "b", "a"))
            xa = torch.zeros((Ea, 14), device=bg.device)
        else:
            xa = bg.edges[("a", "b", "a")].data["x"]
            if xa.dim() == 1:
                xa = xa.view(-1, 14)
        bg.edges[("a", "b", "a")].data["x"] = self.act(proj["w_bond"](xa))

        # p-r-p —— 关键：容错拿 x
        Et = bg.num_edges(("p", "r", "p"))
        x_exist = "x" in bg.edges[("p", "r", "p")].data
        if not x_exist:
            xr = torch.zeros((Et, 34), device=bg.device)
        else:
            xr = bg.edges[("p", "r", "p")].data["x"]
            if xr.dim() == 1:
                xr = xr.view(-1, 34)
            # 某些构图里可能写入了空形状，兜底到 (E,34)
            if xr.numel() == 0 and Et >= 0:
                xr = torch.zeros((Et, 34), device=bg.device)
            if xr.size(-1) == 0:
                xr = torch.zeros((Et, 34), device=bg.device)

        bg.edges[("p", "r", "p")].data["x"] = self.act(proj["w_reac"](xr))

        # 节点片段与 junc
        bg.nodes["p"].data["f"] = self.act(proj["w_pharm"](bg.nodes["p"].data["f"]))

        # junc 特征（拼接得到的 f_junc 一定存在；只做线性映射）
        bg.nodes["a"].data["f_junc"] = self.act(proj["w_junc"](bg.nodes["a"].data["f_junc"]))
        bg.nodes["p"].data["f_junc"] = self.act(proj["w_junc"](bg.nodes["p"].data["f_junc"]))

    def _encode_one_view(self, graphs_a, graphs_b, view_key, cell_vec=None, return_attn=False):
        # batch & init
        ga = dgl.batch(graphs_a); gb = dgl.batch(graphs_b)
        self.init_feature_for_view(ga, view_key)
        self.init_feature_for_view(gb, view_key)

        self.mvmp[view_key](ga); self.mvmp[view_key](gb)

        # node readout
        h_a, valid_a, node_size_a = self.readout(ga, "h")
        h_b, valid_b, node_size_b = self.readout(gb, "h")

        if self.use_C3Attn:
            h_a, h_b, attn = self.C3A[view_key](h_a, h_b, valid_a, valid_b, cell_vec=cell_vec, return_attn=return_attn)
        else:
            attn = None

        # GRU with init (max-pool → init h0)
        init_a = h_a.max(1)[0].unsqueeze(0).repeat(2, 1, 1)
        init_b = h_b.max(1)[0].unsqueeze(0).repeat(2, 1, 1)
        h_a, _ = self.gru(h_a, init_a)
        h_b, _ = self.gru(h_b, init_b)

        # graph-level mean over valid nodes
        def pool_valid(h, sizes):
            embs = []
            for i in range(h.shape[0]):
                n = int(sizes[i]); embs.append(h[i, :n].reshape(n, -1).mean(0, keepdim=True))
            return torch.cat(embs, dim=0)  # [B, 2*D]
        emb_a = pool_valid(h_a, node_size_a)
        emb_b = pool_valid(h_b, node_size_b)
        return emb_a, emb_b, attn

    def _aggregate_views(self, embs, mode, cell_vec):
        """
        embs: list of [B, 2D] for each view
        mode: 'mlp' | 'gate' | 'cell_attn'
        """
        if mode == "mlp":
            z = torch.cat(embs, dim=-1)
            return self.agg(z), None

        elif mode == "gate":
            # scalar gate per view
            V = torch.stack([e for e in embs], dim=1)  # [B, V, 2D]
            g = torch.stack([self.gate(e) for e in embs], dim=1)  # [B, V, 1]
            w = F.softmax(g, dim=1)  # [B, V, 1]
            return (V * w).sum(dim=1), None  # [B, 2D]

        elif mode == "cell_attn":
            # cell as query
            V = torch.stack([self.v(e) for e in embs], dim=1)  # [B, V, 2D]
            Q = self.q(cell_vec).unsqueeze(1)  # [B, 1, 2D]
            K = torch.stack([self.k(e) for e in embs], dim=1)  # [B, V, 2D]
            att = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(V.size(-1))  # [B,1,V]
            w = F.softmax(att, dim=-1)  # [B,1,V]
            return torch.matmul(w, V).squeeze(1), w  # [B, 2D]
        else:
            raise ValueError(mode)

    def forward(self, data, return_attn: bool=False):
        # cell
        cell = F.normalize(data.cell1, 2, 1)
        # cell_vec = self.reduction(cell)     # [B, cell_hid]
        cell_agg = self.aggregation(cell)   # [B, cell_agg]
        cell_pred = self.prediction(cell)   # [B, cell_pred]

        # 准备各视角图列表（来自 dataset.py 的四个字段）
        def get_graphs(data, key):
            if key == "brics":
                return data.graph1, data.graph2
            elif key == "fg":
                return data.graph1_fg, data.graph2_fg
            elif key == "murcko":
                return data.graph1_murcko, data.graph2_murcko
            else:
                raise KeyError(key)

        # 逐视角编码 → 每药物得到一个 [B, 2D] 表示
        embsA = []; embsB = []; extra_per_view = {}
        for k in self.frag_list:
            ga, gb = get_graphs(data, k)
            emb_a, emb_b, extra = self._encode_one_view(ga, gb, k, cell_vec=cell if self.tri_attn else None, return_attn=return_attn)
            embsA.append(emb_a); embsB.append(emb_b)
            if return_attn: extra_per_view[k] = extra

        # 视角聚合
        agg_attn = None
        if len(embsA) == 1:
            aggA = embsA[0]
            aggB = embsB[0]
        else:
            aggA, agg_attnA = self._aggregate_views(embsA, self.frag_agg, cell_agg)
            aggB, agg_attnB = self._aggregate_views(embsB, self.frag_agg, cell_agg)
            if agg_attnA is not None:
                agg_attn = {'A': agg_attnA, 'B': agg_attnB}

        # 预测
        xc = torch.cat((aggA, aggB, cell_pred), dim=1)
        xc = F.normalize(xc, 2, 1)
        out = self.pred(xc)

        if return_attn:
            return out, extra_per_view, agg_attn
        return out
