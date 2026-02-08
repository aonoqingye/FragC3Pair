import argparse
import itertools
import os
import pandas as pd
import numpy as np

def _merge_param_summaries(args, sweep_values):
    """
    将本次 hyperparam 实验（sweep_values 中的每个取值）对应的
    summary/{out_info}_cv{folds}.xlsx 的 mean_std['mean'] 行合并成一张总表。
    输出到：summary/{dataset}_Group_{groups}_{hyper}_cv{folds}.xlsx
    """

    # 与 process_folds 一致的目录结构
    work_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    summary_dir = os.path.join(work_dir, "summary")
    ablation_dir = os.path.join(work_dir, "ablation")
    os.makedirs(summary_dir, exist_ok=True)

    # 汇总目标文件名（不包含具体取值，代表本次 sweep 的总表）
    hyper = str(args.hyperparam)
    combined_name = f"{args.dataset}_Group_{args.groups}_{hyper}_汇总.xlsx"
    combined_path = os.path.join(ablation_dir, combined_name)

    rows = []
    for val in sweep_values:
        args.Lc = val
        out_info = (f'{args.dataset}_Group{args.groups}_Frags{"_".join(args.frag_list)}_Agg{args.frag_agg}'
                    f'_C3Attn{args.use_C3Attn}_tri{args.tri_attn}_tokenizer{args.tokenizer}_Lc{args.Lc}')
        xlsx_path = os.path.join(summary_dir, f"{out_info}_cv{int(args.folds)}.xlsx")
        if not os.path.exists(xlsx_path):
            print(f"[merge] 缺失 {xlsx_path}，跳过。")
            continue

        try:
            # 只取 mean_std 表的 mean 行
            ms = pd.read_excel(xlsx_path, sheet_name="mean_std")
            mean_row = ms.loc[ms["stat"] == "mean"].copy()
            if mean_row.empty:
                print(f"[merge] {xlsx_path} 中未找到 mean 行，跳过。")
                continue
            # 添加本次超参取值列，列名用 hyperparam 名称
            mean_row.insert(0, hyper, val)
            rows.append(mean_row)
        except Exception as e:
            print(f"[merge] 读取 {xlsx_path} 出错：{type(e).__name__}: {e}")

    if not rows:
        print("[merge] 未收集到可合并的结果，跳过写入总表。")
        return

    combined = pd.concat(rows, ignore_index=True)

    # 为了便于查看，把 'stat' 列放后面（它对合并仅用于标记 mean）
    if "stat" in combined.columns:
        cols = [c for c in combined.columns if c != "stat"] + ["stat"]
        combined = combined[cols]

    with pd.ExcelWriter(combined_path, engine="xlsxwriter") as writer:
        combined.to_excel(writer, index=False, sheet_name="sweep_mean")

    print(f"[OK] 参数实验总表已写入：{combined_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--folds", type=int, default=5, help="随机划分折数（默认 5）")
    p.add_argument("--groups", type=str, default="Drug", choices=["Cell", "Drug", "none"], help="分组依据")
    # 模型参数
    p.add_argument("--hidden", type=int, default=300, help="隐层维度")
    p.add_argument("--encoder", type=str, default="FragC3",
                   choices=["FragC3", "SDDS", "MultiSyn", "AttenSyn", "DeepDDS_GCN", "DeepDDS_GAT", "GCN", "GAT"])
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
    p.add_argument("--cell_agg", type=int, default=512)
    p.add_argument("--cell_pred", type=int, default=128)
    p.add_argument("--Lc", type=int, default=32)
    # 数据与设备
    p.add_argument("--dataset", type=str, default="ONeil",
                   choices=["ALMANAC", "DrugComb", "ONeil"], help="数据集前缀名（ONeil 自动切换为二分类）")
    p.add_argument("--hyperparam", type=str, default="Lc", help="对比条目")
    args = p.parse_args()

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    if args.encoder == "FragC3":
        out_info = f'{args.dataset}_{args.encoder}_Frags{"_".join(args.frag_list)}_Group{args.groups}'
    else:
        out_info = f'{args.dataset}_{args.encoder}_Group{args.groups}'
    # sweep = [0.0, 0.05, 0.1, 0.15, 0.2]
    # sweep = [1, 2, 3, 4, 5, 6]
    sweep = [16, 32, 64, 128, 256, 384]
    # sweep = [16, 32, 64, 128]
    # sweep = ['scale_dot', 'add', 'dot', 'trilinear']
    # frag_list = ["brics", "fg", "murcko", "ringpaths"]
    # sweep = [["brics"], ["fg"], ["murcko"], ["brics", "fg"], ["brics", "fg", "murcko"]]
    # sweep = ["FragC3", "SDDS", "MultiSyn", "AttenSyn", "DeepDDS_GCN", "DeepDDS_GAT", "GCN", "GAT"]
    # sweep = []
    # for r in range(1, 5):  # 长度从 1 到 4
    #     sweep.extend(itertools.combinations(frag_list, r))

    _merge_param_summaries(args, sweep)