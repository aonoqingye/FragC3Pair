import argparse
import itertools
import os
import pandas as pd
import numpy as np

def process_folds(args, out_dir, out_info):
    folds = int(args.folds)
    per_fold_best_rows = []
    csv_paths = []

    for i in range(1, folds + 1):
        path_i = os.path.join(out_dir, f"{out_info}_fold_{i}.csv")
        csv_paths.append(path_i)

        if not os.path.exists(path_i):
            print(f"[WARN] 缺失 {path_i}，跳过。")
            continue

        df_i = pd.read_csv(path_i)
        if df_i.empty:
            print(f"[WARN] {path_i} 为空，跳过。")
            continue

        # 1) 有 test 行则优先取 test
        epoch_str = df_i["Epoch"].astype(str)
        if (epoch_str == "test").any():
            best_row = df_i.loc[epoch_str == "test"].copy()          # ★ 关键：copy()
        else:
            # 2) 否则用验证指标选最优：分类取最大，回归取最小
            #    与你训练时 csv 第二列含义保持一致（AUC_dev 或 MSE）
            metric_col = df_i.columns[1]
            if args.dataset.startswith("ONeil"):                         # classification
                idx = df_i[metric_col].idxmax()
            else:                                                    # regression
                idx = df_i[metric_col].idxmin()
            best_row = df_i.loc[[idx]].copy()

        # 安全赋值：使用 .loc
        best_row.loc[:, "fold"] = i
        per_fold_best_rows.append(best_row)

    if not per_fold_best_rows:
        print("[WARN] 未找到可汇总的折结果，跳过写 Excel。")
        return

    per_fold_best = pd.concat(per_fold_best_rows, ignore_index=True)

    # 仅对数值列做 mean/std
    numeric_cols = [c for c in per_fold_best.columns if c not in ["fold", "Epoch"]]
    mean_row = per_fold_best[numeric_cols].mean()
    std_row = per_fold_best[numeric_cols].std(ddof=1)  # 样本方差

    mean_df = pd.DataFrame([mean_row])
    std_df = pd.DataFrame([std_row])
    mean_df.insert(0, "stat", "mean")
    std_df.insert(0, "stat", "std")
    mean_std = pd.concat([mean_df, std_df], ignore_index=True)

    # 目标目录：<out_dir>/summary/
    summary_dir = os.path.join(os.path.dirname(out_dir), 'summary')
    os.makedirs(summary_dir, exist_ok=True)                          # ★ 关键：先建目录
    xlsx_path = os.path.join(summary_dir, f"{out_info}_cv{folds}.xlsx")

    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
        per_fold_best.to_excel(writer, index=False, sheet_name="per_fold_best")
        mean_std.to_excel(writer, index=False, sheet_name="mean_std")

    print(f"[OK] CV 汇总已写入：{xlsx_path}")

def _merge_param_summaries(args, sweep_values):
    """
    将本次 hyperparam 实验（sweep_values 中的每个取值）对应的
    summary/{out_info}_cv{folds}.xlsx 的 mean_std['mean'] 行合并成一张总表。
    输出到：summary/{dataset}_Group_{groups}_{hyper}_cv{folds}.xlsx
    """

    # 与 process_folds 一致的目录结构
    work_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(work_dir, "results")
    summary_dir = os.path.join(work_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)

    # 汇总目标文件名（不包含具体取值，代表本次 sweep 的总表）
    hyper = str(args.hyperparam)
    combined_name = f"{args.dataset}_Group_{args.groups}_{hyper}_汇总.xlsx"
    combined_path = os.path.join(summary_dir, combined_name)

    rows = []
    for val in sweep_values:
        # for c in ["mul", "add", "bilinear"]:
        #     args.cv_mode = c
        # 复用 main() 里生成 out_info 的规则
        # out_info = f"{args.dataset}_Group_{args.groups}_{args.hyperparam}_{val}"
        f'{args.dataset}_Group{args.groups}_Frags{"_".join(args.frag_list)}_H{args.heads}'
        f'_CellAgg{args.cell_agg}_CellPred{args.cell_pred}_Lc{args.Lc}'
        out_info = (f'{args.dataset}_Group{args.groups}_Frags{"_".join(args.frag_list)}_H{args.heads}'
                f'_CellAgg{val}_CellPred{args.cell_pred}_Lc{args.Lc}')
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
    p.add_argument("--encoder", type=str, default="SDDS",
                   choices=["FragC3", "SDDS", "MultiSyn", "AttenSyn", "DeepDDS_GCN", "DeepDDS_GAT", "GCN", "GAT"])
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
    args = p.parse_args()

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    if args.encoder == "FragC3":
        out_info = f'{args.dataset}_{args.encoder}_Frags{"_".join(args.frag_list)}_Group{args.groups}'
    else:
        out_info = f'{args.dataset}_{args.encoder}_Group{args.groups}'
    process_folds(args, out_dir, out_info)
    # sweep = [0.0, 0.05, 0.1, 0.15, 0.2]
    # sweep = [1, 2, 3, 4, 5, 6]
    # sweep = [32, 64, 128, 256, 512]
    # sweep = [16, 32, 64, 128]
    # sweep = ['scale_dot', 'add', 'dot', 'trilinear']
    # frag_list = ["brics", "fg", "murcko", "ringpaths"]
    # sweep = [["brics"], ["fg"], ["murcko"], ["brics", "fg"], ["brics", "fg", "murcko"]]
    # sweep = []
    # for r in range(1, 5):  # 长度从 1 到 4
    #     sweep.extend(itertools.combinations(frag_list, r))

    # _merge_param_summaries(args, sweep)

    # out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    # for f in [["brics"], ["brics", "fg"], ["brics", "fg", "murcko"]]:
    #     for d in ["ONeil", "OncoPolyPharmacology"]:
    #         for c in [True, False]:
    #             args.dataset = d
    #             args.frag_list = f
    #             args.tri_attn = c
    #             out_info = f'{args.dataset}_Frags_'
    #             for i in range(len(args.frag_list)):
    #                 out_info += f"{args.frag_list[i]}_"
    #             out_info += f"{args.frag_agg}_agg_Group_{args.groups}_notest"
    #             if args.tri_attn:
    #                 out_info += '_withCell'
    #             else:
    #                 out_info += '_woCell'
    #             process_folds(args, out_dir, out_info)