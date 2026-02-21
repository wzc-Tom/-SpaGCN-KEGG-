import os
import pandas as pd
import re

# ======================
# 配置
# ======================

SEARCH_ROOT = r"C:\Users\20199\Desktop\实验报告\生信\kegg_results"

CATEGORY_MAP = {
    "1. 神经血管生态位 (Neurovascular niche)": [
        "ECM-receptor interaction",
        "Focal adhesion",
        "Axon guidance",
        "Smooth muscle contraction",
        "Leukocyte transendothelial migration"
    ],

    "2. 代谢增殖核心 (Metabolic–proliferative core)": [
        "Glycolysis / Gluconeogenesis",
        "HIF-1 signaling pathway",
        "Carbon metabolism",
        "PPAR signaling pathway",
        "Oxidative phosphorylation",
        "Protein digestion and absorption"
    ],

    "3. 血管屏障界面 (Vascular–barrier interface)": [
        "Cell junction",
        "Antigen processing and presentation"
    ],

    "4. 缺氧应激区 (Hypoxic–stress zone)": [
        "Apoptosis",
        "Necroptosis",
        "Endoplasmic reticulum stress"
    ]
}

# 扁平关键词列表（regex 安全）
ALL_KEYWORDS = [kw for sub in CATEGORY_MAP.values() for kw in sub]
PATTERN = "|".join(re.escape(k) for k in ALL_KEYWORDS)

output_rows = []

# ======================
# 遍历目录读取数据
# ======================

if not os.path.exists(SEARCH_ROOT):
    print(f"找不到路径: {SEARCH_ROOT}")
else:
    for d in os.listdir(SEARCH_ROOT):

        full_d_path = os.path.join(SEARCH_ROOT, d)

        if os.path.isdir(full_d_path) and d.startswith("domain_kegg_results_"):

            sample_name = d.replace("domain_kegg_results_", "")

            for domain_folder in os.listdir(full_d_path):

                if not domain_folder.startswith("domain_"):
                    continue

                domain_id = domain_folder.replace("domain_", "")
                csv_path = os.path.join(full_d_path, domain_folder, "kegg_results.csv")

                if not os.path.exists(csv_path):
                    continue

                try:
                    df = pd.read_csv(csv_path)

                    if "Term" not in df.columns:
                        continue

                    p_col = "Adjusted P-value" if "Adjusted P-value" in df.columns else "P-value"

                    df["Term"] = df["Term"].astype(str)

                    # 安全匹配 KEGG
                    target_df = df[df["Term"].str.contains(PATTERN, case=False, na=False)].copy()

                    for _, row in target_df.iterrows():

                        term = row["Term"].strip()
                        p_val = row[p_col]

                        # ===== 精确分类匹配 =====
                        category = "其他"
                        term_clean = term.lower()

                        for cat, keywords in CATEGORY_MAP.items():
                            for kw in keywords:
                                if term_clean == kw.lower():
                                    category = cat
                                    break
                            if category != "其他":
                                break

                        # ===== P值分组 =====
                        if p_val < 0.05:
                            p_group = "1. < 0.05 (显著)"
                        elif 0.05 <= p_val <= 0.2:
                            p_group = "2. 0.05 - 0.2 (趋势)"
                        else:
                            p_group = "3. > 0.2 (不显著)"

                        output_rows.append({
                            "功能分类": category,
                            "通路名称": term,
                            "P-value区间": p_group,
                            "样本与Domain": f"{sample_name}_D{domain_id}",
                            "数值P值": p_val
                        })

                except Exception as e:
                    print(f"处理 {csv_path} 出错: {e}")

# ======================
# 汇总输出
# ======================

if output_rows:

    final_df = pd.DataFrame(output_rows)

    category_order = list(CATEGORY_MAP.keys())
    p_order = [
        "1. < 0.05 (显著)",
        "2. 0.05 - 0.2 (趋势)",
        "3. > 0.2 (不显著)"
    ]

    final_df["功能分类"] = pd.Categorical(
        final_df["功能分类"],
        categories=category_order,
        ordered=True
    )

    final_df["P-value区间"] = pd.Categorical(
        final_df["P-value区间"],
        categories=p_order,
        ordered=True
    )

    # ===== 关键修复：observed=True 防垃圾空行 =====
    summary_table = (
        final_df
        .groupby(["功能分类", "通路名称", "P-value区间"], observed=True)
        .agg(
            出现次数=("样本与Domain", lambda x: len(set(x))),
            涉及样本=("样本与Domain", lambda x: ", ".join(sorted(set(x))))
        )
        .reset_index()
        .sort_values(["功能分类", "P-value区间", "通路名称"])
    )

    # ======================
    # Excel 输出
    # ======================

    output_excel = "SpaGCN_Biological_Categories_Summary.xlsx"

    with pd.ExcelWriter(output_excel, engine="xlsxwriter") as writer:

        summary_table.to_excel(writer, index=False, sheet_name="Summary")

        workbook = writer.book
        worksheet = writer.sheets["Summary"]

        # 自动列宽
        for i, col in enumerate(summary_table.columns):
            max_len = max(
                summary_table[col].astype(str).map(len).max(),
                len(col)
            ) + 2
            worksheet.set_column(i, i, min(max_len, 80))

        worksheet.freeze_panes(1, 0)

        # 动态高亮范围
        last_row = len(summary_table) + 1
        cell_range = f"C2:C{last_row}"

        red_format = workbook.add_format({"bg_color": "#FFD6D6"})
        yellow_format = workbook.add_format({"bg_color": "#FFF2CC"})

        worksheet.conditional_format(
            cell_range,
            {"type": "text", "criteria": "containing", "value": "< 0.05", "format": red_format}
        )

        worksheet.conditional_format(
            cell_range,
            {"type": "text", "criteria": "containing", "value": "0.05 - 0.2", "format": yellow_format}
        )

    print(f"\n✅ 成功生成最终汇总表: {output_excel}")
    print(summary_table.head(20))

else:
    print("未匹配到任何指定通路的结果。")
