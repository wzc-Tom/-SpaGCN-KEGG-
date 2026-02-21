import re
import os
import ast
import gseapy as gp
import pandas as pd
BASE_DIR = r"C:\Users\20199\Desktop\实验报告\生信\SpaGCN"

# 2. 定义文件名列表（你可以根据需要添加更多文件名）
FILE_LIST = ["459.txt", "459_2.txt", "848.txt", "723.txt", "1269.txt", "727.txt", "723_2.txt", "812.txt", "821.txt", "928.txt", "928_2.txt", "1101.txt", "1239.txt", "1513.txt"] 

KEGG_DB = "KEGG_2021_Human"

# ======================
# 开始遍历文件列表
# ======================
for fileName in FILE_LIST:
    INPUT_FILE = os.path.join(BASE_DIR, fileName)
    
    # 检查文件是否存在
    if not os.path.exists(INPUT_FILE):
        print(f"\n跳过文件: {fileName} (文件不存在)")
        continue

    # 为每个文件创建独立的输出总目录
    file_prefix = os.path.splitext(fileName)[0]
    OUT_BASE_DIR = f"domain_kegg_results_{file_prefix}"
    os.makedirs(OUT_BASE_DIR, exist_ok=True)

    print(f"\n{'='*30}")
    print(f"正在处理文件: {fileName}")
    print(f"{'='*30}")

    # ======================
    # Step 1: 读取文件（自动编码）
    # ======================
    with open(INPUT_FILE, "rb") as f:
        raw = f.read()

    text = ""
    for enc in ["utf-8", "utf-16", "gbk"]:
        try:
            text = raw.decode(enc)
            print(f"使用编码: {enc}")
            break
        except:
            continue
    
    if not text:
        print(f"✗ 错误: 无法解析文件 {fileName} 的编码")
        continue

    # ======================
    # Step 2: 提取 domain 基因
    # ======================
    pattern = r"SVGs for domain\s+(\d+)\s*:\s*(\[[^\]]+\])"
    matches = re.findall(pattern, text)
    domain_genes = {int(domain_id): ast.literal_eval(gene_str) for domain_id, gene_str in matches}

    print(f"共检测到 {len(domain_genes)} 个 domains")

    # ======================
    # Step 3: KEGG 富集
    # ======================
    for domain, genes in domain_genes.items():
        print(f"  分析 domain {domain} ({len(genes)} genes)...", end=" ")

        # 基因数量检查
        if len(genes) < 1:
            print("跳过 (无基因)")
            continue

        domain_dir = os.path.join(OUT_BASE_DIR, f"domain_{domain}")
        os.makedirs(domain_dir, exist_ok=True)

        try:
            enr = gp.enrichr(
                gene_list=genes,
                gene_sets=KEGG_DB,
                organism="human",
                outdir=domain_dir,
                cutoff=0.2  # 使用您设定的放宽阈值
            )

            if enr.results is not None and not enr.results.empty:
                result_df = enr.results.sort_values("Adjusted P-value")
                result_df.to_csv(os.path.join(domain_dir, "kegg_results.csv"), index=False)
                print("✓")
            else:
                print("⚠ (无富集结果)")

        except Exception as e:
            print(f"✗ (失败: {e})")

print("\n[所有文件处理完成]")