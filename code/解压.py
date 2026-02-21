import os
import zipfile
import gzip
import shutil
from pathlib import Path

def decompress_all(data_dir):
    """
    解压目录下所有的 .zip 和 .gz 文件
    """
    base_path = Path(data_dir)
    
    # 1. 处理 .zip 文件
    for zip_file in base_path.glob('**/*.zip'):
        # 创建与压缩包同名的目标目录
        extract_dir = zip_file.parent / zip_file.stem
        print(f"正在解压 ZIP: {zip_file.name} -> {extract_dir}")
        
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
        except Exception as e:
            print(f"解压 {zip_file.name} 失败: {e}")

    # 2. 处理 .gz 文件
    for gz_file in base_path.glob('**/*.gz'):
        # 移除 .gz 后缀得到原始文件名
        output_file = gz_file.parent / gz_file.stem
        
        # 避免重复解压：如果去掉.gz后的文件已存在，则跳过
        if output_file.exists():
            print(f"跳过已存在的文件: {output_file.name}")
            continue
            
        print(f"正在解压 GZ: {gz_file.name} -> {output_file.name}")
        try:
            with gzip.open(gz_file, 'rb') as f_in:
                with open(output_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        except Exception as e:
            print(f"解压 {gz_file.name} 失败: {e}")

    print("\n--- 所有任务处理完成 ---")

# --- 使用设置 ---
# 将下面的路径替换为你存放下载数据的根目录
# 例如: 'BrainEpendymoma_GSE195661/GSE195661_RAW'
target_directory = './GSE195661_RAW' 

if __name__ == "__main__":
    decompress_all(target_directory)