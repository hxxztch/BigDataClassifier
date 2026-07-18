import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# 配置：目标生成文件大小 (例如 1GB)
TARGET_SIZE_BYTES = 1 * 1024 * 1024 * 1024 

def generate_big_csv(seed_file_path, output_file_path):
    print(f"🚀 开始基于 {os.path.basename(seed_file_path)} 生成大数据...")
    
    # 1. 读取种子数据
    df = pd.read_csv(seed_file_path)
    seed_size = os.path.getsize(seed_file_path)
    
    # 2. 计算需要复制的倍数
    multiplier = int(TARGET_SIZE_BYTES / seed_size) + 1
    print(f"📊 种子大小: {seed_size/1024/1024:.2f} MB, 目标倍数: {multiplier}x")
    
    # 3. 准备写入
    # 如果文件存在先删除
    if os.path.exists(output_file_path):
        os.remove(output_file_path)
        
    # 先写入 Header
    df.iloc[0:0].to_csv(output_file_path, index=False)
    
    # 4. 循环生成并追加写入
    with open(output_file_path, 'a', encoding='utf-8') as f:
        for i in tqdm(range(multiplier), desc="生成进度"):
            # 复制一份数据
            temp_df = df.copy()
            
            # --- 数据扰动 (增加多样性，避免过拟合) ---
            # 对数值列添加微小的随机噪声
            numeric_cols = temp_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                # 保持 ID 类列不变
                if 'id' in col.lower() or 'label' in col.lower() or temp_df[col].nunique() < 10:
                    continue
                # 添加 -1% 到 +1% 的随机波动
                noise = np.random.uniform(-0.01, 0.01, size=len(temp_df))
                temp_df[col] = temp_df[col] * (1 + noise)
            
            # --- 混淆 ID (确保 ID 唯一性) ---
            id_cols = [c for c in temp_df.columns if 'id' in c.lower()]
            for id_col in id_cols:
                if pd.api.types.is_numeric_dtype(temp_df[id_col]):
                    temp_df[id_col] = temp_df[id_col] + (len(df) * (i + 1))
                else:
                    temp_df[id_col] = temp_df[id_col] + f"_{i}"

            # 追加写入 (不含 Header)
            temp_df.to_csv(f, header=False, index=False)

    final_size = os.path.getsize(output_file_path)
    print(f"✅ 生成完成: {output_file_path}")
    print(f"📦 最终大小: {final_size/1024/1024:.2f} MB")

if __name__ == "__main__":
    # 使用示例：将 datasets/shop_shipping.csv 扩充为 big_test_data.csv
    seed = r"E:\Study\code\.vscode\BigDataClassifier\backend\datasets\ind_safety.csv"# 确保这里有你的种子文件
    output = r"E:\Study\code\.vscode\BigDataClassifier\backend\datasets\ind_safety_big.csv"
    
    if os.path.exists(seed):
        generate_big_csv(seed, output)
    else:
        print("请先下载并放入种子数据集 CSV")