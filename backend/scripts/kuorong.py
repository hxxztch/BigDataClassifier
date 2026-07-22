# ?????? ?? ?? CSV ???????????
# ??: python backend/scripts/kuorong.py
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

TARGET_SIZE_BYTES = 1 * 1024 * 1024 * 1024  # 1GB


def generate_big_csv(seed_file_path, output_file_path):
    print(f"Generating big CSV from {os.path.basename(seed_file_path)} ...")
    df = pd.read_csv(seed_file_path)
    seed_size = os.path.getsize(seed_file_path)
    multiplier = int(TARGET_SIZE_BYTES / seed_size) + 1
    print(f"Seed size: {seed_size/1024/1024:.2f} MB, target multiplier: {multiplier}x")

    if os.path.exists(output_file_path):
        os.remove(output_file_path)

    df.iloc[0:0].to_csv(output_file_path, index=False)

    with open(output_file_path, 'a', encoding='utf-8') as f:
        for i in tqdm(range(multiplier), desc="Generating"):
            temp_df = df.copy()
            numeric_cols = temp_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if 'id' in col.lower() or 'label' in col.lower() or temp_df[col].nunique() < 10:
                    continue
                noise = np.random.uniform(-0.01, 0.01, size=len(temp_df))
                temp_df[col] = temp_df[col] * (1 + noise)

            id_cols = [c for c in temp_df.columns if 'id' in c.lower()]
            for id_col in id_cols:
                if pd.api.types.is_numeric_dtype(temp_df[id_col]):
                    temp_df[id_col] = temp_df[id_col] + (len(df) * (i + 1))
                else:
                    temp_df[id_col] = temp_df[id_col] + f"_{i}"

            temp_df.to_csv(f, header=False, index=False)

    final_size = os.path.getsize(output_file_path)
    print(f"Done: {output_file_path} ({final_size/1024/1024:.2f} MB)")


if __name__ == "__main__":
    seed = r"E:\Study\Spark???????\BigDataClassifier\backend\datasets\ind_safety.csv"
    output = r"E:\Study\Spark???????\BigDataClassifier\backend\datasets\ind_safety_big.csv"
    if os.path.exists(seed):
        generate_big_csv(seed, output)
    else:
        print("Seed file not found")
