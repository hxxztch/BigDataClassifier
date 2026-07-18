import pandas as pd
import os

# ================= 配置 =================
FILE_NAME = 'ind_quality.csv'
OUTPUT_NAME = 'ind_quality_clean.csv'
# =======================================

def find_file(filename):
    """自动在常见位置寻找文件"""
    # 1. 检查脚本所在的目录 (backend)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path1 = os.path.join(script_dir, filename)
    
    # 2. 检查脚本上一级的 datasets 目录
    path2 = os.path.join(os.path.dirname(script_dir), 'datasets', filename)
    
    # 3. 检查当前运行命令的目录
    path3 = os.path.join(os.getcwd(), filename)

    if os.path.exists(path1): return path1
    if os.path.exists(path2): return path2
    if os.path.exists(path3): return path3
    return None

print("-" * 50)
print(f"🔍 正在寻找文件: {FILE_NAME} ...")

input_path = find_file(FILE_NAME)

if not input_path:
    print("\n❌ 错误：找不到文件！")
    print(f"请确保 '{FILE_NAME}' 位于以下任一位置：")
    print(f"1. 脚本同级目录: {os.path.dirname(os.path.abspath(__file__))}")
    print(f"2. 数据集目录:   {os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets')}")
    print(f"3. 当前运行目录: {os.getcwd()}")
    exit()

print(f"✅ 找到文件: {input_path}")
print("📖 正在读取数据...")

try:
    df = pd.read_csv(input_path)

    # === 清洗逻辑 ===
    # 1. 删除时间列
    if 'Time' in df.columns:
        print("✂️ 删除 'Time' 列")
        df.drop(columns=['Time'], inplace=True)

    # 2. 删除高缺失值的噪音列 (与训练时保持一致)
    drop_cols = ['157', '158', '220', '245', '292', '293', '358', '492', '517', '85']
    existing_drop_cols = [c for c in drop_cols if c in df.columns]
    if existing_drop_cols:
        print(f"🗑️ 删除 {len(existing_drop_cols)} 个噪音列")
        df.drop(columns=existing_drop_cols, inplace=True)

    # 3. 修正标签列 (Pass/Fail -> Quality)
    if 'Pass/Fail' in df.columns:
        print("🔄 转换标签: Pass/Fail -> Quality")
        df['Quality'] = df['Pass/Fail'].apply(lambda x: 1 if x == 1 else 0)
        df.drop(columns=['Pass/Fail'], inplace=True)
    
    # 确定保存路径（保存在源文件旁边）
    output_path = os.path.join(os.path.dirname(input_path), OUTPUT_NAME)
    
    print(f"💾 保存清洗后的文件到: {output_path}")
    df.to_csv(output_path, index=False)
    
    print("\n🎉 处理成功！")
    print(f"👉 请将文件 [{output_path}] 上传到前端进行测试。")

except Exception as e:
    print(f"\n❌ 处理过程中发生错误: {e}")