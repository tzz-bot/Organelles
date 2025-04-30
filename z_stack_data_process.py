import pandas as pd
import numpy as np


def build_asymmetric_matrix(csv_path, output_path):
    # 读取CSV文件
    df = pd.read_csv(csv_path)  # 如果是制表符分隔

    # 获取所有唯一通道名（按字母排序）
    channels = sorted(set(df['通道1'].unique()) | set(df['通道2'].unique()))

    # 初始化空矩阵（全NaN）
    matrix = pd.DataFrame(
        np.nan,
        index=channels,
        columns=channels,
        dtype=float
    )

    # 对角线填充1
    np.fill_diagonal(matrix.values, 1.0)

    # 填充矩阵数据（基于"重叠比(通道2)"列）
    for _, row in df.iterrows():
        ch1, ch2 = row['通道1'], row['通道2']
        overlap_ratio_ch2 = row['重叠比(通道2)']
        matrix.at[ch1, ch2] = overlap_ratio_ch2

        # 反向关系（如果需要填充对称位置的原始"重叠比(通道1)"）
        overlap_ratio_ch1 = row['重叠比(通道1)']
        matrix.at[ch2, ch1] = overlap_ratio_ch1

    # 保存到CSV
    matrix.to_csv(output_path, float_format='%.6f')
    print(f"非对称矩阵已保存至: {output_path}")
    return matrix


# 使用示例
input_csv = r"C:\工作文件夹\data\莱卡显微镜图\process data\20250402\铁死亡\105  min\output\z_stack_汇总统计.csv"  # 替换为你的CSV路径
output_csv = r"C:\工作文件夹\data\莱卡显微镜图\process data\20250402\铁死亡\105  min\output\z_stack.csv"
matrix = build_asymmetric_matrix(input_csv, output_csv)

# 打印矩阵
#print("生成的非对称矩阵:")
#print(matrix)