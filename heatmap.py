import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 读取Excel文件（需安装openpyxl：pip install openpyxl）
file_path = r"C:\Users\谈自主\Desktop\OrganelleInteractome\20250327\output\result.xlsx"  # 修改为实际文件路径
df = pd.read_excel(file_path, engine='openpyxl')

# 提取唯一通道并排序（确保包含所有可能的通道）
channels = sorted(set(df['通道1'].unique()) | set(df['通道2'].unique()))
n = len(channels)

# 初始化全零矩阵
matrix = np.zeros((n, n))

# 填充矩阵下三角（包括可能存在的上三角数据）
for _, row in df.iterrows():
    ch1, ch2 = row['通道1'], row['通道2']
    i, j = channels.index(ch1), channels.index(ch2)

    # 自动处理行列顺序，确保填充到下方
    if i > j:
        matrix[i, j] = row['重叠像素数']
    else:
        matrix[j, i] = row['重叠像素数']

# 创建上三角掩膜（隐藏上三角）
mask = np.triu(np.ones_like(matrix, dtype=bool))

# 可视化设置
plt.figure(figsize=(6.2, 5))
heatmap = sns.heatmap(
    matrix,
    mask=mask,
    vmin=0,  # 颜色最小值
    vmax=200000,  # 颜色最大值
    cbar_kws={'ticks': []},  # 关键修改
    cmap='Greens',
    linewidths=0.1,
    xticklabels=channels,
    yticklabels=channels,
)

# 优化标签显示
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, ha='right')
heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0)
plt.tight_layout()

# 保存图片（可选）
plt.savefig(r'C:\Users\谈自主\Desktop\OrganelleInteractome\20250327\output\channel_overlap_heatmap.png', dpi=300, bbox_inches='tight')

plt.show()