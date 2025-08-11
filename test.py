import numpy as np
import pandas as pd

# 假设你有一个 NumPy 矩阵
matrix = np.array([
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6],
    [0.7, 0.8, 0.9]
])

# 转换为 DataFrame（默认保留行列结构）
df = pd.DataFrame(matrix)

# 保存为 Excel 文件（保持行列结构）
df.to_excel('output.xlsx', index=False, header=False)  # 不加索引、不加列标题
