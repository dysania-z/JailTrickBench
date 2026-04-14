import pandas as pd

# 读取原始 50 个样本
df = pd.read_csv('./harmful_bench_50.csv')

# 挑选我们选定的 ID
selected_ids = [0, 1, 2, 7, 10, 14, 16, 18, 25, 44]
df_10 = df[df['id'].isin(selected_ids)]

# 保存为新文件
df_10.to_csv('harmful_bench_10.csv', index=False)