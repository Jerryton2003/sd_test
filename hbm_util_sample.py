import pandas as pd
import sqlite3
import glob

# 初始化一个 DataFrame 来存储所有测试的结果
all_results = pd.DataFrame()

# 循环处理每个测试索引
for i in range(56):
    # 获取文件路径
    dir_to_perf = glob.glob(f'sd_perf/test{i}/PROF*')
    if not dir_to_perf:
        continue
    util_path = f'{dir_to_perf[0]}/mindstudio_profiler_output'
    hbm_path = f'{dir_to_perf[0]}/device_0/sqlite/hbm.db'
    json_file_path = glob.glob(f'{util_path}/msprof*.json')
    if not json_file_path:
        continue

    # 连接数据库并读取数据
    conn = sqlite3.connect(hbm_path)
    hbm_df = pd.read_sql_query("SELECT * FROM HBMbwData", conn)
    conn.close()

    # 按时间戳分组并求和
    hbm_df_grouped = hbm_df.groupby('timestamp')['bandwidth'].sum().reset_index().head(500)

    # 采样每第2个点
    sampled_bandwidth = hbm_df_grouped['bandwidth'][::2].reset_index(drop=True)

    # 添加到 DataFrame 中
    all_results[f'Test {i}'] = sampled_bandwidth

# 写入 CSV
all_results.to_csv('all_average_hbmbw.csv', index=False)