import pandas as pd
import sqlite3
import glob

# 获取文件路径
i = 10
dir_to_perf = glob.glob(f'sd_perf/test{i}/PROF*')[0]
util_path = f'{dir_to_perf}/mindstudio_profiler_output'
hbm_path = f'{dir_to_perf}/device_0/sqlite/hbm.db'
json_file_path = glob.glob(f'{util_path}/msprof*.json')[0]

conn = sqlite3.connect(hbm_path)
hbm_df = pd.read_sql_query("SELECT * FROM HBMbwData", conn)
hbm_df_grouped = hbm_df.groupby(hbm_df['timestamp'])['bandwidth'].sum().reset_index()
hbm_timescape = pd.DataFrame(hbm_df_grouped, columns=['bandwidth'])
hbm_timescape.to_csv('average_hbmbw.csv', index=False)