import pandas as pd
#处理db文件
import sqlite3
#处理json文件
import json
#匹配路径用
import glob


#一共54个setting, 结果存储到列表
n = 54
results = []

for i in range(n):
    # perf文件夹路径,使用glob匹配到/PROF*路径
    dir_to_perf = glob.glob(f'sd_perf/test{i}/PROF*')[0]
    hbm_path = f'{dir_to_perf}/device_0/sqlite/hbm.db'
    util_path = f'{dir_to_perf}/mindstudio_profiler_output'
    if hbm_path and util_path:
        # 对hbm,数据格式为.db,连接到 SQLite 数据库
        conn = sqlite3.connect(hbm_path)
        hbm_df = pd.read_sql_query("SELECT * FROM HBMbwData", conn)
        hbm_df_grouped = hbm_df.groupby(hbm_df['timestamp'])['bandwidth'].sum().reset_index()
        bandwidth_avg = hbm_df_grouped['bandwidth'].mean()

        # 对npu util，数据格式为json,直接打开
        '''
        数据格式为形如：{"name": "Average", "ts": "1727167523354587.200", "pid": 480, "tid": 0, 
        "args": {"Utilization(%)": "81.229140"}, "ph": "C"}的字典组成的列表
        '''
        json_file_path = glob.glob(f'{util_path}/msprof*.json')[0]
        with open(json_file_path, 'r') as file:
            trace_data = json.load(file)
        
        # 提取 Utilization 数据
        utilization_values = []
        for event in trace_data:
            if event.get('name') == 'Average':
                utilization = event.get('args', {}).get('Utilization(%)')
                if utilization is not None:
                    utilization_values.append(float(utilization))
        
        # 计算平均值
        if utilization_values:
            npUtil_avg = sum(utilization_values) / len(utilization_values)

        #结果保存到results
        results.append({
            "n": n,
            "bindwidth_avg": bindwidth_avg,
            "npUtil_avg": npUtil_avg
        })

        # 将 results 转换为 DataFrame
        new_data_df = pd.DataFrame(results)

        # 读取现有的 CSV 文件
        csv_file_path = 'inference_results.csv'
        existing_data_df = pd.read_csv(csv_file_path)

        # 合并两个 DataFrame，按 'n' 列对齐
        merged_df = pd.merge(existing_data_df, new_data_df, on='n', how='left')

        # 写回到 CSV 文件
        merged_df.to_csv(csv_file_path, index=False)
    else:
        print(f'Test{i} has not been executed')
        break