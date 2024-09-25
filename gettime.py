import pandas as pd
#处理db文件
import sqlite3
#处理json文件
import json
#匹配路径用
import glob
from collections import defaultdict

results = []
i = 10
dir_to_perf = glob.glob(f'sd_perf/test{i}/PROF*')[0]
hbm_path = f'{dir_to_perf}/device_0/sqlite/hbm.db'
util_path = f'{dir_to_perf}/mindstudio_profiler_output'
# 对npu util，数据格式为json,直接打开
'''
数据格式为形如：{"name": "Average", "ts": "1727167523354587.200", "pid": 480, "tid": 0, 
"args": {"Utilization(%)": "81.229140"}, "ph": "C"}的字典组成的列表
'''
json_file_path = glob.glob(f'{util_path}/msprof*.json')[0]
with open(json_file_path, 'r') as file:
    trace_data = json.load(file)
core_utils = defaultdict(list)

# 提取 Utilization 数据
utilization_values = []
for event in trace_data:
    if 'Core ' in event.get('name'):
    #if event.get('name') == 'Average':
        utilization = event.get('args', {}).get('Utilization(%)')
        if utilization is not None:
            utilization_values.append(float(utilization))
            core_utils[event.get('name')].append(float(utilization))
for j in range(31):
    print(len(core_utils[f'Core {j}']))
# 计算平均值
#if utilization_values:
#    npUtil_avg = sum(utilization_values) / len(utilization_values)