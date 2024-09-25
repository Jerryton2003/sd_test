import pandas as pd
import matplotlib.pyplot as plt
import json
import glob
from collections import defaultdict

# 获取文件路径
i = 10
dir_to_perf = glob.glob(f'sd_perf/test{i}/PROF*')[0]
util_path = f'{dir_to_perf}/mindstudio_profiler_output'
json_file_path = glob.glob(f'{util_path}/msprof*.json')[0]

# 读取 JSON 数据
with open(json_file_path, 'r') as file:
    trace_data = json.load(file)

# 提取 Utilization 数据
core_utils = defaultdict(list)

for event in trace_data:
    if 'Core ' in event.get('name'):
        utilization = event.get('args', {}).get('Utilization(%)')
        if utilization is not None:
            core_utils[event.get('name')].append(float(utilization))

# 只保留 core0 到 core30
core_keys = [f'Core {i}' for i in range(31) if i != 27]
selected_cores = {key: core_utils[key][:1000] for key in core_keys if key in core_utils}
# save as csv
# 计算每个时间点的平均值
average_utilization = [sum(values) / len(values) for values in zip(*selected_cores.values())]

# 写入 CSV
average_df = pd.DataFrame(average_utilization, columns=['Average Utilization'])
average_df.to_csv('average_utilization.csv', index=False)