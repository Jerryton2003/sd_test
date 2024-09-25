import pandas as pd
import json
import glob
from collections import defaultdict

# 初始化一个 DataFrame 来存储所有测试的结果
all_results = pd.DataFrame()

# 循环处理每个测试索引
for i in range(56):
    # 获取文件路径
    dir_to_perf = glob.glob(f'sd_perf/test{i}/PROF*')
    if not dir_to_perf:
        continue
    util_path = f'{dir_to_perf[0]}/mindstudio_profiler_output'
    json_file_path = glob.glob(f'{util_path}/msprof*.json')
    if not json_file_path:
        continue

    # 读取 JSON 数据
    with open(json_file_path[0], 'r') as file:
        trace_data = json.load(file)

    # 提取 Utilization 数据
    core_utils = defaultdict(list)

    for event in trace_data:
        if 'Core ' in event.get('name'):
            utilization = event.get('args', {}).get('Utilization(%)')
            if utilization is not None:
                core_utils[event.get('name')].append(float(utilization))

    # 只保留 core0 到 core30，去掉 core27
    core_keys = [f'Core {j}' for j in range(31) if j != 27]
    selected_cores = {key: core_utils[key][:1000] for key in core_keys if key in core_utils}

    # 每五个时间点采样一个时间点
    sampled_utilization = {key: values[::5] for key, values in selected_cores.items()}

    # 计算每个时间点的平均值
    if sampled_utilization:
        average_utilization = [sum(values) / len(values) for values in zip(*sampled_utilization.values())]
        # 添加到 DataFrame 中
        all_results[f'Test {i}'] = average_utilization

# 写入 CSV
all_results.to_csv('all_average_utilizations.csv', index=False)