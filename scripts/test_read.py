import subprocess
import re
import numpy as np
import time
import multiprocessing as mp

def monitor_gpu_util(power_list):
    """
    监控GPU和MC的利用率，并将结果存储到列表中
    """
    cmd = ['npu-smi', 'info', 'watch']
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    start_time = time.time()
    print("=========================I'm working======================")
    while true:
        # if time.time() - start_time > 10:
        #     print("Execution time exceeded 5 seconds, stopping...")
        #     break
        line = process.stdout.readline()
        print(type(line))
        if not line:
            print("error")
        if line:
            print(line)
        # 去掉空格并按空格分割
        parts = line.strip().split()
        if len(parts) > 3 and parts[2].replace('.', '', 1).isdigit():
            # 提取功率值并转换为浮点数
            print("getting power value")
            power_value = float(parts[2])
            power_list.append(power_value)

power_list = mp.Manager().list()
monitor_process = mp.Process(target=monitor_gpu_util, args=(power_list,))
monitor_process.start()
time.sleep(30)
monitor_process.terminate()
monitor_process.join()
print(power_list)