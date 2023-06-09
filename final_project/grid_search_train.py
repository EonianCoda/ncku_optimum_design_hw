import subprocess
import sys
import copy
import itertools
import time

BASE_CMD = [sys.executable, 'main.py']
max_concurrent_tasks = 1

if __name__ == '__main__':
    
    grid_setting = {'--dataset': ['cifar10', 'cifar100', 'mnist'],
                    '--batch_size': [2048],
                    '--epochs': [100],
                    '--model': ['resnet'],
                    '--optimizer': ['shampoo']
                    }

    grid_setting_keys = list(grid_setting.keys())
    grid_setting_values = list(grid_setting.values())
    
    tasks = []
    for combination in itertools.product(*grid_setting_values):
        settings = {grid_setting_keys[i]: combination[i] for i in range(len(grid_setting_keys))}
        cmd = copy.deepcopy(BASE_CMD)
        for k, v in settings.items():
            cmd.extend([str(k), str(v)])
        tasks.append(cmd)
    running_tasks = []
    for task in tasks:
        while len(running_tasks) >= max_concurrent_tasks:
            for running_task in running_tasks:
                if running_task.poll() is not None:
                    running_tasks.remove(running_task)
                    break
            time.sleep(1)

        process = subprocess.Popen(task)
        running_tasks.append(process)

    for process in running_tasks:
        process.wait()