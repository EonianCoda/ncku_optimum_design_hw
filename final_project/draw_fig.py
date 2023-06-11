import os
from pathlib import Path
ROOT = './data/logs/'
dataset_folders = [Path(ROOT) / dataset for dataset in os.listdir(ROOT)]

experiments = {}
for dataset_folder in dataset_folders:
    experiments[dataset_folder.stem] = [dataset_folder / exp / 'logs' for exp in os.listdir(dataset_folder)]
    
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
def get_spending_secs(event) -> int:
    tags = event.Tags()['scalars']
    scalars = event.Scalars(tags[0])
    return int(scalars[-1].wall_time - scalars[0].wall_time)


exp_names = ['Adam', 'Rmsprop', 'SGD', 'Shampoo', 'Shampoo AdaGrag']
ylims = {'cifar10': {'Train/loss':[0, 2], 'Train/accuracy':[50, 100], 'Validation/accuracy':[50, 100]},
         'cifar100': {'Train/loss':[0, 5], 'Train/accuracy':[40, 100], 'Validation/accuracy':[40, 100]},
         'mnist': {'Train/loss':[0, 1], 'Train/accuracy':[90, 100], 'Validation/accuracy':[90, 100]}}


for dataset_name, exp_log_folders in experiments.items():
    metrics = defaultdict(list)
    for exp_log_folder in exp_log_folders:
        exp_names.append(exp_log_folder.parents[0].stem)
        event = EventAccumulator(exp_log_folder)
        event.Reload()
        
        spending_secs = get_spending_secs(event)
        for tag in event.Tags()['scalars']:
            values = np.array([v.value for v in event.Scalars(tag)])
            if 'loss' not in tag:
                values *= 100
            metrics[tag].append(values)

    
    colors = 'rgbcy'
    plt.figure(figsize=(25,8), tight_layout=True)
    plt.suptitle(dataset_name)
    for i, (metric_name, metric_result) in enumerate(metrics.items()):
        ax = plt.subplot(1, len(metrics),i + 1)    
        ax.set_xlabel('Epoch')
        ax.set_title(metric_name)
        ax.set_ylim(*ylims[dataset_name][metric_name])
        metric_name = metric_name.split('/')[1]
        if 'loss' not in metric_name:
            ax.set_ylabel(f'{metric_name}(%)')
        else:
            ax.set_ylabel(f'{metric_name}')
        for exp_i, (exp_name, result) in enumerate(zip(exp_names, metric_result)):
            ax.plot(range(len(result)), result, label=exp_name, color=colors[exp_i])
        ax.legend()
    plt.savefig(f'{dataset_name}.png', dpi=300)
plt.show()
    