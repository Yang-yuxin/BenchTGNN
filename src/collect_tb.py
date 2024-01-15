import tensorboard as tb
import os
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Set the path to your TensorBoard log directory
logdir = 'log_tb'
experiments = {}
# Walk through the log directory and read all 'events.out.tfevents' files
for root, dirs, files in os.walk(logdir):
    if '01-08' in root:
        exp = root.split(' ')[0]
        if exp not in experiments.keys():
            experiments[exp] = {
                'Loss/Train': [],
                'AP/Val': [], 
                'MRR/Val': [],
                'MRR/Train': [],
                'AP/Train': []
                }
        for file in files:
            if 'events.out.tfevents' in file:
                event_file = osp.join(root, file)
                event_acc = EventAccumulator(event_file)
                event_acc.Reload()
                if len(event_acc.Tags()['scalars']): 
                    tags = event_acc.Tags()['scalars']
                else:
                    continue
                for tag in tags:
                    for e in event_acc.Scalars(tag):
                        experiments[exp][tag].append(e.value)

fig, axs = plt.subplots(1, 1, figsize=(15, 10))
print(experiments.keys())
for t in experiments.keys():
    if 'WIKI' in t:
        exp = experiments[t]
        for tt in exp.keys():
            exp[tt] = np.mean(np.array(exp[tt]).reshape(5, -1), axis=0)
            if len(exp[tt]) < 50:
                continue
            # if 'AP' in tt and 'Train' in tt:
            #     axs[0][0].plot(np.arange(len(exp[tt])), exp[tt], label=t.split('/')[1].split('_')[2:5])
            # elif 'MRR' in tt and 'Train' in tt:
            #     axs[0][1].plot(np.arange(len(exp[tt])), exp[tt], label=t.split('/')[1].split('_')[2:5])
            # elif 'AP' in tt and 'Val' in tt:
            #     axs[1][0].plot(np.arange(len(exp[tt])), exp[tt], label=t.split('/')[1].split('_')[2:5])
            # elif 'MRR' in tt and 'Val' in tt:
            #     axs[1][1].plot(np.arange(len(exp[tt])), exp[tt], label=t.split('/')[1].split('_')[2:5])
            # else:
            #     continue
            if 'Loss' in tt and 'Train' in tt:
                axs.plot(np.arange(len(exp[tt])), exp[tt], label=t.split('/')[1].split('_')[2:5])

axs.legend(loc="lower right")
# axs[0][1].legend(loc="lower right")
# axs[1][0].legend(loc="lower right")
# axs[1][1].legend(loc="lower right")

fig.tight_layout()
plt.savefig('WIKI_loss_training_curve.png', dpi=100)

    # for file in files:
    #     if 'events.out.tfevents' in file:
    #         event_file = os.path.join(root, file)
    #         for e in tf.compat.v1.train.summary_iterator(event_file):
    #             for v in e.summary.value:
    #                 if v.HasField('simple_value'):
    #                     print(e.wall_time, e.step, v.tag, v.simple_value)
