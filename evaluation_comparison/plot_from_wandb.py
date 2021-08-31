import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})

import wandb
# import numpy as np
# import pandas as pd

plt.clf()
fig = plt.figure()
api = wandb.Api()
runs = api.runs("catta/deep_lcd", {"$and": [{"config.head": "PointNet"}, {"tags": "sensitivity"}]},
                order='-config.learning_rate.value')
for run in runs:
    rot_error = run.history(keys=['Rotation Mean Error'], pandas=False)
    rot_error = [line['Rotation Mean Error'] for line in rot_error[0]]
    plt.plot(rot_error, label=f'LR: {run.config["learning_rate"]}')
plt.xlim([0, 50])
plt.ylim([0, 160])
plt.xlabel('Epoch')
plt.ylabel('Rotation Error')
# plt.show()
# fig.savefig(f'./results_for_paper/new/sensitivity_mlp_nolegend.pdf', bbox_inches = 'tight', pad_inches = 0)
plt.legend(loc='upper right')
# fig.savefig(f'./results_for_paper/new/sensitivity_mlp.pdf', bbox_inches = 'tight', pad_inches = 0)

plt.clf()
fig = plt.figure()
runs = api.runs("catta/deep_lcd", {"$and": [{"config.head": "SuperGlue"}, {"tags": "sensitivity"}]},
                order='-config.learning_rate.value')
for run in runs:
    rot_error = run.history(keys=['Rotation Mean Error'], pandas=False)
    rot_error = [line['Rotation Mean Error'] for line in rot_error[0]]
    plt.plot(rot_error, label=f'LR: {run.config["learning_rate"]}')
plt.xlim([0, 50])
plt.ylim([0, 160])
plt.xlabel('Epoch')
plt.ylabel('Rotation Error')
# plt.show()
# fig.savefig(f'./results_for_paper/new/sensitivity_uot_nolegend.pdf', bbox_inches = 'tight', pad_inches = 0)
plt.legend(loc='upper right')
# fig.savefig(f'./results_for_paper/new/sensitivity_uot.pdf', bbox_inches = 'tight', pad_inches = 0)
