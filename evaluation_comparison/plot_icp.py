import numpy as np
import pickle
import matplotlib.pyplot as plt

sequence = '360-09'

path1 = f'results_for_paper/icp_{sequence}.pickle'
path2 = f'results_for_paper/icp_{sequence}_ransac.pickle'

with open(path1, 'rb') as f:
    dict1 = pickle.load(f)

with open(path2, 'rb') as f:
    dict2 = pickle.load(f)

x = range(dict1['times_without_init'].shape[0])
plt.clf()
plt.rcParams.update({'font.size': 16})
fig = plt.figure()
plt.scatter(x, dict1['times_without_init'], s=10, label='Without Initial')
plt.scatter(x, dict2['times_with_init'], s=10, label='With Initial')
plt.xlabel("Loop Index")
plt.ylabel("Time (s)")
plt.legend(loc='upper right')
plt.ylim([0, 160])
# plt.xlim([0, dict1['times_without_init'].shape[0]])
# plt.show()
fig.savefig(f'./results_for_paper/new/icp_time_{sequence}_ransac.pdf', bbox_inches='tight', pad_inches=0)

plt.clf()
plt.rcParams.update({'font.size': 16})
fig = plt.figure()
plt.scatter(x, dict1['rmse_without_init'], s=10, label='Without Initial')
plt.scatter(x, dict2['rmse_with_init'], s=10, label='With Initial')
plt.xlabel("Loop Index")
plt.ylabel("RMSE (m)")
plt.legend(loc='upper right')
plt.ylim([0, 35])
# plt.xlim([0, dict1['times_without_init'].shape[0]])
# plt.show()
fig.savefig(f'./results_for_paper/new/icp_rmse_{sequence}_ransac.pdf', bbox_inches='tight', pad_inches=0)