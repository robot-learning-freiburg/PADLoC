import numpy as np
import csv
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns


path = '/home/cattaneo/CODES/scancontext/example/place_recognition/KITTI/00/M2DP/result/precision_recall/08'
file_list = os.listdir(path)
file_list.sort()
precision = [1.0]
recall = [0.0]
with open('/home/cattaneo/CODES/deep_lcd/real_loop_4m_08.pickle', 'rb') as f:
    real_loop = pickle.load(f)
print(real_loop)

for file in file_list:
    detected_loop = np.loadtxt(os.path.join(path, file))
    if len(detected_loop) == 0:
        continue
    tp = len(set(real_loop).intersection(set(detected_loop)))
    fp = len(set(detected_loop) - set(real_loop))
    fn = len(set(real_loop) - set(detected_loop))
    precision.append(tp/(tp+fp))
    recall.append(tp/(tp+fn))

print(file_list)
